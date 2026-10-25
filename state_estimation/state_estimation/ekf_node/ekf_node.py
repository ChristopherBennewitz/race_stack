#!/usr/bin/env python3
import math
import os
import threading
import time

import numpy as np
import rclpy
import tf_transformations as tft
from ackermann_msgs.msg import AckermannDriveStamped
from filterpy.kalman import ExtendedKalmanFilter
from nav_msgs.msg import Odometry
from pbl_config import load_car_config_ros, load_pacejka_tire_config_ros, get_remote_parameter
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult

from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from . import defaults, diag_ros, models
from .diagnostics import is_moving
from .models import CommandDelay, SimulatorSingleTrackModel, VehicleModelParams
from .prediction_history import PredictionHistory
from .sensors import ImuSensor, OdomSensor, SlamPoseSensor
from .slam_pose import SlamPoseUpdater


class EkfNode(Node):
    """
    EKF Node implementing an Extended Kalman Filter. Applies sequential updates to the filter with newest sensor data.

    Measurement data used:
        - IMU: [psi, vpsi, ax, ay]
        - VESC: [x, y, psi, vx, vy, vpsi]
        - VIO: [x, y, psi, vx, vy, vpsi]
        - ACKERMANN: [accel, steer_angle]
        - SLAM pose (global filter only): [x, y, psi] of base_link in the map frame

    The node runs in one of two roles, chosen by the ``slam_pose_topic`` parameter:
        - local (empty): knows nothing of SLAM. Its odometry lives in the ``odom`` frame, is smooth and drifts,
          and is the motion prior the localization starts from. It broadcasts ``odom -> base_link``.
        - global (a pose topic): the same model and inputs, plus the SLAM pose, in the ``map`` frame. SLAM
          corrections reach it as measurements, so its estimate stays smooth where ``map -> odom`` jumps.
          It publishes odometry only; the localization owns ``map -> odom``.
    Keeping SLAM out of the local filter means the localization's prior is never made of its own output.

    Sensors are configured in a single list in __init__ (see self.sensors); adding
    a new sensor is one line there.
    """

    REQUIRED_SENSORS = ('imu', 'vesc', 'slam_pose')  # a silent one of these is a fault; the others are optional inputs

    def __init__(self):
        super().__init__('ekf_node')

        self.declare_parameter('frequency', 100)
        self.declare_parameter('model_type', 'point_mass_model')
        self.declare_parameter('odom_topic', '/state_estimation/odom')
        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('vesc_topic', '/odom')
        self.declare_parameter('vio_topic', '/basalt/odom')
        self.declare_parameter('send_transform', False)
        self.declare_parameter('frame_id', 'odom')  # the frame the published odometry is in
        # Role: empty is the local filter; a PoseStamped topic (base_link in map) makes this the global one.
        self.declare_parameter('slam_pose_topic', '')
        self.declare_parameter('floor', rclpy.Parameter.Type.STRING)
        self.declare_parameter('R_imu', rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('R_vesc', rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('R_vio', rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('Q', rclpy.Parameter.Type.DOUBLE_ARRAY)

        # Parameters of model_type 'simulator_single_track_model' (values and their provenance in defaults.py). They
        # live apart from the pbl_config car/tire config and from the shared R/Q above, which are sized for the
        # other models.
        self.declare_parameter('sim_cmd_topic', '/ackermann_cmd_applied')
        noise = defaults.NOISE_GLOBAL if self.get_parameter('slam_pose_topic').value else defaults.NOISE
        for name, default in {**defaults.MODEL, **noise}.items():
            self.declare_parameter('sim_' + name, default)
        # Global filter only. The pose stamps lag the car by sim_slam_pose_delay (the scan match, TF lookup and
        # republish), so a pose is compared with the estimate that far in the past. R is generous because the
        # localization is fed the local filter's odometry, so its pose is not independent of this filter's inputs.
        self.declare_parameter('sim_slam_pose_delay', defaults.SLAM['pose_delay'])
        self.declare_parameter('sim_R_slam_pose', defaults.SLAM['R_slam_pose'])
        self.declare_parameter('sim_slam_pose_gate', defaults.SLAM['pose_gate'])
        self.declare_parameter('sim_slam_pose_reseed', defaults.SLAM['pose_reseed'])

        self.Hz = self.get_parameter('frequency').value
        self.racecar_version = os.environ.get('RACECAR_VERSION', 'NUC6')
        self.floor = get_remote_parameter(self, 'global_parameters', 'floor', default='dubi')

        self.model_type = self.get_parameter('model_type').value
        self.ODOM_TOPIC = self.get_parameter('odom_topic').value
        self.IMU_TOPIC = self.get_parameter('imu_topic').value
        self.VESC_TOPIC = self.get_parameter('vesc_topic').value
        self.VIO_TOPIC = self.get_parameter('vio_topic').value
        
        self.send_transform = self.get_parameter('send_transform').value
        self.frame_id = self.get_parameter('frame_id').value
        self.slam_pose_topic = self.get_parameter('slam_pose_topic').value
        self.is_global = bool(self.slam_pose_topic)
        if self.is_global and self.model_type != "simulator_single_track_model":
            raise ValueError("the global filter needs model_type simulator_single_track_model")

        self.get_logger().info(f"[EKF Node] Racecar is: {self.racecar_version}")

        self.tf_broadcaster = TransformBroadcaster(self)


        # Get covariance matrices
        self.uses_sim_model = self.model_type == "simulator_single_track_model"
        prefix = 'sim_' if self.uses_sim_model else ''
        self.R_imu = np.diag(self.get_parameter(prefix + 'R_imu').value)
        if self.is_global:
            # The IMU heading is zero-referenced at its first sample, which is not the map frame, and it drifts;
            # the global filter takes its heading from SLAM and keeps only the IMU's rates and accelerations.
            self.R_imu[0, 0] = 1e6
        self.R_vesc = np.diag(self.get_parameter(prefix + 'R_vesc').value)
        self.R_vio = np.diag(self.get_parameter('R_vio').value)
        q_values = list(self.get_parameter(prefix + 'Q').value)
        if self.model_type not in ("kinematic_bicycle_model", "simulator_single_track_model") \
                and len(q_values) == 7:
            q_values = q_values[0:6]
        self.Q = np.diag(q_values)

        # Get Car Parameters. The simulator model carries its own (sim_*), so it needs no pbl_config files
        # (a car without a pacejka/<floor> folder, like NUC2, can still run it).
        self.p = {}
        if not self.uses_sim_model:
            tire_config = load_pacejka_tire_config_ros(self.racecar_version, self.floor)
            car_config = load_car_config_ros(self.racecar_version)
            self.get_logger().info(f"[EKF Node] Loaded Config with: {self.racecar_version} and {self.floor}")
            self.p = {
                # Tire Parameters
                'Bf': tire_config.Bf,
                'Cf': tire_config.Cf,
                'Df': tire_config.Df,
                'Ef': tire_config.Ef,
                'Br': tire_config.Br,
                'Cr': tire_config.Cr,
                'Dr': tire_config.Dr,
                'Er': tire_config.Er,
                'mu': tire_config.friction_coeff,

                # Car Physical Configs
                'm': car_config.m,
                'I_z': car_config.Iz,
                'l_f': car_config.lf,
                'l_r': car_config.lr,
                'h_cg': car_config.h_cg,
                'wheelbase': car_config.wheelbase
            }

        # Build the right Model Object
        if self.model_type == "point_mass_model":
            self.model = models.point_mass_model(self.Q)
        elif self.model_type == "kinematic_bicycle_model":
            self.model = models.kinematic_bicycle_model(self.p, self.Q)
        elif self.model_type == "single_track_model":
            self.model = models.single_track_model(self.p, self.Q)
        elif self.uses_sim_model:
            self.model = SimulatorSingleTrackModel(self._sim_model_params(), self.Q)
        else:
            self.get_logger().warn("[EKF Node] Unknown Model provided. Using point-mass motion model instead")
            self.model = models.point_mass_model(self.Q)

        # The single-track states are at the CoG, but the odometry and TF are for base_link, the rear axle.
        if self.uses_sim_model:
            self.cog_to_base_link = self.model.params.l_r
        elif self.model_type == "single_track_model":
            self.cog_to_base_link = self.p['l_r']
        else:
            self.cog_to_base_link = 0.0

        # Initialize EKF
        # dim_z is just for initializaiton (can change depending on update)
        self.ekf = ExtendedKalmanFilter(dim_x=self.model.dim_x, dim_z=4)
        self.ekf.x = np.array(self.model.x_0, dtype=float)
        self.ekf.P = np.array(self.model.P_0, dtype=float)

        # Control input array
        self.ackermann_data = np.zeros(2)  # [accel, steer_angle]; the simulator model takes [speed, steer_angle]
        # Command histories; the simulator model is driven by the commands as delayed by steer_delay / speed_delay.
        self.steer_history = CommandDelay(self.get_parameter('sim_steer_delay').value)
        self.speed_history = CommandDelay(self.get_parameter('sim_speed_delay').value)

        # General stuff
        self.lock = threading.Lock()
        self.last_filter_time = None
        self.is_initialized = False

        # Commanded velocity deriviation
        self.last_cmd_vel = 0.0
        self.last_cmd_time = None

        # Sensors. Each sensor owns its own subscription, buffering and EKF update.
        # Adding a new sensor is a single line here. VIO shares the full-odometry
        # measurement model with the VESC.
        # ``relative=True`` zero-references a sensor to its own first measurement,
        # so it contributes no absolute origin. The IMU needs it: its heading comes
        # from the VESC AHRS and shares no reference with the odometry sources.
        self.prediction_history = PredictionHistory(self.get_parameter('sim_imu_accel_lag').value)
        if self.uses_sim_model:
            # The global filter starts when SLAM provides its first pose, not on the IMU heading.
            imu = ImuSensor(self, self.IMU_TOPIC, self.R_imu, self._lagged_imu_measurement,
                            self.model.HJacobian_imu, on_init=None if self.is_global else self.init_state_from_imu,
                            relative=True, alpha=1.0, accel_bias=self.get_parameter('sim_imu_accel_bias').value)
        else:
            imu = ImuSensor(self, self.IMU_TOPIC, self.R_imu, self.model.Hx_imu, self.model.HJacobian_imu,
                            on_init=self.init_state_from_imu, relative=True)
        self.sensors = [
            imu,
            OdomSensor(self, 'vesc', self.VESC_TOPIC, self.R_vesc,
                       self.model.Hx_vesc, self.model.HJacobian_vesc,
                       relative=False),
            OdomSensor(self, 'vio', self.VIO_TOPIC, self.R_vio,
                       self.model.Hx_vesc, self.model.HJacobian_vesc,
                       relative=False),
        ]
        if self.is_global:
            self.slam_updater = SlamPoseUpdater(
                self.model, np.diag(self.get_parameter('sim_R_slam_pose').value), self.prediction_history,
                self.get_parameter('sim_slam_pose_delay').value, self.get_parameter('sim_slam_pose_gate').value,
                self.get_parameter('sim_slam_pose_reseed').value)
            self.sensors.append(SlamPoseSensor(self, self.slam_pose_topic, self.slam_updater,
                                               on_init=self.init_state_from_slam))

        # Control input subscriptions
        if self.uses_sim_model:
            # The actuation manager's output: the rate/range-limited command the car actually receives,
            # which is the plant input the model was identified against.
            self.create_subscription(AckermannDriveStamped, self.get_parameter('sim_cmd_topic').value,
                                     self.ackermann_callback, 1)
        else:
            self.create_subscription(AckermannDriveStamped, '/drive', self.ackermann_callback, 1)
            self.create_subscription(AckermannDriveStamped, '/ackermann_cmd', self.ackermann_callback, 1)

        # Publisher
        self.odom_pub = self.create_publisher(Odometry, self.ODOM_TOPIC, 1)
        self.diag_pub = self.create_publisher(diag_ros.DiagnosticArray, '/state_estimation/diagnostics', 10)
        self.loop_dt_max = 0.0  # longest filter step since the last diagnostics message
        self.skipped_ticks = 0  # steps dropped because dt was not usable (a stalled or restarted clock)

        # Evaulating computational expences
        self.predict_count = 0
        self.predict_mean_ms = 0.0
        self.predict_max_ms = 0.0

        self.tot_update_count = 0
        self.tot_update_mean_ms = 0.0
        self.tot_update_max_ms = 0.0

        self.add_on_set_parameters_callback(self.parameter_callback)

        self.timer = self.create_timer(1.0 / self.Hz, self.timer_callback)
        self.create_timer(1.0, self.publish_diagnostics)

        self.get_logger().info(f"[EKF Node] EKF node initialized sesfully using {self.model_type}")

    def _lagged_imu_measurement(self, x, u):
        """IMU prediction with the accelerations as the accelerometer reports them, i.e. through its lag."""
        h = self.model.Hx_imu(x, u)
        if self.prediction_history.accel is not None:
            h[2:] = self.prediction_history.accel
        return h

    def _sim_model_params(self):
        return defaults.vehicle_model_params(lambda name: self.get_parameter('sim_' + name).value)

    def parameter_callback(self, params):
        self.get_logger().info(f"[EKF Node] Parameters updated: {params}")
        result = SetParametersResult()
        result.successful = True  # Assume success by default
        return result

    def init_state_from_imu(self, imu_data):
        """Seed the filter state from the first IMU orientation reading."""
        if self.model_type in ("single_track_model", "point_mass_model", "simulator_single_track_model"):
            self.ekf.x[2] = imu_data[0]
            self.ekf.x[5] = imu_data[1]
            self.ekf.P[2, 2] = self.R_imu[0][0]
            self.ekf.P[5, 5] = self.R_imu[1][1]
        elif self.model_type == "kinematic_bicycle_model":
            self.ekf.x[4] = imu_data[0]
            self.ekf.x[5] = imu_data[1]
            self.ekf.P[4, 4] = self.R_imu[0][0]
            self.ekf.P[5, 5] = self.R_imu[1][1]

        self.get_logger().info(f"[EKF Node] Initial yaw set to {imu_data[0]}")
        self.get_logger().info(f"[EKF Node] Initial omega set to {imu_data[1]}")
        self.is_initialized = True

    def init_state_from_slam(self, pose):
        """Seed the global filter's pose from the first SLAM pose."""
        self.slam_updater.seed(self.ekf, pose[:3])
        self.get_logger().info(f"[EKF Node] Initial map pose set to {np.round(pose[:3], 3).tolist()}")
        self.is_initialized = True

    # Callbacks
    def ackermann_callback(self, msg):
        if self.uses_sim_model:
            with self.lock:
                self.ackermann_data[0] = msg.drive.speed
                self.ackermann_data[1] = msg.drive.steering_angle
                stamp = self.get_clock().now().nanoseconds * 1e-9
                self.steer_history.push(stamp, msg.drive.steering_angle)
                self.speed_history.push(stamp, msg.drive.speed)
            return
        with self.lock:
            now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            current_vel = msg.drive.speed

            if self.last_cmd_time is None:
                calculated_accel = 0.0
            else:
                dt = now - self.last_cmd_time

                if dt > 1e-4:
                    raw_accel = (current_vel - self.last_cmd_vel) / dt
                    calculated_accel = np.clip(raw_accel, -15.0, 15.0)
                else:
                    calculated_accel = self.ackermann_data[0]

            self.last_cmd_vel = current_vel
            self.last_cmd_time = now

            self.ackermann_data[0] = calculated_accel
            self.ackermann_data[1] = -msg.drive.steering_angle

    def is_moving(self):
        """Whether the NIS of this filter's sensors means anything right now (see diagnostics.is_moving)."""
        if not self.uses_sim_model:
            return True
        return is_moving(self.ekf.x[3], self.ekf.x[5])

    def publish_diagnostics(self):
        """Per-sensor arrival and live NIS, and the filter's own timing and uncertainty, on /state_estimation/diagnostics."""
        now = self.get_clock().now().nanoseconds * 1e-9
        role = 'global' if self.is_global else 'local'
        statuses = []
        for sensor in self.sensors:
            h = sensor.health
            values = {**h.values(now), 'update_mean_ms': sensor.update_mean_ms, 'update_max_ms': sensor.update_max_ms}
            # The pose's NIS is not a calibration check (its variance is deliberately generous), so only the others are judged.
            verdict = '' if sensor.name == 'slam_pose' else h.nis_verdict()
            if h.last_receive is None and sensor.name not in self.REQUIRED_SENSORS:
                level, message = 'ok', f'{sensor.name}: not used (no messages)'
            elif h.age(now) > 0.5:
                level, message = 'warn', f'{sensor.name}: no message for {min(h.age(now), 1e3):.1f} s'
            elif verdict:
                level, message = 'warn', verdict
            else:
                level, message = 'ok', f'{sensor.name}: NIS {h.nis_mean:.2f} (expected {h.dof}), counted while moving'
            statuses.append(diag_ros.status(f'ekf_{role}/{sensor.name}', level, message, values))

        with self.lock:
            P = np.array(self.ekf.P)
        values = {'initialized': float(self.is_initialized), 'loop_dt_max_ms': 1e3 * self.loop_dt_max,
                  'skipped_ticks': float(self.skipped_ticks), 'predict_mean_ms': self.predict_mean_ms,
                  'predict_max_ms': self.predict_max_ms, 'update_mean_ms': self.tot_update_mean_ms,
                  'update_max_ms': self.tot_update_max_ms, 'trace_P': float(np.trace(P))}
        if self.uses_sim_model:
            for name, i in (('x', 0), ('y', 1), ('yaw', 2), ('vx', 3), ('vy', 4), ('yaw_rate', 5)):
                values[f'std_{name}'] = float(np.sqrt(max(P[i, i], 0.0)))
        if self.is_global:
            updater = self.slam_updater
            values.update(pose_gated=float(updater.rejected), pose_reseeds=float(updater.reseeds))
        period = 1.0 / self.Hz
        if not self.is_initialized:
            level, message = 'error', 'not initialized (waiting for the first IMU or SLAM message)'
        elif self.loop_dt_max > 3 * period:
            level, message = 'warn', f'filter step took up to {1e3 * self.loop_dt_max:.0f} ms (period {1e3 * period:.0f} ms)'
        else:
            level, message = 'ok', 'running'
        statuses.append(diag_ros.status(f'ekf_{role}', level, message, values))
        self.loop_dt_max = 0.0
        self.diag_pub.publish(diag_ros.array(self.get_clock().now().to_msg(), statuses))

    # Helper functions
    def shutdown(self):
        print(f"--- EKF Performance Report ---")
        print(f"**** Predict ****")
        print(f"Predict Steps Run: {self.predict_count}")
        print(f"Mean Execution Time: {self.predict_mean_ms:.3f} ms")
        print(f"Peak Execution Time: {self.predict_max_ms:.3f} ms")
        print(f"**** Update Total****")
        print(f"Update Steps Run: {self.tot_update_count}")
        print(f"Mean Execution Time: {self.tot_update_mean_ms:.3f} ms")
        print(f"Peak Execution Time: {self.tot_update_max_ms:.3f} ms")
        for sensor in self.sensors:
            print(f"**** Update {sensor.name.upper()} ****")
            print(f"Update Steps Run: {sensor.update_count}")
            print(f"Mean Execution Time: {sensor.update_mean_ms:.3f} ms")
            print(f"Peak Execution Time: {sensor.update_max_ms:.3f} ms")
        self.get_logger().info("Shutting down...")

    def publish_odometry(self):
        state = np.array(self.model.get_odom_state(self.ekf.x), dtype=float)
        # Move the CoG pose back to the rear axle (base_link): the position along the heading, vy by omega * l_r.
        state[0] -= self.cog_to_base_link * np.cos(state[2])
        state[1] -= self.cog_to_base_link * np.sin(state[2])
        state[4] -= self.cog_to_base_link * state[5]

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = state[0]
        odom.pose.pose.position.y = state[1]

        q = tft.quaternion_from_euler(0, 0, state[2])
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        odom.twist.twist.linear.x = state[3]
        odom.twist.twist.linear.y = state[4]
        odom.twist.twist.angular.z = state[5]

        # Covariances
        pose_cov = np.zeros((6, 6))
        pose_cov[0:2, 0:2] = self.ekf.P[0:2, 0:2]
        pose_cov[0:2, 5] = self.ekf.P[0:2, 2]
        pose_cov[5, 0:2] = self.ekf.P[2, 0:2]
        pose_cov[5, 5] = self.ekf.P[2, 2]
        odom.pose.covariance = pose_cov.flatten().tolist()

        twist_cov = np.zeros((6, 6))
        twist_cov[0:2, 0:2] = self.ekf.P[3:5, 3:5]
        twist_cov[0:2, 5] = self.ekf.P[3:5, 5]
        twist_cov[5, 0:2] = self.ekf.P[5, 3:5]
        twist_cov[5, 5] = self.ekf.P[5, 5]
        odom.twist.covariance = twist_cov.flatten().tolist()


        t = TransformStamped()
        t.header.stamp = odom.header.stamp
        t.header.frame_id = self.frame_id
        t.child_frame_id = 'base_link'    # 'base_link'

        t.transform.translation.x = float(state[0])
        t.transform.translation.y = float(state[1])
        t.transform.translation.z = 0.0
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])

        if self.send_transform:
            self.tf_broadcaster.sendTransform(t)


        self.odom_pub.publish(odom)

    def timer_callback(self):
        """
        Main Loop of the EKF Node
        """
        with self.lock:
            # Get snapshot of newest measurement data, cmd inputs, and flags to use in prediction and update steps
            initialized = self.is_initialized
            u = np.copy(self.ackermann_data)
            snapshots = [(sensor, *sensor.snapshot()) for sensor in self.sensors]

        if not initialized:
            self.get_logger().log(f"[EKF Node] Initial state not set yet", rclpy.logging.LoggingSeverity.WARN, once=True)
            return

        self.get_logger().log(f"[EKF Node] Initial state is set", rclpy.logging.LoggingSeverity.INFO, once=True)
        now = self.get_clock().now().nanoseconds / 1e9
        if self.last_filter_time is None:  # first loop
            self.last_filter_time = now
            return

        dt = now - self.last_filter_time
        self.loop_dt_max = max(self.loop_dt_max, dt)
        if self.uses_sim_model:
            if dt <= 0.0 or dt > 0.25:
                self.skipped_ticks += 1
                self.last_filter_time = now
                return
        elif dt == 0:
            return
        self.last_filter_time = now

        # The simulator model integrates with RK4 in substeps, each driven by the commands as delayed by
        # steer_delay / speed_delay at the middle of that substep; the measurement updates use the command at the current time.
        predict_inputs = [(dt, u)]
        if self.uses_sim_model:
            substeps = max(1, math.ceil(dt / self.get_parameter('sim_predict_substep').value))
            h = dt / substeps
            with self.lock:
                predict_inputs = [(h, np.array([self.speed_history.at(now - dt + (k + 0.5) * h),
                                                self.steer_history.at(now - dt + (k + 0.5) * h)]))
                                  for k in range(substeps)]
                u = np.array([self.speed_history.at(now), self.steer_history.at(now)])

        t0 = time.perf_counter_ns()

        # predict
        for step, u_step in predict_inputs:
            self.model.predict(ekf=self.ekf, dt=step, u=u_step)

        t1 = time.perf_counter_ns()
        duration_ms = (t1 - t0) / 1_000_000.0
        if duration_ms > self.predict_max_ms:
            self.predict_max_ms = duration_ms

        self.predict_count += 1
        self.predict_mean_ms += (duration_ms - self.predict_mean_ms) / self.predict_count

        t2 = time.perf_counter_ns()
        # update: apply each sensor's update sequentially when it has fresh data
        for sensor, fresh, z in snapshots:
            if fresh:
                sensor.update(self.ekf, z, u)

        t3 = time.perf_counter_ns()
        duration_ms = (t3 - t2) / 1_000_000.0
        if duration_ms > self.tot_update_max_ms:
            self.tot_update_max_ms = duration_ms

        self.tot_update_count += 1
        self.tot_update_mean_ms += (duration_ms - self.tot_update_mean_ms) / self.tot_update_count

        if self.uses_sim_model:
            self.prediction_history.record(now, self.model.Hx_pose(self.ekf.x), self.model.Hx_imu(self.ekf.x, u)[2:])

        self.publish_odometry()


def main(args=None):
    rclpy.init(args=args)
    node = EkfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
