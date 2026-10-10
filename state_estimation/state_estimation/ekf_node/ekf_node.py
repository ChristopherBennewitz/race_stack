#!/usr/bin/env python3
"""
EKF Node implementing an Extended Kalman Filter (filterpy) with a CasADi
symbolic mirror of the simulator's blended single-track model. Applies
sequential updates to the filter with newest sensor data.

Measurement data used:
    - IMU: [yaw, yaw_rate, ax, ay]
    - VESC: [x, y, yaw, vx, vy, yaw_rate]
    - ACKERMANN: [speed, steer_angle] (control input only)

VIO is intentionally not integrated. Sensors are configured in a single list
in __init__ (see self.sensors); adding a new sensor is one line there.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import rclpy
import tf_transformations as tft
from ackermann_msgs.msg import AckermannDriveStamped
from filterpy.kalman import ExtendedKalmanFilter
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from .models import SimulatorSingleTrackModel, VehicleModelParams
from .sensors import ImuSensor, OdomSensor


class EkfNode(Node):
    """IMU + VESC EKF; VIO is deliberately not part of this implementation."""

    def __init__(self):
        super().__init__('ekf_node')
        for name, default in (
            ('frequency', 100.0), ('odom_topic', '/early_fusion/odom'),
            ('imu_topic', '/sensors/imu/raw'), ('vesc_topic', '/odom'),
            ('cmd_topic', '/ackermann_cmd_applied'),
            # Actuator model, fitted by the simulator's sysid.
            ('steer_gain', 1.50936), ('steer_bias', 0.00332491), ('steer_latency', 0.111489),
            ('drive_gain', 3.59866), ('brake_gain', 5.00361),
            ('front_drive_frac', 0.5), ('constraint_time_constant', 0.03),
            ('blend_center', 0.6), ('slip_speed_regularizer', 0.1),
            ('imu_accel_alpha', 0.2),
            # Off by default: Cartographer publishes map->base_link and a static map->odom exists, so an
            # odom->base_link broadcast would give base_link two parents and make the SLAM pose jump.
            ('publish_tf', False),
            # Nominal values from f1tenth_parallel_sim/configs/f1tenth_car.toml.
            ('mass', 3.718), ('yaw_inertia', 0.0561),
            ('front_axle_distance', 0.17334), ('rear_axle_distance', 0.14766),
            ('cg_height', 0.074),
            ('front_B', 12.0306), ('front_C', 1.71178),
            ('front_D', 0.402674), ('front_E', 0.0),
            ('rear_B', 5.06904), ('rear_C', 1.64946),
            ('rear_D', 0.572528), ('rear_E', 0.0),
        ):
            self.declare_parameter(name, default)
        # Noise from the simulator's sysid (fit_clean/20261004-233031): gyro yaw_rate sigma 0.03,
        # odometry speed sigma 0.20. Q is the sysid parameter covariance propagated through the model
        # (rate error variance x nominal dt); x/y/yaw have no parameter dependence and use the sysid
        # pose sigmas (0.007, 0.006, 0.006).
        self.declare_parameter('R_imu', [0.1, 0.0009, 1000000.0, 1000000.0])
        self.declare_parameter('R_vesc', [1.0, 1.0, 1.0, 0.04, 100.0, 0.1])
        self.declare_parameter('Q', [4.9e-5, 3.6e-5, 3.6e-5, 5.8e-4, 1.8e-3, 6.7e-2, 2.3e-5, 6.9e-3])

        self.Hz = self.get_parameter('frequency').value
        self.publish_tf = self.get_parameter('publish_tf').value
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        # Get covariance matrices
        self.R_imu = np.diag(self.get_parameter('R_imu').value)
        self.R_vesc = np.diag(self.get_parameter('R_vesc').value)
        self.Q = np.diag(self.get_parameter('Q').value)

        # Build the model
        self.model = SimulatorSingleTrackModel(VehicleModelParams(
            m=self.get_parameter('mass').value,
            I_z=self.get_parameter('yaw_inertia').value,
            l_f=self.get_parameter('front_axle_distance').value,
            l_r=self.get_parameter('rear_axle_distance').value,
            h_cg=self.get_parameter('cg_height').value,
            Bf=self.get_parameter('front_B').value,
            Cf=self.get_parameter('front_C').value,
            Df=self.get_parameter('front_D').value,
            Ef=self.get_parameter('front_E').value,
            Br=self.get_parameter('rear_B').value,
            Cr=self.get_parameter('rear_C').value,
            Dr=self.get_parameter('rear_D').value,
            Er=self.get_parameter('rear_E').value,
            front_drive_frac=self.get_parameter('front_drive_frac').value,
            constraint_time_constant=self.get_parameter('constraint_time_constant').value,
            blend_center=self.get_parameter('blend_center').value,
            slip_speed_regularizer=self.get_parameter('slip_speed_regularizer').value,
            steer_gain=self.get_parameter('steer_gain').value,
            steer_bias=self.get_parameter('steer_bias').value,
            steer_latency=self.get_parameter('steer_latency').value,
            drive_gain=self.get_parameter('drive_gain').value,
            brake_gain=self.get_parameter('brake_gain').value,
        ), self.Q)

        # Initialize EKF
        # dim_z is just for initialization (filterpy resizes per update call,
        # so the same filter takes both the 4-vector IMU and 6-vector VESC updates)
        self.ekf = ExtendedKalmanFilter(dim_x=self.model.dim_x, dim_z=4)
        self.ekf.x = self.model.x_0
        self.ekf.P = self.model.P_0

        # Control input array
        self.ackermann_data = np.zeros(2)  # [speed, steer_angle]

        # General stuff
        self.lock = threading.Lock()
        self.last_filter_time = None
        self.is_initialized = False

        # Sensors. Each sensor owns its own subscription, buffering and EKF
        # update. Adding a new sensor is a single line here.
        self.sensors = [
            ImuSensor(self, self.get_parameter('imu_topic').value, self.R_imu,
                      self.model.Hx_imu, self.model.HJacobian_imu,
                      on_init=self.init_state_from_imu,
                      alpha=self.get_parameter('imu_accel_alpha').value),
            OdomSensor(self, 'vesc', self.get_parameter('vesc_topic').value, self.R_vesc,
                       self.model.Hx_vesc, self.model.HJacobian_vesc),
        ]

        # Control input subscriptions
        # The actuation manager's output: the rate/range-limited command the car actually receives,
        # which is the plant input the model was identified against.
        self.create_subscription(AckermannDriveStamped, self.get_parameter('cmd_topic').value, self.ackermann_callback, 1)

        # Publisher
        self.odom_pub = self.create_publisher(Odometry, self.get_parameter('odom_topic').value, 1)

        # Evaluating computational expenses
        self.predict_count = 0
        self.predict_mean_ms = 0.0
        self.predict_max_ms = 0.0

        self.tot_update_count = 0
        self.tot_update_mean_ms = 0.0
        self.tot_update_max_ms = 0.0

        self.add_on_set_parameters_callback(self.parameter_callback)

        self.timer = self.create_timer(1.0 / self.Hz, self.timer_callback)

        self.get_logger().info('[EKF Node] EKF node initialized using the CasADi single-track model (IMU + VESC only).')

    def parameter_callback(self, params):
        self.get_logger().info(f'[EKF Node] Parameters updated: {params}')
        result = SetParametersResult()
        result.successful = True  # Assume success by default
        return result

    def init_state_from_imu(self, imu_data):
        """Seed the filter state from the first IMU orientation reading."""
        self.ekf.x[2] = imu_data[0]
        self.ekf.x[5] = imu_data[1]
        self.ekf.P[2, 2] = self.R_imu[0, 0]
        self.ekf.P[5, 5] = self.R_imu[1, 1]

        self.get_logger().info(f'[EKF Node] Initial yaw set to {imu_data[0]}')
        self.get_logger().info(f'[EKF Node] Initial omega set to {imu_data[1]}')
        self.is_initialized = True

    # Callbacks
    def ackermann_callback(self, msg):
        with self.lock:
            self.ackermann_data[0] = msg.drive.speed
            self.ackermann_data[1] = msg.drive.steering_angle

    # Helper functions
    def shutdown(self):
        print("--- EKF Performance Report ---")
        print("**** Predict ****")
        print(f"Predict Steps Run: {self.predict_count}")
        print(f"Mean Execution Time: {self.predict_mean_ms:.3f} ms")
        print(f"Peak Execution Time: {self.predict_max_ms:.3f} ms")
        print("**** Update Total****")
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
        state = self.model.get_odom_state(self.ekf.x)

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
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

        self.odom_pub.publish(odom)

        if not self.publish_tf:
            return

        t = TransformStamped()
        t.header.stamp = odom.header.stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = float(state[0])
        t.transform.translation.y = float(state[1])
        t.transform.translation.z = 0.0
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])

        self.tf_broadcaster.sendTransform(t)

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
            self.get_logger().log("[EKF Node] Initial state not set yet", rclpy.logging.LoggingSeverity.WARN, once=True)
            return

        self.get_logger().log("[EKF Node] Initial state is set", rclpy.logging.LoggingSeverity.INFO, once=True)
        now = self.get_clock().now().nanoseconds / 1e9
        if self.last_filter_time is None:  # first loop
            self.last_filter_time = now
            return

        dt = now - self.last_filter_time
        if dt <= 0.0 or dt > 0.25:
            self.last_filter_time = now
            return
        self.last_filter_time = now

        t0 = time.perf_counter_ns()

        # predict
        self.model.predict(ekf=self.ekf, dt=dt, u=u)

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
