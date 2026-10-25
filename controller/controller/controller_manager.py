import math
import os
import yaml
import rclpy
from stack_master.qos import LATCHED_QOS
import numpy as np
from scipy.spatial.transform import Rotation
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener
from rclpy.parameter import Parameter
from rclpy.client import Client
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rcl_interfaces.srv import GetParameters

from ament_index_python import get_package_share_directory
from ackermann_msgs.msg import AckermannDriveStamped
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from f110_msgs.msg import (CarStateStamped, GapData, ObstacleArray, PidData, Wpnt, WpntArray)
from geometry_msgs.msg import Point, PointStamped, Pose, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Float64, String, UInt32
from steering_lookup.lookup_steer_angle import LookupSteerAngle
from visualization_msgs.msg import Marker, MarkerArray
from frenet_conversion.frenet_converter import FrenetConverter

from controller.map import MAP_Controller
from controller.pp import PP_Controller
from controller.ftg import FTG_Controller
from controller.frames import pose_in_frame
from controller.mpc.mpc_controller import base_link_to_cog
from rcl_interfaces.msg import ParameterValue, ParameterType, ParameterDescriptor, FloatingPointRange, IntegerRange
from tf_transformations import quaternion_from_euler
from stack_master.parameter_event_handler import ParameterEventHandler

from pbl_config import get_remote_parameter

from typing import List, Dict, Union


class Controller(Node):
    # Modes driven by the centerline and the state estimate alone: no global waypoints, Frenet converter or L1 tuning.
    STANDALONE_MODES = ("RL", "MPC")

    def __init__(self):
        super().__init__('controller_manager',
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)

        # remote parameters
        self.map_path = get_remote_parameter(self, '/global_parameters', 'map_path')
        self.racecar_version = get_remote_parameter(self, '/global_parameters', 'racecar_version')
        self.sim = get_remote_parameter(self, '/global_parameters', 'sim')
        # The RL policy and the MPC run without the state machine, which is the only other user of this rate.
        self.state_machine_rate = (None if self.get_parameter('mode').value in self.STANDALONE_MODES
                                   else get_remote_parameter(self, 'state_machine', 'rate_hz'))

        # variables
        self.rate = 40
        self.state = "GB_TRACK"

        self.LUT_name = self.get_parameter('LU_table').value  # name of lookup table
        self.get_logger().info(f"Using {self.LUT_name}")
        self.mode = self.get_parameter('mode').value
        self.mapping = self.get_parameter('mapping').value

        # Publishers
        self.publish_topic = '/drive'  # TODO adapt car topic name here
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.publish_topic, 10)
        self.steering_pub = self.create_publisher(Marker, 'steering', 10)
        self.lookahead_pub = self.create_publisher(Marker, 'lookahead_point', 10)
        self.trailing_pub = self.create_publisher(Marker, 'trailing_opponent_marker', 10)
        self.waypoint_pub = self.create_publisher(MarkerArray, 'my_waypoints', 10)
        self.l1_pub = self.create_publisher(Point, 'l1_distance', 10)
        self.gap_data = self.create_publisher(PidData, '/trailing/gap_data', 10)

        self.track_length = None
        self.opponent = None
        self.waypoint_array_in_map = None
        self.speed_now = None
        self.position_in_map = None
        self.position_in_map_frenet = None
        self.waypoint_safety_counter = 0

        # buffers for improved computation
        self.waypoint_array_buf = MarkerArray()
        self.markers_buf = [Marker() for _ in range(1000)]

        # controller choice
        # TODO having to set prioritize_dyn like this is really ugly.
        if self.mode == "MAP":
            self.get_logger().info("Initializing MAP  controller")
            self.init_map_controller()
            self.prioritize_dyn = self.l1_params["prioritize_dyn"]
        elif self.mode == "PP":
            self.get_logger().info("Initializing PP controller")
            self.init_pp_controller()
            self.prioritize_dyn = self.l1_params["prioritize_dyn"]
        elif self.mode == "FTG":
            self.get_logger().info("Initializing FTG controller")
            self.init_ftg_controller()
            self.prioritize_dyn = False
        elif self.mode == "RL":
            self.get_logger().info("Initializing RL controller")
            self.init_rl_controller()
            self.prioritize_dyn = False
        elif self.mode == "MPC":
            self.get_logger().info("Initializing MPC controller")
            self.init_mpc_controller()
            self.prioritize_dyn = False
        else:
            self.get_logger().error(f"Invalid mode: {self.mode}")
            return

        # Subscribers
        # self.create_subscription(Config,
        # '/l1_param_tuner/parameter_updates',self.l1_params_cb) #l1 param
        # tuning/updating
        self.create_subscription(String, '/state', self.state_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints', self.track_length_cb, LATCHED_QOS)
        self.create_subscription(ObstacleArray, '/perception/obstacles', self.obstacle_cb, 10)
        self.create_subscription(WpntArray, '/local_waypoints', self.local_waypoint_cb,
                                 10)  # waypoints (x, y, v, norm trackbound, s, kappa)
        # RL and MPC take their state from one odometry topic, in any frame: the local filter's `odom` (needs
        # map->odom) or the smooth `map` estimate (`/state_estimation/odom_map`). The other modes read the local filter's speed.
        odom_topic = '/state_estimation/odom'
        if self.mode in self.STANDALONE_MODES:
            odom_topic = self.get_parameter_or('odom_topic', Parameter('odom_topic', value=odom_topic)).value
        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)  # car speed
        self.create_subscription(
            Odometry,
            '/car_state/frenet/odom',
            self.car_state_frenet_cb,
            10)  # car frenet coordinates
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)  # lidar scan

        if self.mode in self.STANDALONE_MODES:
            # These only need the centerline and the state estimate: no global-waypoint wait, Frenet
            # converter or L1 tuning parameters.
            self.create_timer(1 / self.rate, self.control_loop)
            self.get_logger().info("Controller ready")
            return

        # Block until relevant data is here
        self.wait_for_messages()

        # Init converter
        self.converter = FrenetConverter(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2])

        # dyn params
        self.param_handler = ParameterEventHandler(self)
        self.callback_handle = self.param_handler.add_parameter_event_callback(
            callback=self.l1_param_cb,
        )
        param_dicts = [{'name': 't_clip_min',
                        'default': self.l1_params["t_clip_min"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.5, step=0.01)])},
                       {'name': 't_clip_max',
                        'default': self.l1_params["t_clip_max"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=10.0, step=0.01)])},
                       {'name': 'm_l1',
                        'default': self.l1_params["m_l1"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.0, step=0.001)])},
                       {'name': 'q_l1',
                        'default': self.l1_params["q_l1"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=-1.0, to_value=1.0, step=0.001)])},
                       {'name': 'speed_lookahead',
                        'default': self.l1_params["speed_lookahead"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.0, step=0.01)])},
                       {'name': 'lat_err_coeff',
                        'default': self.l1_params["lat_err_coeff"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.0, step=0.01)])},
                       {'name': 'acc_scaler_for_steer',
                        'default': self.l1_params["acc_scaler_for_steer"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.5, step=0.01)])},
                       {'name': 'dec_scaler_for_steer',
                        'default': self.l1_params["dec_scaler_for_steer"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.5, step=0.01)])},
                       {'name': 'start_scale_speed',
                        'default': self.l1_params["start_scale_speed"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=10.0, step=0.01)])},
                       {'name': 'end_scale_speed',
                        'default': self.l1_params["end_scale_speed"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=10.0, step=0.01)])},
                       {'name': 'downscale_factor',
                        'default': self.l1_params["downscale_factor"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=0.5, step=0.01)])},
                       {'name': 'steer_boost_per_mps',
                        'default': self.l1_params["steer_boost_per_mps"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=0.5, step=0.01)])},
                       {'name': 'steer_boost_max',
                        'default': self.l1_params["steer_boost_max"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=1.0, to_value=2.0, step=0.01)])},
                       {'name': 'speed_lookahead_for_steer',
                        'default': self.l1_params["speed_lookahead_for_steer"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=0.2, step=0.01)])},
                       {'name': 'prioritize_dyn',
                        'default': self.prioritize_dyn,
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_BOOL)},
                       {'name': 'trailing_gap',
                        'default': self.l1_params["trailing_gap"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=3.0, step=0.1)])},
                       {'name': 'trailing_p_gain',
                        'default': self.l1_params["trailing_p_gain"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=3.0, step=0.01)])},
                       {'name': 'trailing_i_gain',
                        'default': self.l1_params["trailing_i_gain"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=0.5, step=0.001)])},
                       {'name': 'trailing_d_gain',
                        'default': self.l1_params["trailing_d_gain"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.0, step=0.01)])},
                       {'name': 'blind_trailing_speed',
                        'default': self.l1_params["blind_trailing_speed"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=3.0, step=0.01)])}]
        params = self.delcare_dyn_parameters(param_dicts)
        self.set_parameters(params)

        # main loop
        self.create_timer(1 / self.rate, self.control_loop)

        self.get_logger().info("Controller ready")

    def wait_for_messages(self):
        self.get_logger().info('Controller Manager waiting for messages...')
        track_length_print = False
        waypoint_array_print = False
        state_print = False

        while not track_length_print or not waypoint_array_print or not state_print:
            rclpy.spin_once(self)
            if self.track_length is not None and not track_length_print:
                self.get_logger().info('Received track length')
                track_length_print = True
            if self.waypoint_array_in_map is not None and not waypoint_array_print:
                self.get_logger().info('Received waypoint array')
                waypoint_array_print = True
            if self.speed_now is not None and self.position_in_map is not None and self.position_in_map_frenet is not None and not state_print:
                self.get_logger().info('Received car state messages')
                state_print = True

        self.get_logger().info('All required messages received. Continuing...')

    def delcare_dyn_parameters(self, param_dicts):
        params = []
        for param_dict in param_dicts:
            param = self.declare_parameter(param_dict['name'], param_dict['default'], param_dict['descriptor'])
            params.append(param)
        return params

    def init_map_controller(self):

        # get l1 parameres
        stack_master_path = get_package_share_directory('stack_master')
        config_path = os.path.join(stack_master_path, 'config', self.racecar_version, 'l1_params.yaml')
        with open(config_path, 'r') as f:
            self.l1_params = yaml.safe_load(f)
            self.l1_params = self.l1_params['controller']['ros__parameters']

        self.create_subscription(Imu, '/sensors/imu/raw', self.imu_cb,
                                 10)  # acceleration subscriber for steer change
        self.acc_now = np.zeros(10)  # rolling buffer for acceleration

        self.map_controller = MAP_Controller(
            self.l1_params["t_clip_min"],
            self.l1_params["t_clip_max"],
            self.l1_params["m_l1"],
            self.l1_params["q_l1"],
            self.l1_params["speed_lookahead"],
            self.l1_params["lat_err_coeff"],
            self.l1_params["acc_scaler_for_steer"],
            self.l1_params["dec_scaler_for_steer"],
            self.l1_params["start_scale_speed"],
            self.l1_params["end_scale_speed"],
            self.l1_params["downscale_factor"],
            self.l1_params["steer_boost_per_mps"],
            self.l1_params["steer_boost_max"],
            self.l1_params["speed_lookahead_for_steer"],
            self.l1_params["prioritize_dyn"],
            self.l1_params["trailing_gap"],
            self.l1_params["trailing_p_gain"],
            self.l1_params["trailing_i_gain"],
            self.l1_params["trailing_d_gain"],
            self.l1_params["blind_trailing_speed"],
            self.rate,
            self.LUT_name)

    def init_pp_controller(self):

        # get l1 parameres
        stack_master_path = get_package_share_directory('stack_master')
        config_path = os.path.join(stack_master_path, 'config', self.racecar_version, 'l1_params.yaml')
        with open(config_path, 'r') as f:
            self.l1_params = yaml.safe_load(f)
            self.l1_params = self.l1_params['controller']['ros__parameters']

        # get wheelbase
        if self.sim:
            config_path = os.path.join(stack_master_path, 'config', self.racecar_version, 'car_model.yaml')
            with open(config_path, 'r') as f:
                car_params = yaml.safe_load(f)
                self.wheelbase = car_params['lr'] + car_params['lf']
        else:
            config_path = os.path.join(stack_master_path, 'config', self.racecar_version, 'vesc.yaml')
            with open(config_path, 'r') as f:
                car_params = yaml.safe_load(f)
                self.wheelbase = car_params['vesc_to_odom_node']['ros__parameters']['wheelbase']

        self.create_subscription(Imu, '/sensors/imu/raw', self.imu_cb,
                                 10)  # acceleration subscriber for steer change
        self.acc_now = np.zeros(10)  # rolling buffer for acceleration

        self.pp_controller = PP_Controller(
            self.l1_params["t_clip_min"],
            self.l1_params["t_clip_max"],
            self.l1_params["m_l1"],
            self.l1_params["q_l1"],
            self.l1_params["speed_lookahead"],
            self.l1_params["lat_err_coeff"],
            self.l1_params["acc_scaler_for_steer"],
            self.l1_params["dec_scaler_for_steer"],
            self.l1_params["start_scale_speed"],
            self.l1_params["end_scale_speed"],
            self.l1_params["downscale_factor"],
            self.l1_params["steer_boost_per_mps"],
            self.l1_params["steer_boost_max"],
            self.l1_params["speed_lookahead_for_steer"],
            self.l1_params["prioritize_dyn"],
            self.l1_params["trailing_gap"],
            self.l1_params["trailing_p_gain"],
            self.l1_params["trailing_i_gain"],
            self.l1_params["trailing_d_gain"],
            self.l1_params["blind_trailing_speed"],
            self.rate,
            self.wheelbase)

    def init_ftg_controller(self):

        # TODO fix to work without alternative value
        self.state_machine_debug = get_remote_parameter(self, 'state_machine', 'debug')
        self.state_machine_safety_radius = get_remote_parameter(self, 'state_machine', 'safety_radius')
        self.state_machine_max_lidar_dist = get_remote_parameter(self, 'state_machine', 'max_lidar_dist')
        self.state_machine_max_speed = get_remote_parameter(self, 'state_machine', 'max_speed')
        self.state_machine_range_offset = get_remote_parameter(self, 'state_machine', 'range_offset')
        self.state_machine_track_width = get_remote_parameter(self, 'state_machine', 'track_width')

        self.get_logger().info(
            f"FTG Controller parameters: {self.state_machine_debug}, {self.state_machine_safety_radius}, {self.state_machine_max_lidar_dist}, {self.state_machine_max_speed}, {self.state_machine_range_offset}, {self.state_machine_track_width}")

        self.ftg_controller = FTG_Controller(
            mapping=self.mapping,
            debug=self.state_machine_debug,
            safety_radius=self.state_machine_safety_radius,
            max_lidar_dist=self.state_machine_max_lidar_dist,
            max_speed=self.state_machine_max_speed,
            range_offset=self.state_machine_range_offset,
            track_width=self.state_machine_track_width)

    def init_rl_controller(self):
        # torch is an optional dependency, so only import the RL code when this mode is selected.
        try:
            from controller.rl.rl_controller import RL_Controller
        except ImportError as e:
            raise RuntimeError("RL mode needs PyTorch; build with `make build RL=cpu|gpu`") from e

        checkpoint = self.get_parameter_or('checkpoint', Parameter('checkpoint', value='')).value
        if not checkpoint:
            raise ValueError("parameter 'checkpoint' is required in RL mode")
        device = self.get_parameter_or('device', Parameter('device', value='cpu')).value
        direction = self.get_parameter_or('direction', Parameter('direction', value='ccw')).value
        self.rl_max_odom_age = self.get_parameter_or('max_odom_age_s', Parameter('max_odom_age_s', value=0.1)).value
        # The odometry is for base_link (the rear axle); the policy was trained on the CoG pose (l_r in f1tenth_car.toml).
        self.rl_cog_offset = self.get_parameter_or('cog_offset_m', Parameter('cog_offset_m', value=0.14766)).value

        self.rl_controller = RL_Controller(checkpoint, device, direction, self.get_logger())
        self.rate = self.rl_controller.control_hz
        self.odom_msg = None
        # The state estimate is in the odom frame but the centerline in map: Cartographer's map->odom bridges them.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # The stack's centerline publisher uses the default (volatile) QoS and republishes every 10 s. A
        # TRANSIENT_LOCAL subscription is incompatible with a volatile publisher and would never connect, so
        # use the default reliable/volatile profile, which also matches a latched publisher (without its history).
        self.create_subscription(WpntArray, '/centerline_waypoints', self.centerline_cb, QoSProfile(depth=1))

    def init_mpc_controller(self):
        from controller.mpc.mpc_controller import MPC_Controller

        direction = self.get_parameter_or('direction', Parameter('direction', value='ccw')).value
        self.rl_max_odom_age = self.get_parameter_or('max_odom_age_s', Parameter('max_odom_age_s', value=0.1)).value
        self.mpc_controller = MPC_Controller(direction, self.get_logger())
        self.rate = self.mpc_controller.control_hz
        # The MPC's state is at the centre of gravity; the odometry is for base_link (the rear axle).
        self.mpc_cog_offset = self.get_parameter_or(
            'cog_offset_m', Parameter('cog_offset_m', value=self.mpc_controller.cog_offset)).value
        self.odom_msg = None
        self.applied_cmd = None  # latest (steer, speed, receive time) the actuation manager reported
        # The state estimate is in the odom frame but the centerline in map: Cartographer's map->odom bridges them.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Same QoS reasoning as in init_rl_controller.
        self.create_subscription(WpntArray, '/centerline_waypoints', self.mpc_centerline_cb, QoSProfile(depth=1))
        # The rate-limited command the car actually received: the MPC's rate-limit anchor and steering history.
        self.create_subscription(AckermannDriveStamped, '/ackermann_cmd_applied', self.applied_cmd_cb, 1)
        # The map-frame estimate counts the times it stepped (its correction was set from a SLAM pose); the plan
        # lives in the old frame, so it is dropped. Latched, so the first value is only a baseline.
        self.estimate_jumps = None
        self.create_subscription(
            UInt32, '/state_estimation/odom_map_jumps', self.estimate_jumps_cb,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
        self.odom_age = 0.0
        self.diag_pub = self.create_publisher(DiagnosticArray, '/controller/diagnostics', 10)
        self.create_timer(0.1, self.publish_mpc_diagnostics)

    #############
    # CALLBACKS #
    #############

    def estimate_jumps_cb(self, msg: UInt32):
        if self.estimate_jumps is not None and msg.data != self.estimate_jumps:
            self.get_logger().warn("State estimate jumped; dropping the MPC plan")
            self.mpc_controller.estimate_jumped()
        self.estimate_jumps = msg.data

    def publish_mpc_diagnostics(self):
        """Solver timing, constraint slack and the plan's one-step prediction error on /controller/diagnostics (10 Hz)."""
        values = self.mpc_controller.diagnostics()
        level, message = self.mpc_controller.verdict(values)
        status = DiagnosticStatus()
        status.level = {'ok': DiagnosticStatus.OK, 'warn': DiagnosticStatus.WARN, 'error': DiagnosticStatus.ERROR}[level]
        status.name = 'mpc'
        status.message = message
        status.hardware_id = 'controller'
        status.values = [KeyValue(key=k, value=f'{v:.6g}') for k, v in values.items()]
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.status = [status]
        self.diag_pub.publish(msg)

    def mpc_centerline_cb(self, data: WpntArray):
        if self.mpc_controller.has_track or not data.wpnts:
            return
        self.mpc_controller.set_centerline(
            [(w.x_m, w.y_m) for w in data.wpnts], [w.d_left for w in data.wpnts], [w.d_right for w in data.wpnts])
        self.get_logger().info(f"Received centerline with {len(data.wpnts)} points")

    def applied_cmd_cb(self, data: AckermannDriveStamped):
        self.applied_cmd = (data.drive.steering_angle, data.drive.speed, self.get_clock().now())

    def centerline_cb(self, data: WpntArray):
        if self.rl_controller.centerline is not None or not data.wpnts:
            return
        self.rl_controller.set_centerline(
            [(w.x_m, w.y_m) for w in data.wpnts], [w.d_left for w in data.wpnts], [w.d_right for w in data.wpnts])
        self.get_logger().info(f"Received centerline with {len(data.wpnts)} points")

    def l1_param_cb(self, parameter_event):
        """
        Notices the change in the parameters and alters the spline params accordingly
        """

        if parameter_event.node != self.get_fully_qualified_name() or self.mode == "FTG":
            return

        if self.mode == "MAP":
            self.map_controller.t_clip_min = self.get_parameter('t_clip_min').value
            self.map_controller.t_clip_max = self.get_parameter('t_clip_max').value
            self.map_controller.m_l1 = self.get_parameter('m_l1').value
            self.map_controller.q_l1 = self.get_parameter('q_l1').value
            self.map_controller.speed_lookahead = self.get_parameter('speed_lookahead').value
            self.map_controller.lat_err_coeff = self.get_parameter('lat_err_coeff').value
            self.map_controller.acc_scaler_for_steer = self.get_parameter('acc_scaler_for_steer').value
            self.map_controller.dec_scaler_for_steer = self.get_parameter('dec_scaler_for_steer').value
            self.map_controller.start_scale_speed = self.get_parameter('start_scale_speed').value
            self.map_controller.end_scale_speed = self.get_parameter('end_scale_speed').value
            self.map_controller.downscale_factor = self.get_parameter('downscale_factor').value
            self.map_controller.steer_boost_per_mps = self.get_parameter('steer_boost_per_mps').value
            self.map_controller.steer_boost_max = self.get_parameter('steer_boost_max').value
            self.map_controller.speed_lookahead_for_steer = self.get_parameter('speed_lookahead_for_steer').value
            self.map_controller.prioritize_dyn = self.get_parameter('prioritize_dyn').value
            self.map_controller.trailing_gap = self.get_parameter('trailing_gap').value
            self.map_controller.trailing_p_gain = self.get_parameter('trailing_p_gain').value
            self.map_controller.trailing_i_gain = self.get_parameter('trailing_i_gain').value
            self.map_controller.trailing_d_gain = self.get_parameter('trailing_d_gain').value
            self.map_controller.blind_trailing_speed = self.get_parameter('blind_trailing_speed').value
        elif self.mode == "PP":
            self.pp_controller.t_clip_min = self.get_parameter('t_clip_min').value
            self.pp_controller.t_clip_max = self.get_parameter('t_clip_max').value
            self.pp_controller.m_l1 = self.get_parameter('m_l1').value
            self.pp_controller.q_l1 = self.get_parameter('q_l1').value
            self.pp_controller.speed_lookahead = self.get_parameter('speed_lookahead').value
            self.pp_controller.lat_err_coeff = self.get_parameter('lat_err_coeff').value
            self.pp_controller.acc_scaler_for_steer = self.get_parameter('acc_scaler_for_steer').value
            self.pp_controller.dec_scaler_for_steer = self.get_parameter('dec_scaler_for_steer').value
            self.pp_controller.start_scale_speed = self.get_parameter('start_scale_speed').value
            self.pp_controller.end_scale_speed = self.get_parameter('end_scale_speed').value
            self.pp_controller.downscale_factor = self.get_parameter('downscale_factor').value
            self.pp_controller.steer_boost_per_mps = self.get_parameter('steer_boost_per_mps').value
            self.pp_controller.steer_boost_max = self.get_parameter('steer_boost_max').value
            self.pp_controller.speed_lookahead_for_steer = self.get_parameter('speed_lookahead_for_steer').value
            self.pp_controller.prioritize_dyn = self.get_parameter('prioritize_dyn').value
            self.pp_controller.trailing_gap = self.get_parameter('trailing_gap').value
            self.pp_controller.trailing_p_gain = self.get_parameter('trailing_p_gain').value
            self.pp_controller.trailing_i_gain = self.get_parameter('trailing_i_gain').value
            self.pp_controller.trailing_d_gain = self.get_parameter('trailing_d_gain').value
            self.pp_controller.blind_trailing_speed = self.get_parameter('blind_trailing_speed').value
        self.get_logger().info("Updated parameters")

    def scan_cb(self, data: LaserScan):
        self.scan = data

    def track_length_cb(self, data: WpntArray):
        self.track_length = data.wpnts[-1].s_m
        self.waypoints = np.array([[wpnt.x_m, wpnt.y_m, wpnt.psi_rad] for wpnt in data.wpnts])

    def obstacle_cb(self, data: ObstacleArray):
        if len(data.obstacles) > 0 and \
                self.position_in_map_frenet is not None and len(self.position_in_map_frenet) and \
                self.track_length is not None:

            self.opponent_s = None
            static_flag = False  # If we have a Static and a Dynamic obstacle we prefer the dynamic
            closest_opp = self.track_length
            for obstacle in data.obstacles:
                opponent_dist = (obstacle.s_start - self.position_in_map_frenet[0]) % self.track_length
                if opponent_dist < closest_opp or (static_flag and not obstacle.is_static):
                    closest_opp = opponent_dist
                    opponent_static = obstacle.is_static
                    opponent_s = obstacle.s_center
                    opponent_d = obstacle.d_center
                    opponent_vs = obstacle.vs
                    opponent_visible = obstacle.is_visible
                    if opponent_static:
                        static_flag = self.prioritize_dyn  # Chosen Obstacle is static
                    else:
                        static_flag = False  # Chosen obstacle is dynamic
                    self.opponent = [opponent_s, opponent_d, opponent_vs, opponent_static, opponent_visible]
        else:
            self.opponent = None

    def odom_cb(self, data: Odometry):
        if self.mode in self.STANDALONE_MODES:
            self.odom_msg = data
            return
        self.speed_now = data.twist.twist.linear.x
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        rot = Rotation.from_quat([data.pose.pose.orientation.x, data.pose.pose.orientation.y,
                                    data.pose.pose.orientation.z, data.pose.pose.orientation.w])
        rot_euler = rot.as_euler('xyz', degrees=False)
        theta = rot_euler[2]
        self.position_in_map = np.array([x, y, theta])[np.newaxis]

    def local_waypoint_cb(self, data: WpntArray):
        self.waypoint_list_in_map = []
        for waypoint in data.wpnts:
            waypoint_in_map = [waypoint.x_m, waypoint.y_m]
            speed = waypoint.vx_mps
            if waypoint.d_right + waypoint.d_left != 0:
                self.waypoint_list_in_map.append([waypoint_in_map[0],
                                                  waypoint_in_map[1],
                                                  speed,
                                                  min(waypoint.d_left,
                                                      waypoint.d_right) / (waypoint.d_right + waypoint.d_left),
                                                  waypoint.s_m,
                                                  waypoint.kappa_radpm,
                                                  waypoint.psi_rad,
                                                  waypoint.ax_mps2])
            else:
                self.waypoint_list_in_map.append([waypoint_in_map[0],
                                                  waypoint_in_map[1],
                                                  speed,
                                                  0,
                                                  waypoint.s_m,
                                                  waypoint.kappa_radpm,
                                                  waypoint.psi_rad,
                                                  waypoint.ax_mps2])
        self.waypoint_array_in_map = np.array(self.waypoint_list_in_map)
        self.waypoint_safety_counter = 0

    def imu_cb(self, data):
        # save acceleration in a rolling buffer
        self.acc_now[1:] = self.acc_now[:-1]
        self.acc_now[0] = -data.linear_acceleration.y  # vesc is rotated 90 deg, so (-acc_y) == (long_acc)

    def car_state_frenet_cb(self, data: Odometry):
        s = data.pose.pose.position.x
        d = data.pose.pose.position.y
        vs = data.twist.twist.linear.x
        vd = data.twist.twist.linear.y
        self.position_in_map_frenet = np.array([s, d, vs, vd])

    def state_cb(self, data):
        self.state = data.data

    def map_cycle(self):
        speed, acceleration, jerk, steering_angle, L1_point, L1_distance, idx_nearest_waypoint = self.map_controller.main_loop(self.state,
                                                                                                                               self.position_in_map,
                                                                                                                               self.waypoint_array_in_map,
                                                                                                                               self.speed_now,
                                                                                                                               self.opponent,
                                                                                                                               self.position_in_map_frenet,
                                                                                                                               self.acc_now,
                                                                                                                               self.track_length)
        self.set_lookahead_marker(L1_point, 100)
        self.visualize_steering(steering_angle)
        self.visualize_trailing_opponent()
        self.l1_pub.publish(Point(x=float(idx_nearest_waypoint), y=L1_distance))

        self.waypoint_safety_counter += 1
        print(self.waypoint_safety_counter, self.rate, self.state_machine_rate)
        if self.waypoint_safety_counter >= self.rate / self.state_machine_rate * 10:  # we can use the same waypoints for 5 cycles
            self.get_logger().warning("[controller_manager] Received no local wpnts. STOPPING!!")
            speed = 0
            steering_angle = 0
        self.map_controller.flag1 = False

        # Publish PID data of the Trailing controller for tuning
        if self.map_controller.flag1:
            pid_msg = self.create_pid_msg(self.map_controller.gap_should,
                                          self.map_controller.gap,
                                          self.map_controller.gap_error,
                                          self.map_controller.v_diff,
                                          self.map_controller.i_gap,
                                          self.map_controller.trailing_command)
            self.gap_data.publish(pid_msg)

        return speed, steering_angle

    def pp_cycle(self):
        speed, acceleration, jerk, steering_angle, L1_point, L1_distance, idx_nearest_waypoint = self.pp_controller.main_loop(self.state,
                                                                                                                              self.position_in_map,
                                                                                                                              self.waypoint_array_in_map,
                                                                                                                              self.speed_now,
                                                                                                                              self.opponent,
                                                                                                                              self.position_in_map_frenet,
                                                                                                                              self.acc_now,
                                                                                                                              self.track_length)
        self.set_lookahead_marker(L1_point, 100)
        self.visualize_steering(steering_angle)
        self.visualize_trailing_opponent()
        self.l1_pub.publish(Point(x=float(idx_nearest_waypoint), y=L1_distance))

        self.waypoint_safety_counter += 1
        if self.waypoint_safety_counter >= self.rate / self.state_machine_rate * 10:  # we can use the same waypoints for 5 cycles
            self.get_logger().warning("[controller_manager] Received no local wpnts. STOPPING!!")
            speed = 0
            steering_angle = 0
        self.pp_controller.flag1 = False

        # Publish PID data of the Trailing controller for tuning
        if self.pp_controller.flag1:
            pid_msg = self.create_pid_msg(self.pp_controller.gap_should,
                                          self.pp_controller.gap,
                                          self.pp_controller.gap_error,
                                          self.pp_controller.v_diff,
                                          self.pp_controller.i_gap,
                                          self.pp_controller.trailing_command)
            self.gap_data.publish(pid_msg)
        return speed, steering_angle

    def ftg_cycle(self):
        speed, steer = self.ftg_controller.process_lidar(self.scan.ranges)
        self.get_logger().warning("[STATE MACHINE] FTGONLY!!!")
        return speed, steer

    def odom_in_map(self):
        """The latest odometry as (x, y, yaw, twist) in the map frame, or None while it is missing or stale."""
        if self.odom_msg is None:
            return None
        age = (self.get_clock().now() - rclpy.time.Time.from_msg(self.odom_msg.header.stamp)).nanoseconds * 1e-9
        self.odom_age = age
        if age > self.rl_max_odom_age:
            self.get_logger().warn(f"Stale odometry ({age:.2f}s old); not driving", throttle_duration_sec=1.0)
            return None

        pose, twist = self.odom_msg.pose.pose, self.odom_msg.twist.twist
        q = pose.orientation
        x, y = pose.position.x, pose.position.y
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        frame = self.odom_msg.header.frame_id
        if frame and frame != 'map':
            try:
                t = self.tf_buffer.lookup_transform('map', frame, rclpy.time.Time()).transform
            except TransformException as e:
                self.get_logger().warn(f"No map->{frame} transform yet; not driving: {e}", throttle_duration_sec=1.0)
                return None
            r = t.rotation
            frame_yaw = math.atan2(2.0 * (r.w * r.z + r.x * r.y), 1.0 - 2.0 * (r.y * r.y + r.z * r.z))
            x, y, yaw = pose_in_frame(x, y, yaw, t.translation.x, t.translation.y, frame_yaw)
        return x, y, yaw, twist

    def rl_cycle(self):
        """Returns (speed, steer), or None while the centerline/odometry are missing or stale."""
        if self.rl_controller.centerline is None:
            return None
        state = self.odom_in_map()
        if state is None:
            return None
        x, y, yaw, twist = state
        x += self.rl_cog_offset * math.cos(yaw)
        y += self.rl_cog_offset * math.sin(yaw)
        return self.rl_controller.main_loop(x, y, yaw, twist.linear.x, twist.angular.z)

    def mpc_cycle(self):
        """Returns (speed, steer), or None while the centerline/odometry are missing or stale."""
        if not self.mpc_controller.has_track:
            return None
        state = self.odom_in_map()
        if state is None:
            self.mpc_controller.reset()  # the command history no longer matches what the car did
            return None
        x, y, yaw, twist = state
        self.mpc_controller.note_odom_age(self.odom_age)
        # The EKF's velocity is that of base_link, so its sideways part is the CoG's minus l_r * yaw_rate.
        x, y, vy = base_link_to_cog(x, y, yaw, twist.linear.y, twist.angular.z, self.mpc_cog_offset)

        # The command issued last cycle has now been applied for one period.
        if self.mpc_controller.last_cmd is not None:
            steer, speed = self.mpc_controller.last_cmd
            if self.applied_cmd is not None and (self.get_clock().now() - self.applied_cmd[2]).nanoseconds * 1e-9 < 0.2:
                steer, speed = self.applied_cmd[:2]
            self.mpc_controller.report_applied(steer, speed)
        return self.mpc_controller.main_loop(x, y, yaw, twist.linear.x, vy, twist.angular.z)

    def create_pid_msg(self, should, actual, error, d_value, i_value, input):
        pid_msg = PidData()
        pid_msg.header.stamp = self.get_clock().now().to_msg()
        pid_msg.should = should
        pid_msg.actual = actual
        pid_msg.error = error
        pid_msg.d_value = d_value
        pid_msg.i_value = i_value
        pid_msg.input = input
        return pid_msg

    #############
    # MAIN LOOP #
    #############
    def control_loop(self):
        if self.mode == "MAP":
            speed, steer = self.map_cycle()
        elif self.mode == "PP":
            speed, steer = self.pp_cycle()
        elif self.mode == "FTG":
            speed, steer = self.ftg_cycle()
        elif self.mode == "RL":
            cmd = self.rl_cycle()
            if cmd is None:
                return
            speed, steer = cmd
        elif self.mode == "MPC":
            cmd = self.mpc_cycle()
            if cmd is None:
                return
            speed, steer = cmd

        ack_msg = AckermannDriveStamped()
        ack_msg.header.stamp = self.get_clock().now().to_msg()
        ack_msg.header.frame_id = 'base_link'
        ack_msg.drive.steering_angle = steer
        ack_msg.drive.speed = speed
        self.drive_pub.publish(ack_msg)

    ############################################ MSG CREATION############################################
    # visualization utilities
    def visualize_steering(self, theta):

        quaternions = quaternion_from_euler(0, 0, theta)

        lookahead_marker = Marker()
        lookahead_marker.header.frame_id = "car_state/base_link"
        lookahead_marker.header.stamp = self.get_clock().now().to_msg()
        lookahead_marker.type = 0
        lookahead_marker.id = 50
        lookahead_marker.scale.x = 0.6
        lookahead_marker.scale.y = 0.05
        lookahead_marker.scale.z = 0.01
        lookahead_marker.color.r = 1.0
        lookahead_marker.color.g = 0.0
        lookahead_marker.color.b = 0.0
        lookahead_marker.color.a = 1.0
        lookahead_marker.pose.position.x = 0.0
        lookahead_marker.pose.position.y = 0.0
        lookahead_marker.pose.position.z = 0.0
        lookahead_marker.pose.orientation.x = quaternions[0]
        lookahead_marker.pose.orientation.y = quaternions[1]
        lookahead_marker.pose.orientation.z = quaternions[2]
        lookahead_marker.pose.orientation.w = quaternions[3]
        self.steering_pub.publish(lookahead_marker)

    def set_waypoint_markers(self, waypoints):
        wpnt_id = 0

        for waypoint in waypoints:
            waypoint_marker = self.markers_buf[wpnt_id]
            waypoint_marker.header.frame_id = "map"
            waypoint_marker.header.stamp = self.get_clock().now().to_msg()
            waypoint_marker.type = 2
            waypoint_marker.scale.x = 0.1
            waypoint_marker.scale.y = 0.1
            waypoint_marker.scale.z = 0.1
            waypoint_marker.color.r = 0.0
            waypoint_marker.color.g = 0.0
            waypoint_marker.color.b = 1.0
            waypoint_marker.color.a = 1.0
            waypoint_marker.pose.position.x = waypoint[0]
            waypoint_marker.pose.position.y = waypoint[1]
            waypoint_marker.pose.position.z = 0.0
            waypoint_marker.pose.orientation.x = 0.0
            waypoint_marker.pose.orientation.y = 0.0
            waypoint_marker.pose.orientation.z = 0.0
            waypoint_marker.pose.orientation.w = 1.0
            waypoint_marker.id = wpnt_id + 1
            wpnt_id += 1
        self.waypoint_array_buf.markers = self.markers_buf[:wpnt_id]
        self.waypoint_pub.publish(self.waypoint_array_buf)

    def set_lookahead_marker(self, lookahead_point, id):
        lookahead_marker = Marker()
        lookahead_marker.header.frame_id = "map"
        lookahead_marker.header.stamp = self.get_clock().now().to_msg()
        lookahead_marker.type = 2
        lookahead_marker.id = id
        lookahead_marker.scale.x = 0.15
        lookahead_marker.scale.y = 0.15
        lookahead_marker.scale.z = 0.15
        lookahead_marker.color.r = 1.0
        lookahead_marker.color.g = 0.0
        lookahead_marker.color.b = 0.0
        lookahead_marker.color.a = 1.0
        lookahead_marker.pose.position.x = lookahead_point[0]
        lookahead_marker.pose.position.y = lookahead_point[1]
        lookahead_marker.pose.position.z = 0.0
        lookahead_marker.pose.orientation.x = 0.0
        lookahead_marker.pose.orientation.y = 0.0
        lookahead_marker.pose.orientation.z = 0.0
        lookahead_marker.pose.orientation.w = 1.0
        self.lookahead_pub.publish(lookahead_marker)

    def visualize_trailing_opponent(self):
        if (self.state == "TRAILING" and (self.opponent is not None)):
            on = True
        else:
            on = False
        opponent_marker = Marker()
        opponent_marker.header.frame_id = "map"
        opponent_marker.header.stamp = self.get_clock().now().to_msg()
        opponent_marker.type = 2
        opponent_marker.scale.x = 0.3
        opponent_marker.scale.y = 0.3
        opponent_marker.scale.z = 0.3
        opponent_marker.color.r = 1.0
        opponent_marker.color.g = 0.0
        opponent_marker.color.b = 0.0
        opponent_marker.color.a = 1.0
        if self.opponent is not None:
            pos = self.converter.get_cartesian([self.opponent[0]], [self.opponent[1]])
            opponent_marker.pose.position.x = pos[0][0]
            opponent_marker.pose.position.y = pos[1][0]
            opponent_marker.pose.position.z = 0.0

        opponent_marker.pose.orientation.x = 0.0
        opponent_marker.pose.orientation.y = 0.0
        opponent_marker.pose.orientation.z = 0.0
        opponent_marker.pose.orientation.w = 1.0
        if not on:
            opponent_marker.action = Marker.DELETE
        self.trailing_pub.publish(opponent_marker)


def main():
    rclpy.init()
    node = Controller()
    rclpy.spin(node)
    rclpy.shutdown()
