import os
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _racecar_version(context):
    """The launch argument, or RACECAR_VERSION when the including launch file does not pass one."""
    car = LaunchConfiguration('racecar_version').perform(context)
    return car if car not in ('', 'none') else os.environ.get('RACECAR_VERSION', '')


def _car_ekf_yaml(car):
    """(path or None, log message) of stack_master/config/<car>/ekf.yaml, the simulator-model EKF parameters of one car."""
    try:
        path = os.path.join(get_package_share_directory('stack_master'), 'config', car, 'ekf.yaml')
    except PackageNotFoundError:
        return None, '[ekf_node_launch] stack_master is not installed: simulator-model EKF uses the defaults.py values'
    if car and os.path.isfile(path):
        return path, f'[ekf_node_launch] simulator-model EKF parameters of {car} from {path}'
    return None, (f'[ekf_node_launch] no ekf.yaml for racecar "{car}" in stack_master/config: the simulator-model EKF '
                  'uses the defaults.py values, which are the NUC2 fit')


def generate_launch_description():

    # Both nodes are declared, the condition picks one at launch time (custom_ekf is only resolved then)
    custom_ekf = LaunchConfiguration('custom_ekf')

    # The car's own ekf.yaml (sim_* parameters of the simulator model) is added after the literals below, so it wins. It
    # needs the racecar_version, which is only resolved at launch time, hence the function.
    def custom_ekf_node_for_car(context):
        car_yaml, note = _car_ekf_yaml(_racecar_version(context))
        return [LogInfo(msg=note), Node(
                        condition=IfCondition(custom_ekf),
                        package='state_estimation',
                        executable='ekf_node',
                        name='ekf_node',
                        output='screen',
                        emulate_tty=True,
                        parameters=[{
                            # Covariance matrices of the other models. Kept as plain literals (not launch args)
                            # since ROS2 launch arguments are always strings and can't cleanly carry float arrays.
                            # Q needs to match the state-dimension of the selected model.
                            'frequency': 100,
                            'racecar_version': LaunchConfiguration('racecar_version'),
                            'model_type': LaunchConfiguration('model_type'),
                            'odom_topic': LaunchConfiguration('odom_topic'),
                            'imu_topic': LaunchConfiguration('imu_topic'),
                            'vesc_topic': LaunchConfiguration('odom_vesc_topic'),
                            'vio_topic': '/basalt/odom',
                            'floor': LaunchConfiguration('floor'),
                            # robot_localization publishes odom->base_link (publish_tf in ekf.yaml); Cartographer
                            # (provide_odom_frame = false) needs it to publish map->odom, so this node must too.
                            'send_transform': True,
                            'R_imu': [0.1, 0.1, 1000000.0, 1000000.0],
                            'R_vesc': [1, 1, 1, 1, 100, 0.1],
                            'R_vio': [0.1, 0.1, 0.1, 0.1, 0.1, 1000000.0],
                            'Q': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                        }] + ([car_yaml] if car_yaml else []),
                    )]

    # The controllers' map-frame estimate: the local filter's odometry moved into the map frame by a slowly corrected
    # map -> odom. Works with either local filter, as it only needs the local odometry and the pose.
    map_odom_node = Node(
                        condition=IfCondition(LaunchConfiguration('global_ekf')),
                        package='state_estimation',
                        executable='map_odom_node',
                        name='map_odom_node',
                        output='screen',
                        emulate_tty=True,
                        parameters=[{
                            'odom_topic': LaunchConfiguration('odom_topic'),
                            'slam_pose_topic': LaunchConfiguration('slam_pose_topic'),
                            'output_topic': LaunchConfiguration('global_odom_topic'),
                        }],
                    )

    robot_localization_ekf_node = Node(
                        condition=UnlessCondition(custom_ekf),
                        package='robot_localization',
                        executable='ekf_node',
                        name='ekf_filter_node',
                        output='screen',
                        parameters=[os.path.join(get_package_share_directory("state_estimation"), 'config', 'ekf.yaml')],
                        remappings=[('/odometry/filtered', '/state_estimation/odom'),]
                    )
    
    return LaunchDescription([
        DeclareLaunchArgument('custom_ekf', default_value='false'),
        DeclareLaunchArgument('model_type', default_value='point_mass_model'),
        DeclareLaunchArgument('racecar_version', default_value='none'),
        DeclareLaunchArgument('floor', default_value='dubi'),

        DeclareLaunchArgument('odom_vesc_topic', default_value='/odom'),
        DeclareLaunchArgument('imu_topic', default_value='/imu'),
        DeclareLaunchArgument('odom_topic', default_value='/state_estimation/odom'),
        DeclareLaunchArgument('global_ekf', default_value='true'),
        DeclareLaunchArgument('global_odom_topic', default_value='/state_estimation/odom_map'),
        DeclareLaunchArgument('slam_pose_topic', default_value='/car_state/pose'),
        OpaqueFunction(function=custom_ekf_node_for_car),
        map_odom_node,
        robot_localization_ekf_node,
    ])
