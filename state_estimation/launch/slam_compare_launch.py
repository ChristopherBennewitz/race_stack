"""Compare how well Cartographer's scan matching agrees with the old
(robot_localization) and new (simulator-model) EKF odometry.

Replays a bag containing /odom, /sensors/imu/raw, /ackermann_cmd_applied and
a lidar scan topic through both EKFs (see ekf_compare_launch.py) and, in
parallel, through two independent Cartographer *mapping*-mode instances --
one per candidate odometry -- each in its own ROS namespace so they don't
collide on topics or on the transform tree (the standard `/tf` -> `tf`,
`/tf_static` -> `tf_static` remap trick, applied inside the pushed
namespace). cartographer_ros does not expose its internal scan-matcher
cost/residual over ROS, so `slam_match_benchmark_node` reports a proxy: how
well each Cartographer instance's own scan-matched trajectory
(`tracked_pose`) agrees with its candidate's odometry, plus a count of
likely tracking-loss jumps.

    ros2 launch state_estimation slam_compare_launch.py racecar_version:=NUC6 scan_topic:=/scan

Both Cartographer instances reuse the car's existing, unmodified
`f110_2d.lua` (mapping mode) -- nothing about its tuning is changed, only
where its topics/tf end up. This has not been run end-to-end against real
cartographer_ros/hardware in this environment; treat the first run as a
supervised smoke test.
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():
    racecar_version = LaunchConfiguration('racecar_version')
    scan_topic = LaunchConfiguration('scan_topic')

    def cartographer_group(name, odom_topic):
        return GroupAction([
            PushRosNamespace(name),
            Node(
                package='cartographer_ros', executable='cartographer_node',
                name='cartographer_node', output='screen',
                arguments=[
                    '-configuration_directory',
                    [get_package_share_directory('stack_master'), '/config/', racecar_version, '/slam'],
                    '-configuration_basename', 'f110_2d.lua',
                    '--minloglevel', '2',
                ],
                remappings=[
                    ('scan', scan_topic),
                    ('odom', odom_topic),
                    # Isolate this instance's transform tree from the other
                    # Cartographer instance's (both otherwise publish
                    # map -> base_link under the same global /tf).
                    ('/tf', 'tf'),
                    ('/tf_static', 'tf_static'),
                ],
            ),
        ])

    return LaunchDescription([
        DeclareLaunchArgument('racecar_version', default_value='NUC6'),
        DeclareLaunchArgument('scan_topic', default_value='/scan'),

        # Both EKFs (robot_localization "legacy" + the new "simulator_model")
        # plus the existing IMU/VESC-only ekf_benchmark_node.
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            [get_package_share_directory('state_estimation'), '/launch/ekf_compare_launch.py'])),

        cartographer_group('legacy', '/ekf_benchmark/legacy/odom'),
        cartographer_group('candidate', '/ekf_benchmark/simulator_model/odom'),

        Node(package='state_estimation', executable='slam_match_benchmark_node',
             name='slam_match_benchmark', output='screen'),
    ])
