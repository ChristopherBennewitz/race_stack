"""Launch the simulator-model EKF used by the normal state-estimation stack."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('odom_topic', default_value='/early_fusion/odom'),
        DeclareLaunchArgument('vesc_topic', default_value='/odom'),
        DeclareLaunchArgument('imu_topic', default_value='/sensors/imu/raw'),
        Node(
            package='state_estimation', executable='ekf_node', name='ekf_node', output='screen',
            parameters=[{
                'odom_topic': LaunchConfiguration('odom_topic'),
                'vesc_topic': LaunchConfiguration('vesc_topic'),
                'imu_topic': LaunchConfiguration('imu_topic'),
                'frequency': 100.0,
                'R_imu': [0.1, 0.0009, 1e6, 1e6],
                'R_vesc': [1e6, 1e6, 1e6, 0.0225, 100.0, 0.1],
                'Q': [4.9e-5, 3.6e-5, 3.6e-5, 4.0e-4, 2.6e-3, 0.18, 4.2e-3],
            }],
        ),
    ])
