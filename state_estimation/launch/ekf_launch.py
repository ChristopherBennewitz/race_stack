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
                'R_vesc': [1e6, 1e6, 1e6, 0.04, 100.0, 0.1],
                'Q': [4.9e-5, 3.6e-5, 3.6e-5, 5.8e-4, 1.8e-3, 6.7e-2, 2.3e-5, 6.9e-3],
            }],
        ),
    ])
