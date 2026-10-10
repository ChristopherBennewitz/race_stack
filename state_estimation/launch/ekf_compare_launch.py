"""Run legacy robot_localization and the simulator-model EKF side-by-side."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(get_package_share_directory('state_estimation'), 'config', 'ekf.yaml')
    return LaunchDescription([
        Node(package='robot_localization', executable='ekf_node', name='ekf_filter_node', output='screen',
             parameters=[config], remappings=[('/odometry/filtered', '/ekf_benchmark/legacy/odom')]),
        Node(package='state_estimation', executable='ekf_node', name='ekf_node', output='screen',
             parameters=[{'odom_topic': '/ekf_benchmark/simulator_model/odom'}]),
        Node(package='state_estimation', executable='ekf_benchmark_node', name='ekf_benchmark', output='screen'),
    ])
