#!/usr/bin/env python3
"""Ground-truth-free comparison of how well Cartographer's scan matching
agrees with each candidate EKF's odometry.

cartographer_ros does not expose its internal scan-matcher cost/residual
over ROS, so this is a proxy, in the same spirit as ekf_benchmark_node.py:
it compares Cartographer's own scan-matched trajectory (the `tracked_pose`
topic) against the candidate EKF's reported speed and yaw rate at the same
time. A small, steady disagreement means the EKF's motion prediction was a
good prior for the scan matcher's search window -- i.e. matching was "easy".
A large disagreement, or an outright pose jump, means the matcher had to
fight the prior or lost track.
"""

from __future__ import annotations

import math
from collections import defaultdict

import rclpy
import tf_transformations as tft
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


def normalize_angle(x):
    x = x % (2 * math.pi)
    if x > math.pi:
        x -= 2 * math.pi
    return x


class RunningRms:
    def __init__(self):
        self.count = 0
        self.sum_squares = 0.0

    def add(self, value):
        self.count += 1
        self.sum_squares += float(value) ** 2

    @property
    def rms(self):
        return (self.sum_squares / self.count) ** 0.5 if self.count else float('nan')


class SlamMatchBenchmark(Node):
    """Compares each candidate's odometry against Cartographer's own trajectory.

    A large, sustained gap between "what the EKF said we were doing" and
    "what Cartographer's scan matcher actually found" means the odometry
    prior wasn't a good hint for the scan matcher -- i.e. matching was
    harder. Occasional large jumps in Cartographer's own trajectory are
    flagged separately as likely tracking-loss events, and excluded from the
    residual metrics since they are not meaningful disagreement samples.
    """

    # A tracked_pose jump implying a speed/yaw-rate far outside the car's
    # physical envelope is far more likely a matching hiccup than real motion.
    JUMP_SPEED_MPS = 8.0
    JUMP_YAW_RATE_RADPS = 8.0

    def __init__(self):
        super().__init__('slam_match_benchmark')
        self.declare_parameter('legacy_odom_topic', '/ekf_benchmark/legacy/odom')
        self.declare_parameter('candidate_odom_topic', '/ekf_benchmark/simulator_model/odom')
        self.declare_parameter('legacy_tracked_pose_topic', '/legacy/tracked_pose')
        self.declare_parameter('candidate_tracked_pose_topic', '/candidate/tracked_pose')

        self.latest_odom = {}
        self.previous_pose = {}  # name -> (stamp, x, y, yaw)
        self.pose_count = defaultdict(int)
        self.jump_count = defaultdict(int)
        self.metrics = defaultdict(lambda: defaultdict(RunningRms))

        self.create_subscription(Odometry, self.get_parameter('legacy_odom_topic').value,
                                 lambda msg: self.odom_callback('legacy', msg), 10)
        self.create_subscription(Odometry, self.get_parameter('candidate_odom_topic').value,
                                 lambda msg: self.odom_callback('simulator_model', msg), 10)
        self.create_subscription(PoseStamped, self.get_parameter('legacy_tracked_pose_topic').value,
                                 lambda msg: self.tracked_pose_callback('legacy', msg), 10)
        self.create_subscription(PoseStamped, self.get_parameter('candidate_tracked_pose_topic').value,
                                 lambda msg: self.tracked_pose_callback('simulator_model', msg), 10)
        self.create_timer(10.0, self.report)

    def odom_callback(self, name, message):
        self.latest_odom[name] = message

    def tracked_pose_callback(self, name, message):
        self.pose_count[name] += 1
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        x, y = message.pose.position.x, message.pose.position.y
        quaternion = [message.pose.orientation.x, message.pose.orientation.y,
                      message.pose.orientation.z, message.pose.orientation.w]
        yaw = tft.euler_from_quaternion(quaternion)[2]

        previous = self.previous_pose.get(name)
        self.previous_pose[name] = (stamp, x, y, yaw)
        if previous is None:
            return
        prev_stamp, prev_x, prev_y, prev_yaw = previous
        dt = stamp - prev_stamp
        if not (1e-3 < dt < 0.5):
            return

        translation = math.hypot(x - prev_x, y - prev_y)
        d_yaw = normalize_angle(yaw - prev_yaw)
        cartographer_speed = translation / dt
        cartographer_yaw_rate = d_yaw / dt

        if cartographer_speed > self.JUMP_SPEED_MPS or abs(cartographer_yaw_rate) > self.JUMP_YAW_RATE_RADPS:
            # Almost certainly a tracking-loss/re-match jump, not real motion;
            # counted separately rather than folded into the residual RMS.
            self.jump_count[name] += 1
            return

        odom = self.latest_odom.get(name)
        if odom is None:
            return
        ekf_speed = math.hypot(odom.twist.twist.linear.x, odom.twist.twist.linear.y)
        ekf_yaw_rate = odom.twist.twist.angular.z
        self.metrics[name]['cartographer_speed_residual'].add(cartographer_speed - ekf_speed)
        self.metrics[name]['cartographer_yaw_rate_residual'].add(cartographer_yaw_rate - ekf_yaw_rate)

    def report(self):
        lines = ['SLAM-match benchmark (lower RMS = odometry agrees better with scan matching; '
                 'not an absolute matching-quality score):']
        for name in sorted(set(self.metrics) | set(self.pose_count)):
            values = ', '.join(
                f'{metric}={value.rms:.4g} ({value.count})'
                for metric, value in sorted(self.metrics[name].items())
            )
            lines.append(
                f'  {name}: {values}, tracked_pose_count={self.pose_count[name]}, '
                f'jump_events={self.jump_count[name]}'
            )
        self.get_logger().info('\n'.join(lines))


def main(args=None):
    rclpy.init(args=args)
    node = SlamMatchBenchmark()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.report()
        node.destroy_node()
        rclpy.shutdown()
