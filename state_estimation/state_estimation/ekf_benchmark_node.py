"""Ground-truth-free, side-by-side diagnostics for two EKF odometry topics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu


class RunningRms:
    def __init__(self):
        self.count = 0
        self.sum_squares = 0.0

    def add(self, value):
        self.count += 1
        self.sum_squares += float(np.dot(value, value)) if np.ndim(value) else float(value) ** 2

    @property
    def rms(self):
        return (self.sum_squares / self.count) ** 0.5 if self.count else float('nan')


class RunningMean:
    def __init__(self):
        self.count = 0
        self.total = 0.0

    def add(self, value):
        self.count += 1
        self.total += float(value)

    @property
    def mean(self):
        return self.total / self.count if self.count else float('nan')


class EkfBenchmark(Node):
    """Compare innovations and inertial consistency without a pose reference.

    These statistics do not claim absolute accuracy.  They expose whether an
    estimate is self-consistent with independent proprioceptive channels and
    whether its uncertainty is calibrated.  The default new EKF intentionally
    gives IMU acceleration negligible update weight, so acceleration consistency
    is an especially useful held-out diagnostic.
    """

    def __init__(self):
        super().__init__('ekf_benchmark')
        self.declare_parameter('legacy_topic', '/ekf_benchmark/legacy/odom')
        self.declare_parameter('candidate_topic', '/ekf_benchmark/simulator_model/odom')
        self.declare_parameter('vesc_topic', '/odom')
        self.declare_parameter('imu_topic', '/sensors/imu/raw')
        self.latest_vesc = None
        self.latest_imu = None
        self.previous = {}
        self.metrics = defaultdict(lambda: defaultdict(RunningRms))
        self.nis = defaultdict(RunningMean)
        self.create_subscription(Odometry, self.get_parameter('vesc_topic').value, self.vesc_callback, 10)
        self.create_subscription(Imu, self.get_parameter('imu_topic').value, self.imu_callback, 10)
        self.create_subscription(Odometry, self.get_parameter('legacy_topic').value,
                                 lambda msg: self.estimate_callback('legacy', msg), 10)
        self.create_subscription(Odometry, self.get_parameter('candidate_topic').value,
                                 lambda msg: self.estimate_callback('simulator_model', msg), 10)
        self.create_timer(10.0, self.report)

    def vesc_callback(self, message):
        self.latest_vesc = message

    def imu_callback(self, message):
        self.latest_imu = message

    def estimate_callback(self, name, message):
        velocity = np.array([message.twist.twist.linear.x, message.twist.twist.linear.y, message.twist.twist.angular.z])
        if self.latest_vesc is not None:
            vesc = self.latest_vesc.twist.twist
            residual = velocity - [vesc.linear.x, vesc.linear.y, vesc.angular.z]
            self.metrics[name]['vesc_twist_residual'].add(residual)
            candidate_covariance = np.asarray(message.twist.covariance).reshape(6, 6)
            vesc_covariance = np.asarray(self.latest_vesc.twist.covariance).reshape(6, 6)
            indices = [0, 1, 5]
            innovation = (
                candidate_covariance[np.ix_(indices, indices)]
                + vesc_covariance[np.ix_(indices, indices)]
                + np.eye(3) * 1e-6
            )
            self.nis[name].add(
                residual @ np.linalg.pinv(innovation) @ residual
            )
        if self.latest_imu is not None:
            self.metrics[name]['imu_yaw_rate_residual'].add(velocity[2] - self.latest_imu.angular_velocity.z)
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        previous = self.previous.get(name)
        if previous is not None and 1e-3 < stamp - previous[0] < 0.1 and self.latest_imu is not None:
            dt = stamp - previous[0]
            derivative = (velocity[:2] - previous[1][:2]) / dt
            # Body-frame accelerometer prediction includes the rotating-frame terms.
            acceleration = derivative + np.array([-velocity[2] * velocity[1], velocity[2] * velocity[0]])
            measured = np.array([self.latest_imu.linear_acceleration.x, self.latest_imu.linear_acceleration.y])
            self.metrics[name]['held_out_imu_acceleration_residual'].add(acceleration - measured)
            self.metrics[name]['velocity_jerk'].add((velocity[:2] - previous[1][:2]) / dt)
        self.previous[name] = (stamp, velocity)

    def report(self):
        lines = ['EKF benchmark (lower RMS is better; values are not absolute pose error):']
        for name, metrics in sorted(self.metrics.items()):
            values = ', '.join(
                f'{metric}={value.rms:.4g} ({value.count})'
                for metric, value in sorted(metrics.items())
            )
            if name in self.nis:
                values += f', vesc_twist_nis_mean={self.nis[name].mean:.4g} ({self.nis[name].count})'
            lines.append(f'  {name}: {values}')
        self.get_logger().info('\n'.join(lines))


def main(args=None):
    rclpy.init(args=args)
    node = EkfBenchmark()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.report()
        node.destroy_node()
        rclpy.shutdown()
