#!/usr/bin/env python3
"""
Reusable sensor abstraction for the EKF node.

Each sensor owns its subscription, buffers the newest measurement, and knows how
to apply its own sequential EKF update using the model's measurement functions.
Adding a new sensor to the filter is a single line in the node (see EkfNode).
"""
import time

import numpy as np
import tf_transformations as tft
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


def normalize_angle(x):
    x = x % (2 * np.pi)
    if x > np.pi:
        x -= 2 * np.pi
    return x


class Sensor:
    """
    Base class for a measurement source feeding the EKF.

    Subclasses only need to implement ``parse(msg)``, which reads a ROS message
    into ``self.data``. Everything else (subscription, buffering, freshness
    tracking, the EKF update and its timing stats) is handled here.

    Args:
        node:          owning EkfNode (used for the subscription and the shared lock)
        name:          short identifier, used for logging / performance reports
        msg_type:      ROS message type to subscribe to
        topic:         topic name
        dim:           measurement vector length
        R:             measurement covariance matrix
        Hx:            model measurement function h(x[, u])
        HJacobian:     model measurement Jacobian H(x[, u])
        uses_control:  if True, the control input u is passed to Hx/HJacobian
        angle_idx:     index in the measurement that is an angle (wrapped in the
                       residual), or None if there is no angular measurement
    """

    def __init__(self, node, name, msg_type, topic, dim, R, Hx, HJacobian,
                 uses_control=False, angle_idx=None):
        self.node = node
        self.name = name
        self.R = R
        self.Hx = Hx
        self.HJacobian = HJacobian
        self.uses_control = uses_control
        self.angle_idx = angle_idx

        self.data = np.zeros(dim)
        self.fresh = False

        # Performance stats
        self.update_count = 0
        self.update_mean_ms = 0.0
        self.update_max_ms = 0.0

        node.create_subscription(msg_type, topic, self._callback, 1)

    def parse(self, msg):
        """Read ``msg`` into ``self.data``. Called while holding the node lock."""
        raise NotImplementedError

    def _callback(self, msg):
        with self.node.lock:
            self.parse(msg)
            self.fresh = True

    def snapshot(self):
        """Return (fresh, copy of data) and clear the fresh flag. Call under lock."""
        fresh = self.fresh
        data = np.copy(self.data)
        self.fresh = False
        return fresh, data

    def residual(self, a, b):
        y = a - b
        if self.angle_idx is not None:
            y[self.angle_idx] = normalize_angle(y[self.angle_idx])
        return y

    def update(self, ekf, z, u):
        """Apply the sequential EKF update for this sensor and record timing."""
        t0 = time.perf_counter_ns()

        if self.uses_control:
            ekf.update(z=z, HJacobian=self.HJacobian, Hx=self.Hx, R=self.R,
                       args=u, hx_args=u, residual=self.residual)
        else:
            ekf.update(z=z, HJacobian=self.HJacobian, Hx=self.Hx, R=self.R,
                       residual=self.residual)

        duration_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        if duration_ms > self.update_max_ms:
            self.update_max_ms = duration_ms
        self.update_count += 1
        self.update_mean_ms += (duration_ms - self.update_mean_ms) / self.update_count


class ImuSensor(Sensor):
    """
    IMU sensor. Measurement: [yaw, yaw_rate, ax, ay].

    Yaw is relative to the first orientation reading, so the odom frame starts with its x axis along
    the car's initial heading (as robot_localization's ``imu0_relative`` did). The IMU's absolute yaw
    is in its own, mounted-rotated frame and would not match the odometry frame. The gyro rate is
    unaffected by a pure yaw mounting rotation. The accelerations are used as published (their weight
    in the filter is negligible) and are not rotated into the base frame.

    Applies an exponential moving average to the linear accelerations and can
    initialise the filter state from the first orientation reading.

    Args:
        on_init: optional callable invoked with the parsed measurement on the
                 first message, used to seed the filter state.
        alpha:   EMA weight for the newest acceleration sample.
    """

    def __init__(self, node, topic, R, Hx, HJacobian, on_init=None, alpha=0.2):
        super().__init__(node, 'imu', Imu, topic, 4, R, Hx, HJacobian,
                         uses_control=True, angle_idx=0)
        self.on_init = on_init
        self.alpha = alpha
        self.prev_ax = None
        self.prev_ay = None
        self.yaw0 = None
        self._initialized = False

    def parse(self, msg):
        new_ax = msg.linear_acceleration.x
        new_ay = msg.linear_acceleration.y

        if self.prev_ax is not None:
            self.data[2] = self.alpha * new_ax + (1 - self.alpha) * self.prev_ax
            self.data[3] = self.alpha * new_ay + (1 - self.alpha) * self.prev_ay
        else:  # first measurement
            self.data[2] = new_ax
            self.data[3] = new_ay

        self.prev_ax = self.data[2]
        self.prev_ay = self.data[3]

        self.data[1] = msg.angular_velocity.z

        q = [msg.orientation.x, msg.orientation.y,
             msg.orientation.z, msg.orientation.w]
        __, __, yaw = tft.euler_from_quaternion(q)
        yaw = normalize_angle(yaw)
        if self.yaw0 is None:
            self.yaw0 = yaw
        self.data[0] = normalize_angle(yaw - self.yaw0)

        if not self._initialized and self.on_init is not None:
            self.on_init(self.data)
            self._initialized = True


class OdomSensor(Sensor):
    """
    Full-state odometry sensor (e.g. VESC).
    Measurement: [x, y, yaw, vx, vy, yaw_rate].
    """

    def __init__(self, node, name, topic, R, Hx, HJacobian, angle_idx=2):
        super().__init__(node, name, Odometry, topic, 6, R, Hx, HJacobian,
                         uses_control=False, angle_idx=angle_idx)

    def parse(self, msg):
        self.data[0] = msg.pose.pose.position.x
        self.data[1] = msg.pose.pose.position.y
        self.data[3] = msg.twist.twist.linear.x
        self.data[4] = msg.twist.twist.linear.y
        self.data[5] = msg.twist.twist.angular.z

        q = [msg.pose.pose.orientation.x,
             msg.pose.pose.orientation.y,
             msg.pose.pose.orientation.z,
             msg.pose.pose.orientation.w]
        __, __, yaw = tft.euler_from_quaternion(q)
        self.data[2] = yaw
