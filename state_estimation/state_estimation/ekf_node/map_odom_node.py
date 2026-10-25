#!/usr/bin/env python3
import math

import numpy as np
import rclpy
import tf_transformations as tft
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import UInt32

from . import defaults, diag_ros
from .diagnostics import SensorHealth
from .map_odom_correction import MapOdomCorrection


class MapOdomNode(Node):
    """The map-frame odometry for the controllers: the local filter's odometry plus a smoothly corrected ``map -> odom``.

    Subscribes to the local filter's ``odom`` (SLAM-free, the Cartographer prior) and to the SLAM pose
    (``/car_state/pose``) and publishes ``odom_map`` at the local filter's rate. Position and heading are the local
    pose moved into the map frame by the correction (``MapOdomCorrection``); the twist is the local filter's, which is
    in the body frame and so unchanged by a rigid change of frame. Nothing is fed back to the localization and no TF is
    broadcast (Cartographer owns ``map -> odom``).

    Until the first SLAM pose has arrived nothing is published.

    Observability: ``/state_estimation/diagnostics`` (DiagnosticArray, 2 Hz) carries the correction, its drift rate,
    how far the last pose was from the output, the pose outcomes and the pose arrival (rate, lag, age).
    ``/state_estimation/odom_map_jumps`` (UInt32, latched) counts the times the output stepped because the correction was
    set from a pose; a consumer holding a plan in the map frame (the MPC) should drop it when the count changes.
    """

    def __init__(self):
        super().__init__('map_odom_node')
        c = defaults.CORRECTION
        self.declare_parameter('odom_topic', '/state_estimation/odom')
        self.declare_parameter('slam_pose_topic', '/car_state/pose')
        self.declare_parameter('output_topic', '/state_estimation/odom_map')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('tau', c['tau'])
        self.declare_parameter('smooth', c['smooth'])
        self.declare_parameter('pose_delay', c['pose_delay'])
        self.declare_parameter('gate_distance', c['gate_distance'])
        self.declare_parameter('gate_angle', c['gate_angle'])
        self.declare_parameter('reseed', c['reseed'])
        self.declare_parameter('std', c['std'])
        self.declare_parameter('std_growth', c['std_growth'])
        get = lambda name: self.get_parameter(name).value
        self.correction = MapOdomCorrection(get('tau'), get('pose_delay'), get('gate_distance'), get('gate_angle'),
                                            get('reseed'), get('smooth'))
        self.frame_id = get('frame_id')
        self.std = np.array(get('std'), dtype=float)
        self.std_growth = float(get('std_growth'))
        self.pub = self.create_publisher(Odometry, get('output_topic'), 1)
        self.diag_pub = self.create_publisher(diag_ros.DiagnosticArray, '/state_estimation/diagnostics', 10)
        self.jump_pub = self.create_publisher(
            UInt32, get('output_topic') + '_jumps',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
        self.published_jumps = None
        self.pose_health = SensorHealth('slam_pose', 3)
        self.odom_health = SensorHealth('local_odom', 3)
        self.create_timer(0.5, self.publish_diagnostics)
        self.create_subscription(PoseStamped, get('slam_pose_topic'), self.pose_callback, 10)
        self.create_subscription(Odometry, get('odom_topic'), self.odom_callback, 1)
        self.get_logger().info(f"Correcting {get('odom_topic')} with {get('slam_pose_topic')} -> {get('output_topic')}")

    @staticmethod
    def seconds(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    @staticmethod
    def yaw_of(q):
        return tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

    def pose_callback(self, msg):
        self.pose_health.message(self.now_seconds(), self.seconds(msg.header.stamp))
        outcome = self.correction.add_pose(self.seconds(msg.header.stamp),
                                           (msg.pose.position.x, msg.pose.position.y, self.yaw_of(msg.pose.orientation)))
        if outcome == 'jumped' and self.correction.jumps:
            self.get_logger().warn("SLAM pose disagreed with the map-frame odometry repeatedly; jumped to it")
        self.publish_jumps()

    def now_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def publish_jumps(self):
        if self.correction.jumps != self.published_jumps:
            self.published_jumps = self.correction.jumps
            self.jump_pub.publish(UInt32(data=int(self.published_jumps)))

    def publish_diagnostics(self):
        now = self.now_seconds()
        values = {**self.correction.diagnostics(), 'jumps': float(self.correction.jumps),
                  **{f'pose_{k}': v for k, v in self.pose_health.values(now).items() if k in ('rate_hz', 'age_s', 'stamp_lag_ms')},
                  **{f'odom_{k}': v for k, v in self.odom_health.values(now).items() if k in ('rate_hz', 'age_s')}}
        pose_age, odom_age = self.pose_health.age(now), self.odom_health.age(now)
        if self.correction.transform is None:
            level, message = 'error', 'no SLAM pose yet: nothing is published'
        elif pose_age > 2.0:
            level, message = 'error', f'no SLAM pose for {pose_age:.1f} s: the correction is frozen'
        elif odom_age > 0.5:
            level, message = 'error', f'no local odometry for {odom_age:.1f} s'
        elif pose_age > 0.5:
            level, message = 'warn', f'SLAM pose is {pose_age:.1f} s old'
        elif values['innovation_m'] > 0.3:
            level, message = 'warn', f"the output is {values['innovation_m']:.2f} m from the SLAM pose"
        else:
            level, message = 'ok', 'tracking the SLAM pose'
        self.diag_pub.publish(diag_ros.array(self.get_clock().now().to_msg(),
                                             [diag_ros.status('map_odom', level, message, values)]))

    def odom_callback(self, msg):
        t = self.seconds(msg.header.stamp)
        self.odom_health.message(self.now_seconds(), t)
        p = msg.pose.pose
        local = (p.position.x, p.position.y, self.yaw_of(p.orientation))
        self.correction.add_local(t, local)
        if self.correction.transform is None:
            return
        x, y, yaw = self.correction.output(local, t)

        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id
        out.child_frame_id = msg.child_frame_id
        out.pose.pose.position.x, out.pose.pose.position.y = float(x), float(y)
        q = tft.quaternion_from_euler(0, 0, yaw)
        out.pose.pose.orientation.x, out.pose.pose.orientation.y = float(q[0]), float(q[1])
        out.pose.pose.orientation.z, out.pose.pose.orientation.w = float(q[2]), float(q[3])
        out.twist = msg.twist  # body frame: unchanged by the frame change
        out.pose.covariance = self.pose_covariance(msg.pose.covariance, t)
        self.pub.publish(out)

    def pose_covariance(self, local, t):
        """The local pose covariance rotated into the map frame, plus the correction's uncertainty.

        The correction's part grows with the time since the last SLAM pose (``std_growth`` per second, per axis), so a
        consumer sees the output become less certain when the localization stalls.
        """
        stale = max(0.0, t - self.correction.last_stamp) if self.correction.last_stamp is not None else 0.0
        std = self.std + self.std_growth * stale
        cov = np.array(local, dtype=float).reshape(6, 6)
        c, s = math.cos(self.correction.transform[2]), math.sin(self.correction.transform[2])
        R = np.eye(6)
        R[:2, :2] = [[c, -s], [s, c]]
        cov = R @ cov @ R.T
        cov[0, 0] += std[0] ** 2
        cov[1, 1] += std[1] ** 2
        cov[5, 5] += std[2] ** 2
        return cov.flatten().tolist()


def main(args=None):
    rclpy.init(args=args)
    node = MapOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
