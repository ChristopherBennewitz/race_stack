import math

import rclpy
import torch
from ackermann_msgs.msg import AckermannDriveStamped
from f110_msgs.msg import WpntArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile

from rl_controller.observation import Centerline, HistoryBuffer
from rl_controller.policy import load_policy


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class RLController(Node):
    """Runs a trained PPO/RMA policy at its training control rate and publishes `/drive`.

    Inputs: `/centerline_waypoints` (track geometry, latched once) and `/car_state/odom`
    (pose, speed, yaw rate from the EKF). Output: AckermannDriveStamped speed + steering angle,
    which the car's actuation manager rate-limits exactly as the sim's actuation limiter does.
    """

    def __init__(self):
        super().__init__('rl_controller')
        self.declare_parameter('checkpoint', '')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('odom_topic', '/car_state/odom')
        self.declare_parameter('centerline_topic', '/centerline_waypoints')
        self.declare_parameter('max_odom_age_s', 0.1)
        # Driving direction around the track; must match the direction the policy was trained for.
        self.declare_parameter('direction', 'ccw')

        path = self.get_parameter('checkpoint').value
        if not path:
            raise ValueError("parameter 'checkpoint' is required")
        self.policy = load_policy(path, self.get_parameter('device').value)
        shape = self.policy.shape
        self.history = HistoryBuffer(shape.history_len)
        self.max_odom_age = self.get_parameter('max_odom_age_s').value
        self.get_logger().info(
            f"Loaded {path}: {shape.n_waypoints} waypoints, history {shape.history_len}, {shape.control_hz} Hz"
        )
        low = (self.policy.action_bias - self.policy.action_scale).tolist()
        high = (self.policy.action_bias + self.policy.action_scale).tolist()
        self.get_logger().info(
            f"Action bounds [vel_cmd, steer_cmd]: low {low}, high {high}. These must agree with the actuation "
            "manager's limits; the policy's previous-action input is normalized with them."
        )
        self.direction = self.get_parameter('direction').value

        self.centerline = None
        self.odom = None
        self.prev_cmd = (0.0, 0.0)
        self.first_step = True

        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.get_parameter('drive_topic').value, 10)
        # The stack's centerline publisher uses the default (volatile) QoS and republishes every 10 s. A
        # TRANSIENT_LOCAL subscription is incompatible with a volatile publisher and would never connect, so
        # use the default reliable/volatile profile, which also matches a latched publisher (without its history).
        self.create_subscription(
            WpntArray, self.get_parameter('centerline_topic').value, self.centerline_cb, QoSProfile(depth=1)
        )
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value, self.odom_cb, 10)
        self.create_timer(1.0 / shape.control_hz, self.control_loop)

    def centerline_cb(self, msg):
        if self.centerline is not None or not msg.wpnts:
            return
        xy = [(w.x_m, w.y_m) for w in msg.wpnts]
        self.centerline = Centerline(
            xy, [w.d_left for w in msg.wpnts], [w.d_right for w in msg.wpnts], direction=self.direction
        )
        self.get_logger().info(f"Received centerline with {len(xy)} points")

    def odom_cb(self, msg):
        self.odom = msg

    def control_loop(self):
        if self.centerline is None or self.odom is None:
            return
        age = (self.get_clock().now() - rclpy.time.Time.from_msg(self.odom.header.stamp)).nanoseconds * 1e-9
        if age > self.max_odom_age:
            self.get_logger().warn(f"Stale odometry ({age:.2f}s old); not driving", throttle_duration_sec=1.0)
            return

        pose, twist = self.odom.pose.pose, self.odom.twist.twist
        x, y = pose.position.x, pose.position.y
        yaw = yaw_from_quaternion(pose.orientation)
        vel, yaw_rate = twist.linear.x, twist.angular.z

        waypoints = torch.from_numpy(self.centerline.lookahead(x, y, yaw, self.policy.shape.n_waypoints))
        # The newest history row pairs the current measurement with the previously issued command. On the
        # first step the sim's history is still all zeros (rows are only pushed after an action), so mirror that.
        if self.first_step:
            history = self.history.buf
            self.first_step = False
        else:
            history = self.history.push(vel, yaw_rate, *self.prev_cmd)
        vel_cmd, steer_cmd = self.policy(vel, yaw_rate, waypoints, history).tolist()
        self.prev_cmd = (vel_cmd, steer_cmd)

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed = vel_cmd
        msg.drive.steering_angle = steer_cmd
        self.drive_pub.publish(msg)


def main():
    rclpy.init()
    node = RLController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
