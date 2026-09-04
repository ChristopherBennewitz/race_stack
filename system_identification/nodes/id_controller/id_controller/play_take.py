#!/usr/bin/env python3
"""Play one open-loop excitation take and stop. Nothing here reacts to the car.

    ros2 run id_controller play_take --ros-args -p take:=M2_skidpad_L
    ros2 run id_controller play_take --ros-args -p csv:=/path/to/M2_skidpad_L.csv

Publishes ``ackermann_msgs/AckermannDriveStamped`` on ``/drive`` at the take's
own rate, then publishes zero speed and exits.

Open loop means open loop: no planner or lap logic generates the command. The
actuation manager remains in the path for physical limits and deadman safety,
and its applied output is recorded. The only feedback in this node is the abort
path -- it can stop the car, it can never steer it.

Exit codes: 0 completed, 2 bad arguments, 3 aborted.
"""
from __future__ import annotations

import time

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Joy

from id_controller import takes as takes_lib

EXIT_OK = 0
EXIT_ARGS = 2
EXIT_ABORT = 3


class TakePlayer(Node):
    def __init__(self) -> None:
        super().__init__("take_player")
        self.declare_parameter("take", "")
        self.declare_parameter("csv", "")
        self.declare_parameter("topic", "/drive")
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("rate_hz", float(takes_lib.HZ))
        self.declare_parameter("countdown", 3.0)
        self.declare_parameter("tail_sec", 1.0)
        self.declare_parameter("require_deadman", True)
        self.declare_parameter("deadman_button", 5)
        self.declare_parameter("human_deadman_button", 4)
        self.declare_parameter("dry_run", False)

        take = self.get_parameter("take").value
        csv_path = self.get_parameter("csv").value
        if bool(take) == bool(csv_path):
            self.get_logger().error("set exactly one of 'take' or 'csv'")
            raise SystemExit(EXIT_ARGS)

        try:
            self.rows = (takes_lib.build(take) if take
                         else takes_lib.load_csv(csv_path))
        except (KeyError, ValueError, OSError) as exc:
            self.get_logger().error(str(exc))
            raise SystemExit(EXIT_ARGS)

        self.name = take or csv_path
        self.hz = float(self.get_parameter("rate_hz").value)
        self.frame = self.get_parameter("frame_id").value
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.require_deadman = bool(self.get_parameter("require_deadman").value)
        self.deadman_button = int(self.get_parameter("deadman_button").value)
        self.human_deadman_button = int(
            self.get_parameter("human_deadman_button").value)

        # Reliable, small depth: these are commands, and a late one is worse than
        # a dropped one.
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(
            AckermannDriveStamped, self.get_parameter("topic").value, qos)

        # Abort path only. Never touches the commanded values.
        self.deadman_ok = not self.require_deadman
        self.abort_reason = ""
        self.create_subscription(Joy, "/joy", self._joy_cb, 10)

        self.i = 0
        self.tail = 0
        self.n_tail = int(round(float(self.get_parameter("tail_sec").value) * self.hz))
        self.done = False
        self.aborted = False
        self.t_start = None
        self.t_end = None

        dur = len(self.rows) / self.hz
        self.get_logger().info(
            f"{self.name}: {len(self.rows)} rows, {dur:.1f} s at {self.hz:.0f} Hz"
            + (" [DRY RUN, publishing nothing]" if self.dry_run else ""))

        countdown = float(self.get_parameter("countdown").value)
        if self.require_deadman:
            self.get_logger().info(
                f"hold joystick button {self.deadman_button} for the whole take; "
                "releasing it aborts")
            self._wait_for_deadman()
        self.get_logger().info(f"starting in {countdown:.0f} s -- clear the area")
        self._sleep_spinning(countdown)
        if self.aborted:
            self.done = True
            return
        self.t_start = time.monotonic()
        self.timer = self.create_timer(1.0 / self.hz, self.tick)

    # --- abort path -----------------------------------------------------------

    def _joy_cb(self, msg: Joy) -> None:
        held = (len(msg.buttons) > self.deadman_button
                and msg.buttons[self.deadman_button] == 1)
        human_held = (len(msg.buttons) > self.human_deadman_button
                      and msg.buttons[self.human_deadman_button] == 1)
        if self.require_deadman and human_held:
            self._abort(
                f"human deadman button {self.human_deadman_button} pressed")
            return
        if self.require_deadman and self.deadman_ok and not held:
            self._abort(f"deadman button {self.deadman_button} released")
        self.deadman_ok = held or not self.require_deadman

    def _abort(self, reason: str) -> None:
        if self.aborted:
            return
        self.aborted = True
        self.abort_reason = reason
        self.done = True
        self.get_logger().error(f"ABORT: {reason}")

    def _wait_for_deadman(self) -> None:
        self.get_logger().info("waiting for deadman...")
        while rclpy.ok() and not self.deadman_ok and not self.aborted:
            rclpy.spin_once(self, timeout_sec=0.1)

    def _sleep_spinning(self, sec: float) -> None:
        end = time.monotonic() + sec
        while rclpy.ok() and not self.aborted and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    # --- playback -------------------------------------------------------------

    def send(self, steer: float, speed: float) -> None:
        if self.dry_run:
            return
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame
        msg.drive.steering_angle = float(steer)
        msg.drive.speed = float(speed)
        self.pub.publish(msg)

    def tick(self) -> None:
        if self.aborted:
            return
        if self.i < len(self.rows):
            steer, speed = self.rows[self.i]
            self.send(steer, speed)
            self.i += 1
            if self.i % int(self.hz) == 0:
                self.get_logger().info(
                    f"  t={self.i / self.hz:5.1f}s  steer={steer:+.3f}  v={speed:.2f}",
                    throttle_duration_sec=0.9)
            return
        # Hold the last steering angle but command zero speed, so the car coasts
        # to a stop without a steering step in the tail of the recording.
        if self.tail < self.n_tail:
            self.send(self.rows[-1][0], 0.0)
            self.tail += 1
            return
        self.t_end = time.monotonic()
        self.timer.cancel()
        self.done = True
        self.get_logger().info("take complete")

    def report_rate(self) -> None:
        if self.t_start is None or self.t_end is None or self.i == 0:
            return
        elapsed = self.t_end - self.t_start
        expected = (self.i + self.tail) / self.hz
        achieved = (self.i + self.tail) / elapsed if elapsed > 0 else 0.0
        self.get_logger().info(
            f"achieved {achieved:.1f} Hz over {elapsed:.1f} s "
            f"(nominal {self.hz:.0f} Hz / {expected:.1f} s)")
        if abs(achieved - self.hz) > 0.05 * self.hz:
            self.get_logger().warn(
                "rate drifted more than 5% -- extraction assumes a uniform grid, "
                "pass the achieved rate to --control-hz or redo the take")


def main() -> None:
    rclpy.init()
    node = None
    code = EXIT_OK
    try:
        node = TakePlayer()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        code = EXIT_ABORT if node.aborted else EXIT_OK
    except SystemExit as exc:
        code = int(exc.code) if exc.code is not None else EXIT_OK
    except KeyboardInterrupt:
        code = EXIT_ABORT
    finally:
        if node is not None:
            # Whatever happened -- finished, Ctrl-C, abort, exception -- request
            # a stop several times. The actuation-manager and VESC watchdogs are
            # independent fallbacks if these messages do not arrive.
            for _ in range(5):
                node.send(0.0, 0.0)
                time.sleep(0.02)
            node.report_rate()
            node.destroy_node()
        rclpy.try_shutdown()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
