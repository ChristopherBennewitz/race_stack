#!/usr/bin/env python3
"""Pre-take checks on the race_stack command path.

Run standalone::

    ros2 run id_controller preflight
    ros2 run id_controller preflight --ros-args -p require_deadman:=False

or let ``run_take`` call it for you. Every check here exists because it has
silently ruined a recording before:

``drive_exclusive``
    Exactly one thing may publish ``/drive`` during a take. If the controller or
    state machine is up it will publish too, the two streams interleave, and the
    bag looks perfect while being unidentifiable -- the command is then
    correlated with the state it was reacting to.

``teleop_silent``
    Unexpected ``/teleop`` traffic is reported. It cannot override ``/drive``
    while only button 5 is held, because the actuation manager gates each source
    with its own deadman, but silence makes the recorded source unambiguous.

``deadman_held``
    Autonomous button 5 must remain held and human button 4 must remain released.
    The actuation manager enforces this source-specific authorization directly.

``actuator_single_publisher``
    ``/commands/motor/speed`` and ``/commands/servo/position`` should have
    exactly one publisher (``ackermann_to_vesc``), and
    ``/ackermann_cmd_applied`` should have exactly one publisher (the actuation
    manager).

``topics_alive``
    The command and measurement topics needed by the bag are producing data.
"""
from __future__ import annotations

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, Joy
from std_msgs.msg import Float64

#: Topics that must be in every take's bag for command and response reconstruction.
BAG_TOPICS = (
    "/car_state/odom",
    "/sensors/imu/raw",
    "/commands/servo/position",
    "/commands/motor/speed",
    "/ackermann_cmd_applied",
    "/drive",
    "/teleop",
    "/joy",
)

#: Topics that must be producing data before a take starts. /drive is ours and
#: /teleop is required to be silent, so neither belongs here.
LIVE_TOPICS = {
    "/car_state/odom": Odometry,
    "/sensors/imu/raw": Imu,
    "/commands/servo/position": Float64,
    "/commands/motor/speed": Float64,
    "/ackermann_cmd_applied": AckermannDriveStamped,
}

DEADMAN_BUTTON = 5
HUMAN_DEADMAN_BUTTON = 4


class Check:
    """One named pass/fail with a human-readable detail line."""

    def __init__(self, name: str, ok: bool, detail: str, fatal: bool = True) -> None:
        self.name = name
        self.ok = ok
        self.detail = detail
        self.fatal = fatal

    def __str__(self) -> str:
        if self.ok:
            mark = "PASS"
        else:
            mark = "FAIL" if self.fatal else "WARN"
        return f"  [{mark}] {self.name:<26} {self.detail}"


class Preflight(Node):
    """Listens for a fixed window, then reports on the command path."""

    def __init__(self, node_name: str = "sysid_preflight") -> None:
        super().__init__(node_name)
        self.declare_parameter("window", 2.0)
        self.declare_parameter("require_deadman", True)
        self.declare_parameter("deadman_button", DEADMAN_BUTTON)
        self.declare_parameter("human_deadman_button", HUMAN_DEADMAN_BUTTON)
        self.declare_parameter("expect_drive_publishers", 0)

        self.window = float(self.get_parameter("window").value)
        self.require_deadman = bool(self.get_parameter("require_deadman").value)
        self.deadman_button = int(self.get_parameter("deadman_button").value)
        self.human_deadman_button = int(
            self.get_parameter("human_deadman_button").value)
        self.expect_drive_pubs = int(
            self.get_parameter("expect_drive_publishers").value)

        self.counts = {t: 0 for t in LIVE_TOPICS}
        self.teleop_count = 0
        self.joy_count = 0
        self.deadman_seen = False
        self.deadman_always = True
        self.human_deadman_seen = False

        for topic, msg_type in LIVE_TOPICS.items():
            self.create_subscription(
                msg_type, topic, lambda _m, t=topic: self._bump(t), 10)
        self.create_subscription(
            AckermannDriveStamped, "/teleop", self._teleop_cb, 10)
        self.create_subscription(Joy, "/joy", self._joy_cb, 10)

    def _bump(self, topic: str) -> None:
        self.counts[topic] += 1

    def _teleop_cb(self, _msg) -> None:
        self.teleop_count += 1

    def _joy_cb(self, msg: Joy) -> None:
        self.joy_count += 1
        held = (len(msg.buttons) > self.deadman_button
                and msg.buttons[self.deadman_button] == 1)
        human_held = (len(msg.buttons) > self.human_deadman_button
                      and msg.buttons[self.human_deadman_button] == 1)
        self.deadman_seen |= held
        self.deadman_always &= held
        self.human_deadman_seen |= human_held

    def collect(self) -> list:
        """Spin for the window, then evaluate. Returns a list of Check."""
        end = self.get_clock().now().nanoseconds + int(self.window * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        return self._evaluate()

    def _evaluate(self) -> list:
        checks = []

        n_drive = self.count_publishers("/drive")
        checks.append(Check(
            "drive_exclusive",
            n_drive <= self.expect_drive_pubs,
            f"{n_drive} publisher(s) on /drive, expected {self.expect_drive_pubs}"
            + ("" if n_drive <= self.expect_drive_pubs
               else " -- stop the controller / state machine"),
        ))

        checks.append(Check(
            "teleop_silent",
            self.teleop_count == 0,
            f"{self.teleop_count} msg on /teleop in {self.window:.1f} s"
            + ("" if self.teleop_count == 0
               else " -- unexpected traffic; verify button 4 is released"),
            fatal=False,
        ))

        if self.require_deadman:
            ok = (self.joy_count > 0 and self.deadman_always
                  and not self.human_deadman_seen)
            if self.joy_count == 0:
                detail = "no /joy traffic -- is the joystick connected?"
            elif not self.deadman_seen:
                detail = f"button {self.deadman_button} never held"
            elif not self.deadman_always:
                detail = f"button {self.deadman_button} released during the window"
            elif self.human_deadman_seen:
                detail = f"human button {self.human_deadman_button} was pressed"
            else:
                detail = f"button {self.deadman_button} held throughout"
            checks.append(Check("deadman_held", ok, detail))

        for topic in ("/commands/motor/speed", "/commands/servo/position"):
            n = self.count_publishers(topic)
            checks.append(Check(
                f"single_publisher{topic}",
                n == 1,
                f"{n} publisher(s)"
                + ("" if n == 1
                   else " -- expected exactly 1 (ackermann_to_vesc)"),
            ))

        n_applied = self.count_publishers("/ackermann_cmd_applied")
        checks.append(Check(
            "applied_single_publisher",
            n_applied == 1,
            f"{n_applied} publisher(s)"
            + ("" if n_applied == 1
               else " -- expected exactly 1 (actuation_manager)"),
        ))

        for topic, n in self.counts.items():
            checks.append(Check(
                f"alive{topic}", n > 0,
                f"{n} msg in {self.window:.1f} s"
                + ("" if n > 0 else " -- nothing publishing"),
            ))

        return checks


def report(checks: list, log) -> bool:
    """Print every check; return True if no fatal check failed."""
    for c in checks:
        log(str(c))
    failed = [c for c in checks if not c.ok and c.fatal]
    if failed:
        log(f"preflight FAILED: {', '.join(c.name for c in failed)}")
        return False
    warned = [c for c in checks if not c.ok]
    log("preflight passed" + (f" with {len(warned)} warning(s)" if warned else ""))
    return True


def main() -> None:
    rclpy.init()
    node = Preflight()
    try:
        ok = report(node.collect(), lambda s: node.get_logger().info(s))
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
