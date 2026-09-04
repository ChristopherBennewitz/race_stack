#!/usr/bin/env python3
"""Record one joystick-driven system-identification take."""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

from id_controller import containment
from id_controller import preflight as pf
from id_controller import takes as takes_lib
from id_controller.run_take import (BAG_SETTLE_SEC, _actuation_configuration,
                                    _git, _run_preflight, _write_metadata)

HUMAN_DEADMAN_BUTTON = 4
JOY_WAIT_SEC = 10.0


class DeadmanMonitor(Node):
    """Track the human deadman button from the joystick stream."""

    def __init__(self, button: int = HUMAN_DEADMAN_BUTTON) -> None:
        super().__init__("sysid_teleop_recorder")
        self.button = button
        self.held = None
        self.create_subscription(Joy, "/joy", self._joy_cb, 10)

    def _joy_cb(self, message: Joy) -> None:
        self.held = (len(message.buttons) > self.button
                     and message.buttons[self.button] == 1)


def _wait_for(node: DeadmanMonitor, held: bool, timeout: float | None = None) -> bool:
    """Spin until the deadman has the requested state or the timeout expires."""
    end = None if timeout is None else time.monotonic() + timeout
    while rclpy.ok() and node.held is not held:
        if end is not None and time.monotonic() >= end:
            return False
        rclpy.spin_once(node, timeout_sec=0.1)
    return rclpy.ok()


def _spin_for(node: DeadmanMonitor, seconds: float) -> None:
    end = time.monotonic() + seconds
    while rclpy.ok() and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=min(0.1, end - time.monotonic()))


def _stop_bag(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _safe_label(label: str) -> str:
    result = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    return result.strip("_-") or "manual"


def _keep_recording() -> bool:
    try:
        return input("keep this recording? [Y/n] ").strip().lower() not in {"n", "no"}
    except EOFError:
        return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("label", nargs="?", default="manual", help="short bag label")
    ap.add_argument("--out", default=containment.default_bag_directory(),
                    help="bag directory (default: <race_stack>/sysid_bags)")
    ap.add_argument("--tail", type=float, default=1.0,
                    help="seconds to record after button 4 is released")
    ap.add_argument("--preflight-window", type=float, default=2.0)
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    if args.tail < 0.0:
        sys.exit("--tail must be non-negative")
    if shutil.which("ros2") is None:
        sys.exit("ros2 not on PATH")
    if not args.skip_preflight and not _run_preflight(False, args.preflight_window):
        sys.exit("aborting: preflight failed")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_name = f"sysid_{stamp}_teleop_{_safe_label(args.label)}"
    output = os.path.expanduser(args.out)
    bag_dir = os.path.join(output, bag_name)
    os.makedirs(output, exist_ok=True)

    rclpy.init()
    node = DeadmanMonitor()
    bag_proc = None
    started = None
    duration = 0.0
    outcome = "interrupted"
    try:
        print(f"waiting for /joy and button {HUMAN_DEADMAN_BUTTON} release...")
        if not _wait_for(node, False, JOY_WAIT_SEC):
            sys.exit(f"no released button {HUMAN_DEADMAN_BUTTON} on /joy after "
                     f"{JOY_WAIT_SEC:.0f} s")

        cmd = ["ros2", "bag", "record", "-o", bag_dir, *pf.BAG_TOPICS]
        print(f"recording lead-in -> {bag_dir}")
        bag_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        _spin_for(node, BAG_SETTLE_SEC)
        if bag_proc.poll() is not None:
            sys.exit("ros2 bag record exited immediately")

        print(f"press and hold button {HUMAN_DEADMAN_BUTTON} to drive and record")
        if not _wait_for(node, True):
            sys.exit("ROS shut down before recording started")
        started = time.monotonic()
        print(f"recording active; release button {HUMAN_DEADMAN_BUTTON} to stop")
        if not _wait_for(node, False):
            sys.exit("ROS shut down while recording")
        duration = time.monotonic() - started
        print(f"button released; recording {args.tail:.1f} s stop tail")
        _spin_for(node, args.tail)
        outcome = "complete"
    except KeyboardInterrupt:
        duration = 0.0 if started is None else time.monotonic() - started
        print("\ninterrupted; release button 4 before driving again")
    finally:
        if bag_proc is not None:
            _stop_bag(bag_proc)
        node.destroy_node()
        rclpy.try_shutdown()

    if not os.path.isdir(bag_dir):
        print("no bag was created")
        raise SystemExit(1)

    repo = os.path.dirname(os.path.abspath(takes_lib.__file__))
    _write_metadata(bag_dir, {
        "take": f"teleop_{_safe_label(args.label)}",
        "started": stamp,
        "duration_s": round(duration, 3),
        "racecar_version": os.environ.get("RACECAR_VERSION", "unset"),
        "outcome": outcome,
        "input": "joystick_teleop",
        "human_deadman_button": HUMAN_DEADMAN_BUTTON,
        "git_commit": _git(repo, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(repo, "status", "--porcelain")),
        "actuation_configuration": _actuation_configuration(
            os.environ.get("RACECAR_VERSION", "")) or "unavailable",
        "command_topics": {
            "requested": "/teleop",
            "applied_si_units": "/ackermann_cmd_applied",
            "final_motor_erpm": "/commands/motor/speed",
            "final_servo_position": "/commands/servo/position",
        },
        "containment_enabled": False,
        "note": "Joystick-driven data; fit from /ackermann_cmd_applied.",
    })

    if _keep_recording():
        print(f"kept {bag_name} ({duration:.1f} s driven)")
    else:
        shutil.rmtree(bag_dir)
        os.unlink(bag_dir + ".metadata.yaml")
        print(f"discarded {bag_name}")
    raise SystemExit(0 if outcome == "complete" else 1)


if __name__ == "__main__":
    main()
