#!/usr/bin/env python3
"""Run one identification take end to end: preflight, record, play, stop.

    ros2 run id_controller run_take M2_skidpad_L
    ros2 run id_controller run_take --list
    ros2 run id_controller run_take M1_circle_0.20_L --repeat 3 --out ~/sysid_bags

One take = one bag. The recording is started before the player and stopped after
it, so the countdown and the tail are both inside the bag, and the bag is named
after the take. Nothing is concatenated.

The metadata sidecar written next to each bag captures both the VESC conversion
and the physical-unit actuation limits.  Estimation should use
``/ackermann_cmd_applied`` as its input; ``/drive`` is the requested excitation.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime

from id_controller import preflight as pf
from id_controller import takes as takes_lib

BAG_SETTLE_SEC = 1.5    # let ros2 bag subscribe before the first command goes out


def _git(repo: str, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", repo, *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def _actuation_configuration(racecar_version: str) -> dict:
    """Read the conversion, limits and watchdog for this car, best effort."""
    conversion_keys = ("speed_to_erpm_gain", "speed_to_erpm_offset",
                       "steering_angle_to_servo_gain",
                       "steering_angle_to_servo_offset")
    manager_keys = ("speed_min", "speed_max", "max_acceleration",
                    "max_deceleration", "steering_min", "steering_max",
                    "max_steering_rate", "output_rate", "command_timeout",
                    "joy_timeout", "human_deadman_button",
                    "autonomous_deadman_button")
    path = None
    try:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory("stack_master"),
                            "config", racecar_version, "vesc.yaml")
    except Exception:
        pass
    if not path or not os.path.isfile(path):
        return {}
    try:
        import yaml
        with open(path) as fh:
            doc = yaml.safe_load(fh) or {}
        shared = doc.get("/**", {}).get("ros__parameters", {})
        manager = doc.get("actuation_manager", {}).get("ros__parameters", {})
        driver = doc.get("vesc_driver_node", {}).get("ros__parameters", {})
        return {
            "vesc_conversion": {
                k: shared[k] for k in conversion_keys if k in shared},
            "actuation_manager": {
                k: manager[k] for k in manager_keys if k in manager},
            "vesc_driver": {
                "command_timeout": driver.get("command_timeout")},
        }
    except Exception:
        return {}


def _write_metadata(bag_dir: str, info: dict) -> None:
    path = bag_dir + ".metadata.yaml"
    try:
        import yaml
        with open(path, "w") as fh:
            yaml.safe_dump(info, fh, sort_keys=False, default_flow_style=False)
    except Exception:
        with open(path, "w") as fh:
            for k, v in info.items():
                fh.write(f"{k}: {v}\n")
    print(f"  metadata -> {path}")


def _run_preflight(require_deadman: bool, window: float) -> bool:
    import rclpy
    rclpy.init()
    node = None
    try:
        node = pf.Preflight(node_name="sysid_preflight_runner")
        # Preflight reads its parameters into attributes in __init__, so override
        # the attributes directly rather than round-tripping through the
        # parameter server.
        node.require_deadman = require_deadman
        node.window = float(window)
        return pf.report(node.collect(), print)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


def run_one(take: str, args: argparse.Namespace, index: int, total: int) -> int:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_r{index}" if total > 1 else ""
    bag_name = f"sysid_{stamp}_{take}{suffix}"
    bag_dir = os.path.join(os.path.expanduser(args.out), bag_name)
    duration = len(takes_lib.build(take)) / args.rate

    print(f"\n=== {take}  ({index}/{total})  {duration:.1f} s ===")

    if not args.skip_preflight:
        if not _run_preflight(not args.no_deadman, args.preflight_window):
            print("aborting: preflight failed")
            return 1

    bag_proc = None
    if not args.no_bag:
        os.makedirs(os.path.expanduser(args.out), exist_ok=True)
        cmd = ["ros2", "bag", "record", "-o", bag_dir, *pf.BAG_TOPICS]
        print(f"  recording -> {bag_dir}")
        bag_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.STDOUT)
        time.sleep(BAG_SETTLE_SEC)
        if bag_proc.poll() is not None:
            print("aborting: ros2 bag record exited immediately")
            return 1

    player = [
        "ros2", "run", "id_controller", "play_take", "--ros-args",
        "-p", f"take:={take}",
        "-p", f"rate_hz:={args.rate}",
        "-p", f"countdown:={args.countdown}",
        "-p", f"require_deadman:={'false' if args.no_deadman else 'true'}",
        "-p", f"dry_run:={'true' if args.dry_run else 'false'}",
    ]
    code = subprocess.call(player)

    if bag_proc is not None:
        bag_proc.send_signal(signal.SIGINT)
        try:
            bag_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            bag_proc.kill()
            bag_proc.wait()
        repo = os.path.dirname(os.path.abspath(takes_lib.__file__))
        _write_metadata(bag_dir, {
            "take": take,
            "started": stamp,
            "nominal_duration_s": round(duration, 3),
            "control_hz": args.rate,
            "racecar_version": os.environ.get("RACECAR_VERSION", "unset"),
            "player_exit_code": code,
            "outcome": {0: "complete", 3: "aborted"}.get(code, f"error_{code}"),
            "git_commit": _git(repo, "rev-parse", "HEAD"),
            "git_dirty": bool(_git(repo, "status", "--porcelain")),
            "actuation_configuration": _actuation_configuration(
                os.environ.get("RACECAR_VERSION", "")) or "unavailable",
            "command_topics": {
                "requested": "/drive",
                "applied_si_units": "/ackermann_cmd_applied",
                "final_motor_erpm": "/commands/motor/speed",
                "final_servo_position": "/commands/servo/position",
            },
            "note": ("Fit vehicle dynamics from /ackermann_cmd_applied. The "
                     "configured steering conversion already includes the "
                     "former x1.3 correction."),
        })

    if code == 0:
        print(f"  OK  {bag_name}")
    else:
        print(f"  take did not complete (exit {code}) -- redo it, do not salvage it")
    return code


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("takes", nargs="*", help="take names, or 'room' / 'corridor' / 'all'")
    ap.add_argument("--list", action="store_true", help="print the manifest and exit")
    ap.add_argument("--out", default="~/sysid_bags", help="directory for the bags")
    ap.add_argument("--repeat", type=int, default=1, help="runs per take")
    ap.add_argument("--rate", type=float, default=float(takes_lib.HZ),
                    help="control rate; must match extract_bags.py --control-hz")
    ap.add_argument("--countdown", type=float, default=3.0)
    ap.add_argument("--preflight-window", type=float, default=2.0)
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--no-deadman", action="store_true",
                    help="do not require joystick button 5 (bench testing only)")
    ap.add_argument("--no-bag", action="store_true", help="do not record")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the timing loop but publish no commands")
    args = ap.parse_args()

    if args.list or not args.takes:
        for name in takes_lib.TAKES:
            print(takes_lib._summary(name))
        print()
        for name in takes_lib.CORRIDOR:
            print(takes_lib._summary(name))
        if not args.takes:
            print("\nnothing to run; pass take names, or 'room' / 'corridor' / 'all'")
        return

    if not args.no_bag and shutil.which("ros2") is None:
        sys.exit("ros2 not on PATH")

    groups = {"room": list(takes_lib.TAKES),
              "corridor": list(takes_lib.CORRIDOR),
              "all": list(takes_lib.ALL)}
    selected = []
    for item in args.takes:
        selected.extend(groups.get(item, [item]))
    unknown = [t for t in selected if t not in takes_lib.ALL]
    if unknown:
        sys.exit(f"unknown take(s): {', '.join(unknown)}  (--list to see them)")

    plan = [t for t in selected for _ in range(args.repeat)]
    print(f"{len(plan)} run(s) into {os.path.expanduser(args.out)}")
    failures = []
    for i, take in enumerate(plan, 1):
        if run_one(take, args, i, len(plan)) != 0:
            failures.append(take)
            if input("  continue with the rest? [y/N] ").strip().lower() != "y":
                break

    print(f"\ndone: {len(plan) - len(failures)}/{len(plan)} complete")
    if failures:
        print("redo: " + ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
