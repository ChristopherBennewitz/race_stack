"""Helpers shared by the sys-ID recording entry points."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

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
    from id_controller import preflight as pf
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


def default_bag_directory(*starts) -> str:
    """Return ``sysid_bags`` in the nearest repository, with a home fallback."""
    configured_root = os.environ.get("RACE_STACK_ROOT")
    if configured_root:
        return str(Path(configured_root).expanduser() / "sysid_bags")
    candidates = starts or (Path.cwd(), Path(__file__).resolve())
    for candidate in candidates:
        path = Path(candidate).expanduser().resolve()
        if path.is_file():
            path = path.parent
        for directory in (path, *path.parents):
            if (directory / ".git").exists():
                return str(directory / "sysid_bags")
    return str(Path.home() / "sysid_bags")
