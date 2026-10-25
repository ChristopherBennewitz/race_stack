"""What the controller did in one bag, projected onto the track: laps, lateral offset and speed along the centerline.

Reads `/state_estimation/odom_map`, `/drive` and `/controller/diagnostics` with rosbags (no ROS install needed) and the
track's `global_waypoints.json` from `stack_master/maps/<map>/`. Used by plot_run.py.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

COG_OFFSET = 0.14766  # base_link (rear axle) to the centre of gravity, m
MIN_SPEED = 0.3  # m/s: slower than this the car counts as standing


def _typestore():
    from rosbags.typesys import Stores, get_typestore
    from rosbags.typesys.msg import get_types_from_msg

    store = get_typestore(Stores.ROS2_HUMBLE)
    store.register(get_types_from_msg(
        'float32 steering_angle\nfloat32 steering_angle_velocity\nfloat32 speed\nfloat32 acceleration\nfloat32 jerk',
        'ackermann_msgs/msg/AckermannDrive'))
    store.register(get_types_from_msg(
        'std_msgs/Header header\nackermann_msgs/AckermannDrive drive', 'ackermann_msgs/msg/AckermannDriveStamped'))
    return store


def _yaw(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def read_bag(path: str | Path) -> dict[str, np.ndarray]:
    """`odom` rows (t, x, y, yaw, vx, vy, wz), `drive` rows (t, speed, steer), `slack` rows (t, slack_max); t from the first message."""
    from rosbags.highlevel import AnyReader

    odom, drive, slack = [], [], []
    with AnyReader([Path(path)], default_typestore=_typestore()) as reader:
        keep = {"/state_estimation/odom_map", "/drive", "/controller/diagnostics"}
        for conn, stamp, raw in reader.messages(connections=[c for c in reader.connections if c.topic in keep]):
            msg = reader.deserialize(raw, conn.msgtype)
            t = stamp * 1e-9
            if conn.topic == "/state_estimation/odom_map":
                p, tw = msg.pose.pose, msg.twist.twist
                odom.append((t, p.position.x, p.position.y, _yaw(p.orientation), tw.linear.x, tw.linear.y, tw.angular.z))
            elif conn.topic == "/drive":
                drive.append((t, msg.drive.speed, msg.drive.steering_angle))
            else:
                for status in msg.status:
                    if status.name == "mpc":
                        values = {kv.key: float(kv.value) for kv in status.values}
                        slack.append((t, values.get("slack_max", float("nan"))))
    out = {"odom": np.array(odom), "drive": np.array(drive), "slack": np.array(slack)}
    t0 = min(a[0, 0] for a in out.values() if len(a))
    for a in out.values():
        if len(a):
            a[:, 0] -= t0
    return out


@dataclass
class Track:
    """The centerline in driving direction (counter-clockwise or clockwise as asked), with widths and arc length."""

    xy: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray  # left-pointing
    w_left: np.ndarray
    w_right: np.ndarray
    s: np.ndarray
    length: float

    @classmethod
    def load(cls, map_dir: str | Path, direction: str = "ccw") -> "Track":
        data = json.loads((Path(map_dir) / "global_waypoints.json").read_text())["centerline_waypoints"]
        points = data["wpnts"] if isinstance(data, dict) else data
        xy = np.array([[p["x_m"], p["y_m"]] for p in points])
        wl = np.array([p["d_left"] for p in points])
        wr = np.array([p["d_right"] for p in points])
        if np.allclose(xy[0], xy[-1]):  # the planner repeats the first point at the end
            xy, wl, wr = xy[:-1], wl[:-1], wr[:-1]
        area = 0.5 * np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
        if (area > 0) != (direction == "ccw"):
            xy, wl, wr = xy[::-1].copy(), wr[::-1].copy(), wl[::-1].copy()
        tangent = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        steps = np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(steps)[:-1]])
        return cls(xy, tangent, normal, wl, wr, s, float(steps.sum()))

    def project(self, x: np.ndarray, y: np.ndarray):
        """(s along the centerline, lateral offset with left positive, index of the nearest point) per position."""
        pts = np.stack([x, y], axis=1)
        idx = np.argmin(((pts[:, None, :] - self.xy[None]) ** 2).sum(-1), axis=1)
        off = pts - self.xy[idx]
        along = (off * self.tangent[idx]).sum(1)
        lateral = (off * self.normal[idx]).sum(1)
        return (self.s[idx] + along) % self.length, lateral, idx


@dataclass
class Lap:
    number: int
    t0: float
    t1: float
    complete: bool
    s: np.ndarray = field(repr=False)
    lateral: np.ndarray = field(repr=False)
    speed: np.ndarray = field(repr=False)
    xy: np.ndarray = field(repr=False)
    time: np.ndarray = field(repr=False)

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


def laps_from_bag(bag: dict[str, np.ndarray], track: Track) -> tuple[list[Lap], dict[str, np.ndarray]]:
    """Splits the drive into laps at the start/finish line (s = 0 of the centerline) and projects every sample."""
    odom = bag["odom"]
    yaw = odom[:, 3]
    x = odom[:, 1] + COG_OFFSET * np.cos(yaw)
    y = odom[:, 2] + COG_OFFSET * np.sin(yaw)
    s, lateral, idx = track.project(x, y)
    moving = odom[:, 4] > MIN_SPEED
    first, last = np.flatnonzero(moving)[[0, -1]]
    sl = slice(first, last + 1)
    ds = np.diff(s[sl]);  ds = (ds + track.length / 2) % track.length - track.length / 2
    unwrapped = s[first] + np.concatenate([[0.0], np.cumsum(ds)])
    lap_index = np.floor(unwrapped / track.length).astype(int)
    laps = []
    for k in np.unique(lap_index):
        member = np.flatnonzero(lap_index == k) + first
        covered = unwrapped[member - first]
        complete = covered.min() <= k * track.length + 0.3 and covered.max() >= (k + 1) * track.length - 0.3
        laps.append(Lap(number=len(laps) + 1, t0=odom[member[0], 0], t1=odom[member[-1], 0], complete=bool(complete),
                        s=unwrapped[member - first] - k * track.length, lateral=lateral[member], speed=odom[member, 4],
                        xy=np.stack([x[member], y[member]], axis=1), time=odom[member, 0]))
    info = {"s": s, "lateral": lateral, "idx": idx, "x": x, "y": y}
    return laps, info


def on_grid(lap: Lap, grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    """A lap's samples resampled onto the arc-length grid (NaN where the lap does not cover it)."""
    order = np.argsort(lap.s)
    s, v = lap.s[order], values[order]
    keep = np.concatenate([[True], np.diff(s) > 1e-6])
    out = np.interp(grid, s[keep], v[keep], left=np.nan, right=np.nan)
    return out
