"""ROS-independent geometry for loosely containing identification takes."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np


DEFAULT_PARAMETERS = {
    "localization_wait_sec": 10.0,
    "pose_timeout_sec": 0.30,
    "jump_distance": 0.75,
    "jump_yaw": 0.80,
    "hard_clearance": 0.32,
    "path_clearance": 0.42,
    "slow_clearance": 0.75,
    "prediction_horizon": 0.80,
    "heading_gain": 0.35,
    "cross_track_gain": 0.35,
    "speed_softening": 0.60,
    "correction_max": 0.10,
    "correction_tau": 0.60,
    "correction_rate": 0.20,
    "max_cross_track_error": 0.75,
    "max_heading_error": 1.00,
}


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Return planar yaw from a quaternion."""
    return math.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))


def integrate_reference(commands: np.ndarray, reference_steer: np.ndarray,
                        dt: float, wheelbase: float) -> np.ndarray:
    """Integrate a kinematic reference path starting at (0, 0, 0)."""
    commands = np.asarray(commands, dtype=float)
    reference_steer = np.asarray(reference_steer, dtype=float)
    if commands.ndim != 2 or commands.shape[1] != 2:
        raise ValueError("commands must have shape (N, 2)")
    if reference_steer.shape != (len(commands),):
        raise ValueError("reference_steer must have shape (N,)")
    path = np.zeros((len(commands), 3), dtype=float)
    for i in range(1, len(commands)):
        speed = commands[i - 1, 1]
        steer = reference_steer[i - 1]
        yaw_mid = path[i - 1, 2] + 0.5 * dt * speed * math.tan(steer) / wheelbase
        path[i, 0] = path[i - 1, 0] + dt * speed * math.cos(yaw_mid)
        path[i, 1] = path[i - 1, 1] + dt * speed * math.sin(yaw_mid)
        path[i, 2] = wrap_angle(
            path[i - 1, 2] + dt * speed * math.tan(steer) / wheelbase)
    return path


def transform_path(path: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    """Rigidly anchor a local reference path at a map-frame pose."""
    result = np.array(path, dtype=float, copy=True)
    c, s = math.cos(yaw), math.sin(yaw)
    result[:, 0] = x + c * path[:, 0] - s * path[:, 1]
    result[:, 1] = y + s * path[:, 0] + c * path[:, 1]
    result[:, 2] = np.vectorize(wrap_angle)(path[:, 2] + yaw)
    return result


def tracking_errors(path: np.ndarray, x: float, y: float, yaw: float,
                    expected_index: int, rate_hz: float) -> tuple[float, float, int]:
    """Return cross-track/heading errors at the nearby closest path point."""
    lo = max(0, expected_index - int(round(3.0 * rate_hz)))
    hi = min(len(path), expected_index + int(round(1.0 * rate_hz)) + 1)
    segment = path[lo:hi]
    distances = np.square(segment[:, 0] - x) + np.square(segment[:, 1] - y)
    index = lo + int(np.argmin(distances))
    ref_x, ref_y, ref_yaw = path[index]
    dx, dy = x - ref_x, y - ref_y
    cross_track = -math.sin(ref_yaw) * dx + math.cos(ref_yaw) * dy
    heading = wrap_angle(ref_yaw - yaw)
    return float(cross_track), float(heading), index


def steering_correction(cross_track: float, heading_error: float, speed: float,
                        heading_gain: float, cross_track_gain: float,
                        speed_softening: float) -> float:
    """Stanley-like correction; positive steering turns left."""
    direction = 1.0 if speed >= 0.0 else -1.0
    return (direction * heading_gain * heading_error
            - math.atan2(cross_track_gain * cross_track,
                         abs(speed) + speed_softening))


@dataclass
class DistanceField:
    """Approximate clearance to occupied, unknown, or outside map space."""

    distances: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float

    @classmethod
    def from_grid(cls, data, width: int, height: int, resolution: float,
                  origin_x: float, origin_y: float, origin_yaw: float,
                  occupied_threshold: int = 50) -> "DistanceField":
        values = np.asarray(data, dtype=np.int16).reshape(height, width)
        unsafe = (values < 0) | (values >= occupied_threshold)
        unsafe[0, :] = True
        unsafe[-1, :] = True
        unsafe[:, 0] = True
        unsafe[:, -1] = True
        distances = np.full((height, width), np.inf, dtype=float)
        queue = []
        for row, col in np.argwhere(unsafe):
            distances[row, col] = 0.0
            heapq.heappush(queue, (0.0, int(row), int(col)))
        neighbours = ((-1, 0, resolution), (1, 0, resolution),
                      (0, -1, resolution), (0, 1, resolution),
                      (-1, -1, resolution * math.sqrt(2.0)),
                      (-1, 1, resolution * math.sqrt(2.0)),
                      (1, -1, resolution * math.sqrt(2.0)),
                      (1, 1, resolution * math.sqrt(2.0)))
        while queue:
            distance, row, col = heapq.heappop(queue)
            if distance != distances[row, col]:
                continue
            for dr, dc, cost in neighbours:
                nr, nc = row + dr, col + dc
                if 0 <= nr < height and 0 <= nc < width:
                    candidate = distance + cost
                    if candidate < distances[nr, nc]:
                        distances[nr, nc] = candidate
                        heapq.heappush(queue, (candidate, nr, nc))
        return cls(distances, resolution, origin_x, origin_y, origin_yaw)

    def clearance(self, x: float, y: float) -> float:
        """Return clearance at a map-frame point, or zero outside the grid."""
        dx, dy = x - self.origin_x, y - self.origin_y
        c, s = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        col = int(math.floor(local_x / self.resolution))
        row = int(math.floor(local_y / self.resolution))
        if not (0 <= row < self.distances.shape[0]
                and 0 <= col < self.distances.shape[1]):
            return 0.0
        return float(self.distances[row, col])

    def path_clearance(self, path: np.ndarray) -> float:
        """Return the minimum clearance along path centre points."""
        return min(self.clearance(float(p[0]), float(p[1])) for p in path)


def predicted_clearance(field: DistanceField, x: float, y: float, yaw: float,
                        speed: float, steer: float, horizon: float,
                        wheelbase: float, step: float = 0.1) -> float:
    """Return minimum clearance along a constant-input bicycle prediction."""
    result = field.clearance(x, y)
    count = max(1, int(math.ceil(horizon / step)))
    dt = horizon / count
    for _ in range(count):
        yaw_mid = yaw + 0.5 * dt * speed * math.tan(steer) / wheelbase
        x += dt * speed * math.cos(yaw_mid)
        y += dt * speed * math.sin(yaw_mid)
        yaw = wrap_angle(yaw + dt * speed * math.tan(steer) / wheelbase)
        result = min(result, field.clearance(x, y))
    return result


def predicted_twist_clearance(field: DistanceField, x: float, y: float,
                               yaw: float, vx: float, vy: float,
                               yaw_rate: float, horizon: float,
                               step: float = 0.1) -> float:
    """Predict clearance from measured body velocity, including lateral slip."""
    result = field.clearance(x, y)
    count = max(1, int(math.ceil(horizon / step)))
    dt = horizon / count
    for _ in range(count):
        yaw_mid = yaw + 0.5 * dt * yaw_rate
        x += dt * (vx * math.cos(yaw_mid) - vy * math.sin(yaw_mid))
        y += dt * (vx * math.sin(yaw_mid) + vy * math.cos(yaw_mid))
        yaw = wrap_angle(yaw + dt * yaw_rate)
        result = min(result, field.clearance(x, y))
    return result
