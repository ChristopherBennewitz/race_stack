"""A multi-point turn for a planner that has stopped inside the track margin.

The progress MPC only drives forward and keeps a margin from the walls. Once the car stands still with its heading
turned against the track (after a spin, or in a hairpin that was taken too slowly), the plan has nowhere to go, the
planner asks for no speed, and it stays that way: the car needs reverse gear and several short arcs to turn around.

`Recovery` watches for that (the car standing and the planner asking for no movement for `stuck_seconds`), then takes
over the commands: it picks the gear and steering direction whose arc, driven for a short way, turns the heading best
toward the direction of the track without the outline of the car touching a wall, drives it until the walls get close,
stops, and does the same in the other gear, until the heading is within `exit_angle` of the track, the car is out of
the planner's keep-out band and there is room ahead. Then the planner has the car again.

This file is shared by f1tenth_parallel_sim (src/mpc/recovery.py) and the car's controller
(race_stack/controller/controller/mpc/recovery.py); a test keeps the two identical. It needs numpy only.
Poses are those of the centre of gravity (x forward, y left); `wall_distance` and `track_heading` come from the track.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# (N, 2) world points -> distance of each to the nearest wall, negative outside the track
WallDistance = Callable[[np.ndarray], np.ndarray]
# heading of the track, in the driving direction, nearest to a world point
TrackHeading = Callable[[float, float], float]


def wrap(angle: float) -> float:
    """The angle in [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class RecoveryConfig:
    """The car and the thresholds of the maneuver.

    `footprint` is the outline of the car as points in the pose's frame (x forward, y left); for a disk that stands for
    the car's collision circle, points spread on the circle.
    """

    wheelbase: float
    rear_to_cog: float
    footprint: tuple[tuple[float, float], ...]
    steer_gain: float = 1.0  # road-wheel angle per unit of steering command
    max_steer: float = 0.28  # steering command used for the arcs, rad
    maneuver_speed: float = 0.5  # m/s, in both gears
    stuck_speed: float = 0.05  # standing: slower than this
    request_speed: float = 0.15  # not asked to move: the planner's speed request is below this
    stuck_seconds: float = 1.0
    speed_filter_seconds: float = 0.3  # time constant of the speed filter: a stationary car reads noise, not zero
    probe_length: float = 0.6  # how far ahead a candidate arc is checked, and the longest segment, m
    stop_distance: float = 0.06  # the segment ends when less than this is free along it
    min_move: float = 0.08  # a candidate with less free path than this is not used
    clearance: float = 0.03  # the outline keeps this far from the walls, m
    slip: float = 0.01  # a car already closer than that may get this much closer, m
    settle_seconds: float = 0.35  # standstill between segments, for the car to stop and the steering to swing
    exit_angle: float = 0.6  # rad: the planner gets the car back when the heading is within this of the track
    exit_room: float = 0.3  # ... and at least this much is free straight ahead, m
    handback_clearance: float = 0.0  # ... and the CoG is at least this far from the walls: the planner's keep-out band
    violation_weight: float = 3.0  # rad of heading error that one metre inside the keep-out band counts as
    exit_violation: float = 0.02  # the band may still be violated by this much at hand-back, m
    stall_decisions: int = 2  # hand back anyway when this many arcs in a row improve nothing
    cooldown_seconds: float = 1.5  # after handing back, no new maneuver for this long
    max_seconds: float = 40.0
    step: float = 0.02  # spacing of the poses checked along an arc, m


class Recovery:
    """The state machine; call `update` every control period."""

    def __init__(self, config: RecoveryConfig, wall_distance: WallDistance, track_heading: TrackHeading) -> None:
        self.config = config
        self._wall_distance = wall_distance
        self._track_heading = track_heading
        self._footprint = np.asarray(config.footprint, dtype=float)
        self.active = False
        self.maneuvers = 0  # how often it took over
        self.segments = 0  # arcs driven, over all maneuvers
        self.failures = 0  # maneuvers that ran out of time, and the times both gears were blocked
        self.last_event = "idle"
        self._standing_since: float | None = None
        self._cooldown_until = -math.inf
        self._speed: float | None = None  # the filtered speed
        self._speed_time = 0.0
        self._reset_maneuver()

    def _reset_maneuver(self) -> None:
        self._mode = "settle"
        self._started_at = 0.0
        self._settle_until = 0.0
        self._direction = 1.0
        self._steer = 0.0
        self._next: tuple[float, float] | None = None
        self._stalled = 0  # arcs in a row that could not improve the heading or the distance from the walls
        self._driven = 0  # segments driven in this maneuver
        self._segment_start: tuple[float, float] = (0.0, 0.0)

    def reset(self) -> None:
        """Forgets a maneuver in progress and the standing timer, e.g. when the track or the estimate changes."""
        self.active = False
        self._standing_since = None
        self._speed = None
        self._reset_maneuver()

    # --- geometry ---------------------------------------------------------------------------------------

    def _arc(self, x: float, y: float, yaw: float, direction: float, steer: float) -> tuple[np.ndarray, np.ndarray]:
        """CoG poses `(K, 3)` and travelled distances `(K,)` along an arc of the kinematic bicycle.

        The rear axle moves along the arc, forward or backward according to `direction`; the CoG is `rear_to_cog` ahead
        of it. The heading changes by `direction * curvature` per metre.
        """
        c = self.config
        distance = np.arange(int(round(c.probe_length / c.step)) + 1) * c.step
        curvature = math.tan(c.steer_gain * steer) / c.wheelbase
        yaw_k = yaw + direction * curvature * distance
        rear = np.array([x, y]) - c.rear_to_cog * np.array([math.cos(yaw), math.sin(yaw)])
        if abs(curvature) < 1e-9:
            rear_k = rear + direction * distance[:, None] * np.array([math.cos(yaw), math.sin(yaw)])
        else:
            rear_k = rear + np.stack(
                [(np.sin(yaw_k) - math.sin(yaw)) / curvature, -(np.cos(yaw_k) - math.cos(yaw)) / curvature], axis=1
            )
        cog = rear_k + c.rear_to_cog * np.stack([np.cos(yaw_k), np.sin(yaw_k)], axis=1)
        return np.column_stack([cog, yaw_k]), distance

    def _free_length(self, poses: np.ndarray, distance: np.ndarray) -> float:
        """How far along the arc the outline of the car stays at least `clearance` from the walls."""
        cos, sin = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        fx, fy = self._footprint[:, 0], self._footprint[:, 1]
        world_x = poses[:, 0:1] + cos[:, None] * fx[None, :] - sin[:, None] * fy[None, :]
        world_y = poses[:, 1:2] + sin[:, None] * fx[None, :] + cos[:, None] * fy[None, :]
        points = np.stack([world_x.ravel(), world_y.ravel()], axis=1)
        near = self._wall_distance(points).reshape(world_x.shape).min(axis=1)
        # A car that starts closer to a wall than `clearance` may move as long as it does not get (noticeably) closer.
        limit = self.config.clearance if near[0] >= self.config.clearance else near[0] - self.config.slip
        blocked = np.flatnonzero(near < limit)
        if len(blocked) == 0:
            return float(distance[-1])
        return float(distance[blocked[0] - 1]) if blocked[0] > 0 else 0.0

    def _arc_free(self, x: float, y: float, yaw: float, direction: float, steer: float) -> tuple[float, float]:
        """(free length, heading after driving it) of an arc."""
        poses, distance = self._arc(x, y, yaw, direction, steer)
        free = self._free_length(poses, distance)
        index = min(int(round(free / self.config.step)), len(poses) - 1)
        return free, float(poses[index, 2])

    def _violation(self, x: float, y: float) -> float:
        """How far the CoG is inside the planner's keep-out band along the walls, m (0 outside the band)."""
        near = float(self._wall_distance(np.array([[x, y]]))[0])
        return max(0.0, self.config.handback_clearance - near)

    def _cost(self, x: float, y: float, yaw: float) -> float:
        """What the maneuver reduces: the heading error to the track plus the weighted keep-out band violation."""
        error = abs(wrap(self._track_heading(x, y) - yaw))
        return error + self.config.violation_weight * self._violation(x, y)

    def _choose(self, x: float, y: float, yaw: float) -> tuple[float, float, float] | None:
        """The (gear, steering, gain) whose free arc lowers `_cost` most; None if both gears are blocked.

        Ties go to driving forward, then to the longer free path.
        """
        before = self._cost(x, y, yaw)
        best, best_key = None, None
        for direction in (1.0, -1.0):
            for sign in (-1.0, 1.0):
                steer = sign * self.config.max_steer
                poses, distance = self._arc(x, y, yaw, direction, steer)
                free = self._free_length(poses, distance)
                if free < self.config.min_move:
                    continue
                end = poses[min(int(round(free / self.config.step)), len(poses) - 1)]
                gain = before - self._cost(*end)
                key = (round(gain, 2), direction, free)
                if best_key is None or key > best_key:
                    best, best_key = (direction, steer, gain), key
        return best

    # --- state machine ----------------------------------------------------------------------------------

    def update(
        self, now: float, x: float, y: float, yaw: float, speed: float, requested_speed: float
    ) -> tuple[float, float] | None:
        """The (speed, steer) to drive while the maneuver runs, or None to leave the car to the planner."""
        c = self.config
        speed = self._filter(now, speed)
        if not self.active and not self._triggered(now, speed, requested_speed):
            return None
        if not self.active:
            self.active = True
            self.maneuvers += 1
            self._reset_maneuver()
            self._started_at = self._settle_until = now
            self.last_event = "started"

        # Done when the heading is along the track and there is room to drive; at least one segment is driven first,
        # so a car that stands with a good heading still gets moved before the planner has it back.
        aligned = self._driven > 0 and abs(wrap(self._track_heading(x, y) - yaw)) < c.exit_angle
        clear = self._violation(x, y) <= c.exit_violation or self._stalled >= c.stall_decisions
        if aligned and clear and self._arc_free(x, y, yaw, 1.0, 0.0)[0] >= min(c.exit_room, c.probe_length):
            return self._finish(now, "turned around" if self._stalled < c.stall_decisions else "could not improve more")
        if now - self._started_at > c.max_seconds:
            self.failures += 1
            return self._finish(now, "timed out")

        if self._mode == "settle":
            return self._settle(now, x, y, yaw, speed)
        free, _ = self._arc_free(x, y, yaw, self._direction, self._steer)
        travelled = math.hypot(x - self._segment_start[0], y - self._segment_start[1])
        if free < c.stop_distance or travelled >= c.probe_length:
            self._mode = "settle"
            self._settle_until = now + c.settle_seconds
            self._next = self._choose(x, y, yaw)  # so the steering can swing while the car stops
            return 0.0, (self._next[1] if self._next else self._steer)
        return self._direction * c.maneuver_speed, self._steer

    def _filter(self, now: float, speed: float) -> float:
        """The speed through a first-order filter on the signed value, so that noise averages out."""
        if self._speed is None or now < self._speed_time:
            self._speed = speed
        else:
            weight = 1.0 - math.exp(-(now - self._speed_time) / self.config.speed_filter_seconds)
            self._speed += weight * (speed - self._speed)
        self._speed_time = now
        return self._speed

    def _triggered(self, now: float, speed: float, requested_speed: float) -> bool:
        c = self.config
        if now < self._cooldown_until:
            return False
        if not (abs(speed) < c.stuck_speed and requested_speed < c.request_speed):
            self._standing_since = None
            return False
        if self._standing_since is None:
            self._standing_since = now
        return now - self._standing_since >= c.stuck_seconds

    def _settle(self, now: float, x: float, y: float, yaw: float, speed: float) -> tuple[float, float]:
        c = self.config
        if now < self._settle_until or abs(speed) > 2.0 * c.stuck_speed:
            return 0.0, (self._next[1] if self._next else self._steer)
        choice = self._next
        if choice is None or self._arc_free(x, y, yaw, choice[0], choice[1])[0] < c.min_move:
            choice = self._choose(x, y, yaw)
        if choice is None:
            self.failures += 1
            self.last_event = "both gears blocked"
            self._next = None
            self._settle_until = now + 4.0 * c.settle_seconds
            return 0.0, 0.0
        self._direction, self._steer, gain = choice
        self._stalled = self._stalled + 1 if gain < 0.05 else 0
        self._next = None
        self._segment_start = (x, y)
        self._mode = "move"
        self._driven += 1
        self.segments += 1
        self.last_event = "reversing" if self._direction < 0 else "forward"
        return self._direction * c.maneuver_speed, self._steer

    def _finish(self, now: float, event: str) -> None:
        self.active = False
        self._standing_since = None
        self._cooldown_until = now + self.config.cooldown_seconds
        self.last_event = event
        self._reset_maneuver()
