import math
import time

import numpy as np

from controller.mpc.recovery import Recovery, RecoveryConfig


MAX_ODOM_AGE_MS = 50.0  # older odometry than this makes the MPC plan from where the car was, not is
SOLVE_SMOOTHING = 0.2  # weight of the newest solve in the smoothed solve time that sets the compute lag


def signed_area(xy):
    return 0.5 * float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]))


def oriented_centerline(xy, d_left, d_right, direction):
    """The centerline ordered in the driving direction ('ccw' / 'cw'); reversing swaps the left and right widths."""
    if direction not in ('ccw', 'cw'):
        raise ValueError("direction must be 'ccw' or 'cw'")
    xy = np.asarray(xy, dtype=np.float64)
    d_left = np.asarray(d_left, dtype=np.float64)
    d_right = np.asarray(d_right, dtype=np.float64)
    if (signed_area(xy) > 0.0) != (direction == 'ccw'):
        xy, d_left, d_right = xy[::-1].copy(), d_right[::-1].copy(), d_left[::-1].copy()
    return xy, d_left, d_right


def base_link_to_cog(x, y, yaw, vy, yaw_rate, l_r):
    """Moves a base_link (rear axle) pose and sideways velocity to the centre of gravity, ahead by `l_r`."""
    return x + l_r * np.cos(yaw), y + l_r * np.sin(yaw), vy + l_r * yaw_rate


def extrapolate_state(x, y, yaw, vx, vy, yaw_rate, dt):
    """The pose `dt` seconds later at constant body-frame velocity and yaw rate (the velocities stay as they are).

    The MPC plans from the state it is given as if it were the state when the command starts to act. What it is given is
    older than that (the estimator's lag, the message age, the solve), so it is moved forward by that time. The position
    integrates the velocity along the heading at the middle of the interval, which is exact to second order in `dt`.
    """
    heading = yaw + 0.5 * yaw_rate * dt
    c, s = np.cos(heading), np.sin(heading)
    return x + (vx * c - vy * s) * dt, y + (vx * s + vy * c) * dt, yaw + yaw_rate * dt


def track_geometry(xy, d_left, d_right):
    """``(wall_distance, heading)`` of a closed centerline in driving direction with its half-widths to the left and right.

    ``wall_distance(points)`` is the distance of each (N, 2) point to the nearer wall, negative outside the track, from the
    lateral offset to the nearest centerline point. ``heading(x, y)`` is the direction of the track there.
    """
    xy = np.asarray(xy, dtype=np.float64)
    d_left, d_right = np.asarray(d_left, dtype=np.float64), np.asarray(d_right, dtype=np.float64)
    tangent = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    left_normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)

    def nearest(points):
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return points, ((points[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)

    def wall_distance(points):
        points, index = nearest(points)
        offset = ((points - xy[index]) * left_normal[index]).sum(axis=1)
        return np.minimum(d_left[index] - offset, d_right[index] + offset)

    def heading(x, y):
        _, index = nearest([[x, y]])
        return math.atan2(tangent[index[0], 1], tangent[index[0], 0])

    return wall_distance, heading


def track_envelope(xy, d_left, d_right, clearance):
    """The region the MPC keeps the centre of gravity in: the track narrowed by `clearance` on both sides.

    `xy` is the closed centerline in driving direction with the track half-widths to its left and right (what
    `oriented_centerline` returns). Returns the left and right edge points, each (n, 2), in the same order.
    """
    xy = np.asarray(xy, dtype=np.float64)
    tangent = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    left_normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    left = xy + left_normal * (np.asarray(d_left, dtype=np.float64) - clearance)[:, None]
    right = xy - left_normal * (np.asarray(d_right, dtype=np.float64) - clearance)[:, None]
    return left, right


class MPC_Controller:
    """Runs the C++ progress MPC (no ROS). Driven by the controller manager at the MPC's control rate.

    Inputs: the track centerline with widths (set once) and the car state from the EKF in the map frame.
    Output: speed + steering angle for the actuation manager, which rate-limits them as the MPC assumed.
    """

    def __init__(self, direction, logger, track_spacing=0.1, clock=time.monotonic):
        try:
            from progress_mpc import ProgressMpc
        except ImportError as e:
            raise RuntimeError("MPC mode needs the progress_mpc package (it builds against acados; "
                               "see progress_mpc/README.md)") from e
        self.mpc = ProgressMpc(track_spacing)
        self.direction = direction
        self.last_cmd = None
        self.track = None  # the oriented centerline and widths handed to the solver
        self.clock = clock
        self.resets = 0
        self.jump_resets = 0
        self.solved_status = 'not solved'
        self.failures = 0
        self.lateral_error = float('nan')
        self._failures_reported = 0
        self._solve_seconds = None  # smoothed duration of the last solves, None until one has been timed
        self.recovery = None  # the multi-point turn for a car that stopped inside the track margin, if configured
        self._recovery_settings = None
        self._expected = None  # where the plan said the car would be one control period after the last solve, and when
        self._last_call = None
        self._window = self._empty_window()
        logger.info(f"Progress MPC: horizon {self.mpc.horizon} x {self.mpc.control_dt} s, "
                    f"speed <= {self.mpc.speed_max} m/s, steer <= {self.mpc.steer_max} rad")

    @property
    def cog_offset(self):
        """CoG to base_link (rear axle) distance, the car the MPC was exported for."""
        return self.mpc.cog_offset

    @property
    def control_hz(self):
        return 1.0 / self.mpc.control_dt

    @property
    def has_track(self):
        return self.mpc.has_track

    def set_centerline(self, xy, d_left, d_right):
        xy, d_left, d_right = oriented_centerline(xy, d_left, d_right, self.direction)
        self.mpc.set_track(xy, d_left, d_right)
        self.track = (xy, d_left, d_right)
        self.last_cmd = None
        self._build_recovery()

    def configure_recovery(self, wheelbase, steer_gain, radius=None, speed=0.4, clearance=0.06):
        """Turns on the multi-point turn for a car that stands still inside the track margin (see ``recovery.py``).

        ``wheelbase`` and ``steer_gain`` (road-wheel angle per unit of steering command) are those of the car the MPC was
        exported for; ``radius`` is the circle that stands for the car (default: the MPC's CoG clearance less its 0.1 m
        track margin), ``speed`` the speed in both gears and ``clearance`` how far from the walls the outline stays.
        """
        self._recovery_settings = {'wheelbase': wheelbase, 'steer_gain': steer_gain, 'radius': radius, 'speed': speed,
                                   'clearance': clearance}
        self._build_recovery()

    def _build_recovery(self):
        if self._recovery_settings is None or self.track is None:
            return
        settings = self._recovery_settings
        radius = settings['radius'] if settings['radius'] else max(self.mpc.clearance_margin - 0.1, 0.05)
        angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        config = RecoveryConfig(
            wheelbase=settings['wheelbase'], rear_to_cog=self.mpc.cog_offset,
            footprint=tuple((radius * math.cos(a), radius * math.sin(a)) for a in angles),
            steer_gain=settings['steer_gain'], max_steer=0.93 * self.mpc.steer_max, maneuver_speed=settings['speed'],
            clearance=settings['clearance'], handback_clearance=self.mpc.clearance_margin)
        self.recovery = Recovery(config, *track_geometry(*self.track))

    def envelope(self):
        """The (left, right) edges of the region the MPC keeps the centre of gravity in, or None before the track is set."""
        if self.track is None:
            return None
        return track_envelope(*self.track, self.mpc.clearance_margin)

    def reset(self):
        """Forgets the plan and the command history, e.g. after the state estimate was missing for a while."""
        if self.recovery is not None:
            self.recovery.reset()
        if self.last_cmd is not None:
            self.mpc.reset()
            self.last_cmd = None
            self._expected = None
            self.resets += 1

    def estimate_jumped(self):
        """The state estimate stepped (its correction was set from a SLAM pose): the plan is in the old frame."""
        if self.last_cmd is not None:
            self.jump_resets += 1
        self.reset()

    def compute_lag(self, apply_lag, fallback):
        """Seconds from the state sample the MPC plans from to its command acting on the car.

        The smoothed duration of the last solves plus `apply_lag` (publishing the command and the actuation manager's
        next tick), at most one control period; `fallback` until a solve has been timed. The solve is what changes when the
        CPU is busy or the track is complicated (26 ms on average at one point, 6 ms now), so a fixed value goes wrong
        exactly when the state needs moving forward the most.
        """
        if self._solve_seconds is None:
            return fallback
        return min(apply_lag + self._solve_seconds, self.mpc.control_dt)

    def note_odom_age(self, seconds):
        """How old the odometry used for this cycle was."""
        self._window['odom_age'].append(seconds)

    def plan_xy(self):
        """The last plan's predicted centre-of-gravity positions in the map frame, (horizon + 1, 2)."""
        return np.asarray(self.mpc.plan_world)

    def report_applied(self, steer, speed):
        """The command the actuation manager applied during the last control period."""
        self.mpc.report_applied(steer, speed)

    def main_loop(self, x, y, yaw, vx, vy, yaw_rate):
        """The next (speed, steer) for the centre-of-gravity state in the map frame (body-frame velocities)."""
        now = self.clock()
        window = self._window
        if self._last_call is not None:
            window['period'].append(now - self._last_call)
        self._last_call = now
        if self._expected is not None:
            (ex, ey), when = self._expected
            if 0.5 * self.mpc.control_dt < now - when < 2.0 * self.mpc.control_dt:  # a late cycle compares a different instant
                window['prediction_error'].append(float(np.hypot(x - ex, y - ey)))
        recovering = self.recovery is not None and self.recovery.active
        if recovering:
            steer, speed = 0.0, 0.0  # the planner is not asked while the maneuver drives
        else:
            steer, speed = self._plan(x, y, yaw, vx, vy, yaw_rate, now)
        if self.recovery is not None:
            takeover = self.recovery.update(now, x, y, yaw, vx, speed)
            if takeover is not None:
                speed, steer = takeover
                self.last_cmd = (steer, speed)
                self._expected = None  # the plan says nothing about where the maneuver takes the car
            elif recovering:  # handed back: the plan and the command history are from before the maneuver
                self.last_cmd = None
                self.mpc.reset()
                steer, speed = self._plan(x, y, yaw, vx, vy, yaw_rate, now)
        window['cycles'] += 1
        return speed, steer

    def _plan(self, x, y, yaw, vx, vy, yaw_rate, now):
        """One planner step, with its statistics; returns (steer, speed)."""
        window = self._window
        steer, speed = self.mpc.step(np.array([x, y, yaw, vx, vy, yaw_rate]))
        self.last_cmd = (steer, speed)

        stats = self.mpc.stats
        solve = float(stats['solve_seconds'])
        if self._solve_seconds is None:
            self._solve_seconds = solve
        else:
            self._solve_seconds += SOLVE_SMOOTHING * (solve - self._solve_seconds)
        window['solve_ms'].append(1e3 * stats['solve_seconds'])
        window['solver_ms'].append(1e3 * stats['solver_only_seconds'])
        window['iterations'].append(stats['iterations'])
        window['slack'].append(stats['slack_max'])
        self.solved_status = stats['status']
        self.failures = stats['failures']
        plan, frenet = np.asarray(self.mpc.plan_world), np.asarray(self.mpc.plan_frenet)
        if len(plan) > 1:
            self._expected = ((plan[1, 0], plan[1, 1]), now)
        if len(frenet):
            self.lateral_error = float(frenet[0, 1])  # the car's offset from the centerline
        return steer, speed

    @staticmethod
    def _empty_window():
        return {'period': [], 'prediction_error': [], 'solve_ms': [], 'solver_ms': [], 'iterations': [], 'slack': [],
                'odom_age': [], 'cycles': 0}

    def diagnostics(self):
        """Numbers over the cycles since the last call, and what is cumulative; ``{key: value}`` (NaN if no data).

        - ``prediction_error_*``: how far the car was from where the plan put it one period earlier (m). The clearest
          sign of model mismatch, or of a command that is not what the car did.
        - ``period_*``: time between cycles (should be ``control_dt``); ``solve_ms_*``: references + solver;
          ``slack_max``: the largest track-limit slack of an accepted plan (m).
        - ``failures``, ``resets``, ``jump_resets``: cumulative.
        - ``recovering``, ``recoveries``, ``recovery_failures``: the multi-point turn, if configured (``recovery.py``).
        """
        w, self._window = self._window, self._empty_window()
        stat = lambda key, fn: float(fn(w[key])) if len(w[key]) else float('nan')
        return {
            'cycles': float(w['cycles']),
            'control_dt_ms': 1e3 * self.mpc.control_dt,
            'period_mean_ms': 1e3 * stat('period', np.mean), 'period_max_ms': 1e3 * stat('period', np.max),
            'solve_ms_mean': stat('solve_ms', np.mean), 'solve_ms_max': stat('solve_ms', np.max),
            'solver_ms_max': stat('solver_ms', np.max),
            'iterations_mean': stat('iterations', np.mean), 'iterations_max': stat('iterations', np.max),
            'slack_max': stat('slack', np.max),
            'prediction_error_mean': stat('prediction_error', np.mean), 'prediction_error_max': stat('prediction_error', np.max),
            'odom_age_max_ms': 1e3 * stat('odom_age', np.max),
            'lateral_error': self.lateral_error, 'steer_estimate': float(self.mpc.steer_estimate),
            'failures': float(self.failures), 'resets': float(self.resets), 'jump_resets': float(self.jump_resets),
            'recovering': float(self.recovery is not None and self.recovery.active),
            'recoveries': float(self.recovery.maneuvers if self.recovery is not None else 0),
            'recovery_failures': float(self.recovery.failures if self.recovery is not None else 0),
        }

    def verdict(self, d):
        """``(level, message)`` for a ``diagnostics()`` dict: the first thing wrong, most severe first."""
        budget = d['control_dt_ms']
        if d['cycles'] == 0:
            return 'warn', 'no MPC cycles (waiting for the track or the odometry)'
        if d['recovering']:
            return 'warn', f'turning the car around: {self.recovery.last_event}'
        if d['failures'] > self._failures_reported:
            self._failures_reported = d['failures']
            return 'error', f"solver failed ({self.solved_status}); {int(d['failures'])} failures so far"
        self._failures_reported = d['failures']
        if d['solve_ms_max'] > budget:
            return 'warn', f"solve took {d['solve_ms_max']:.1f} ms, over the {budget:.0f} ms period"
        if d['period_max_ms'] > 2.0 * budget:
            return 'warn', f"cycle period reached {d['period_max_ms']:.0f} ms (nominal {budget:.0f})"
        if d['odom_age_max_ms'] > MAX_ODOM_AGE_MS:
            return 'warn', f"odometry up to {d['odom_age_max_ms']:.0f} ms old"
        if d['slack_max'] > 0.05:
            return 'warn', f"track limit violated by up to {100 * d['slack_max']:.0f} cm in the plan"
        if d['prediction_error_max'] > 0.2:
            return 'warn', f"the car was {100 * d['prediction_error_max']:.0f} cm from the plan's prediction"
        return 'ok', 'solving'

