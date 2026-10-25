"""A map-frame pose that is as smooth as the local odometry: the local pose plus a slowly corrected ``map -> odom``.

No ROS here, so it can be replayed offline. The class holds a rigid 2-D transform ``T`` (``map <- odom``) and its rate
of change; the output is ``T`` applied to the local filter's base_link pose, so everything fast (jitter, velocity,
heading rate) is the local filter's own. Each SLAM pose gives a measurement of ``T`` (the pose composed with the inverse
of the local pose at the same moment), and ``T`` follows these through a critically damped alpha-beta filter of
bandwidth ``1 / tau``. What passes through is the local odometry's drift, which is what SLAM is for; the scan-to-scan
noise of the pose is attenuated.

Why the rate: the local odometry drifts at a roughly constant rate (a heading or speed bias), so ``T`` ramps. A plain
low-pass on a ramp trails it by ``rate * tau`` for ever; with the rate estimated there is no steady-state lag, and
between poses ``T`` is extrapolated, which keeps the output continuous instead of stepping at the pose rate.

A pose far from where the output says it should be is rejected; if that continues the transform is set from the pose,
a deliberate jump that ``jumps`` counts, so a consumer can drop its warm start.
"""
import math

import numpy as np

from .prediction_history import PredictionHistory


MAX_EXTRAPOLATION = 0.2  # seconds


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class MapOdomCorrection:
    """``map <- odom`` as ``(tx, ty, theta)`` followed through a low-pass on SLAM base_link poses.

    Args:
        tau:            time constant of the correction, seconds (larger = smoother, slower to follow drift)
        pose_delay:     how much older than its stamp a pose's content is, seconds (see ``SlamPoseUpdater``)
        gate_distance:  a pose farther than this from the output (metres) is rejected
        gate_angle:     a pose whose heading differs from the output by more than this (radians) is rejected
        reseed_after:   consecutive rejected poses before the correction is set from the pose
        smooth:         time constant of a further lag on the output correction, seconds (0 = none); removes the
                        small step each pose makes in the output, at the price of ``smooth`` seconds of lag in the
                        correction, which its rate compensates
        horizon:        how long the local poses are kept, seconds
    """

    def __init__(self, tau, pose_delay, gate_distance, gate_angle, reseed_after, smooth=0.0, horizon=2.0):
        self.tau = float(tau)
        self.pose_delay = float(pose_delay)
        self.gate_distance = float(gate_distance)
        self.gate_angle = float(gate_angle)
        self.reseed_after = int(reseed_after)
        self.smooth = float(smooth)
        self._lag = None  # the lagged correction, and the time it was last advanced to
        self._lag_time = None
        self.history = PredictionHistory(0.0, horizon)  # only its pose interpolation is used
        self.transform = None  # (tx, ty, theta) at ``self.time`` once the first pose arrived
        self.rate = np.zeros(3)  # d(transform)/dt
        self.time = None  # the local-odometry time the transform refers to (the first pose's, then each pose's, less its delay)
        self.rejected = 0
        self.jumps = 0
        self.last_innovation = None  # the pose minus what the output said at its time, [dx, dy, dyaw]
        self.counts = {'updated': 0, 'gated': 0, 'jumped': 0, 'skipped': 0}
        self._streak = 0
        self._last_stamp = None

    @property
    def last_stamp(self):
        """Stamp of the last SLAM pose that was used, or None."""
        return self._last_stamp

    def add_local(self, t, pose):
        """The local filter's base_link pose ``[x, y, yaw]`` in the odom frame at time ``t``."""
        self.history.record(t, pose, (0.0, 0.0))

    def add_pose(self, stamp, pose):
        """One SLAM base_link pose ``[x, y, yaw]`` stamped ``stamp``. Returns 'updated', 'gated', 'jumped' or 'skipped'."""
        outcome = self._add_pose(stamp, pose)
        self.counts[outcome] += 1
        return outcome

    def _add_pose(self, stamp, pose):
        if self._last_stamp is not None and stamp <= self._last_stamp:
            return 'skipped'  # a republished stamp carries no new scan match
        t = stamp - self.pose_delay
        local = self.history.pose_at(t)
        if local is None:
            return 'skipped'
        observed = np.array(self._transform_between(pose, local))
        if self.transform is None:
            self._set(observed, stamp, t)
            return 'jumped'
        predicted = self.to_map(local, t)
        innovation = np.array([pose[0] - predicted[0], pose[1] - predicted[1], wrap_angle(pose[2] - predicted[2])])
        self.last_innovation = innovation
        if math.hypot(innovation[0], innovation[1]) > self.gate_distance or abs(innovation[2]) > self.gate_angle:
            self.rejected += 1
            self._streak += 1
            if self._streak >= self.reseed_after:
                self._set(observed, stamp, t)
                self.jumps += 1
                self._streak = 0
                return 'jumped'
            return 'gated'
        self._streak = 0
        dt = t - self.time
        r = math.exp(-(stamp - self._last_stamp) / self.tau) if self.tau > 0.0 else 0.0
        g, h = 1.0 - r * r, (1.0 - r) ** 2  # critically damped alpha-beta gains
        ahead = np.array(self.transform) + self.rate * dt
        error = observed - ahead
        error[2] = wrap_angle(error[2])
        ahead += g * error
        ahead[2] = wrap_angle(ahead[2])
        self.rate = self.rate + h / dt * error
        self.transform = tuple(ahead)
        self.time, self._last_stamp = t, stamp
        return 'updated'

    def diagnostics(self):
        """The correction's state and the pose outcomes so far, as ``{key: number}``.

        ``innovation_m`` / ``innovation_yaw`` are how far the last pose was from the output (the tracking number the
        defaults were chosen with); ``rate_*`` is the local odometry's drift, per second.
        """
        tx, ty, theta = self.transform if self.transform is not None else (float('nan'),) * 3
        inn = self.last_innovation
        return {'tx': tx, 'ty': ty, 'theta': theta,
                'rate_x': self.rate[0], 'rate_y': self.rate[1], 'rate_theta': self.rate[2],
                'innovation_m': float('nan') if inn is None else math.hypot(inn[0], inn[1]),
                'innovation_yaw': float('nan') if inn is None else inn[2],
                **{k: float(v) for k, v in self.counts.items()}}

    def _set(self, transform, stamp, t):
        self.transform, self.rate = tuple(transform), np.zeros(3)
        self.time, self._last_stamp = t, stamp
        self._lag = None  # a jump is followed at once

    def to_map(self, pose, t=None):
        """A local ``[x, y, yaw]`` in the map frame at time ``t`` (the local pose itself until the first SLAM pose).

        ``T`` is extrapolated by its rate to ``t``, for at most ``MAX_EXTRAPOLATION`` seconds (if the poses stop, the
        correction does not run away).
        """
        if self.transform is None:
            return np.asarray(pose, dtype=float).copy()
        dt = 0.0 if t is None else min(max(t - self.time, -MAX_EXTRAPOLATION), MAX_EXTRAPOLATION)
        tx, ty, theta = np.array(self.transform) + self.rate * dt
        c, s = math.cos(theta), math.sin(theta)
        return np.array([c * pose[0] - s * pose[1] + tx, s * pose[0] + c * pose[1] + ty, wrap_angle(pose[2] + theta)])

    def output(self, pose, t):
        """A local ``[x, y, yaw]`` at time ``t`` in the map frame, through the output lag. Call with increasing ``t``."""
        if self.transform is None or self.smooth <= 0.0:
            return self.to_map(pose, t)
        tx, ty, theta = np.array(self.transform) + self.rate * min(max(t - self.time, -MAX_EXTRAPOLATION), MAX_EXTRAPOLATION)
        target = np.array([tx, ty, theta])
        if self._lag is None:
            self._lag = target
        elif t > self._lag_time:
            weight = 1.0 - math.exp(-(t - self._lag_time) / self.smooth)
            error = target - self._lag
            error[2] = wrap_angle(error[2])
            self._lag = self._lag + weight * error
        self._lag_time = t
        tx, ty, theta = self._lag + self.rate * self.smooth  # the lag trails a ramp by rate * smooth
        c, s = math.cos(theta), math.sin(theta)
        return np.array([c * pose[0] - s * pose[1] + tx, s * pose[0] + c * pose[1] + ty, wrap_angle(pose[2] + theta)])

    @staticmethod
    def _transform_between(map_pose, odom_pose):
        """The rigid transform that takes ``odom_pose`` onto ``map_pose``: ``(tx, ty, theta)``."""
        theta = wrap_angle(map_pose[2] - odom_pose[2])
        c, s = math.cos(theta), math.sin(theta)
        return (map_pose[0] - (c * odom_pose[0] - s * odom_pose[1]),
                map_pose[1] - (s * odom_pose[0] + c * odom_pose[1]), theta)
