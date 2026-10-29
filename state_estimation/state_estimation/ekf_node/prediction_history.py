"""The filter's recent estimates, for fusing measurements that describe the past."""
import math
from collections import deque

import numpy as np


class PredictionHistory:
    """Recent base_link poses, and the estimated IMU accelerations passed through the accelerometer's lag.

    Two measurements refer to an earlier time than the filter's present. A SLAM pose is stamped when the scan
    was matched, so ``MapOdomCorrection`` compares it with the pose the filter estimated then. The accelerometer reports the
    acceleration through a first-order lag (the sensor's own low-pass), so it is compared with the estimated
    acceleration passed through the same lag. The innovation is applied to the present state through the
    present Jacobian, which assumes the estimation error changed little over the delay.

    Args:
        accel_lag: time constant of the accelerometer's first-order lag, seconds
        horizon:   how far back poses are kept, seconds
    """

    def __init__(self, accel_lag, horizon=1.0):
        self.accel_lag = float(accel_lag)
        self.horizon = float(horizon)
        self._t = deque()
        self._pose = deque()
        self.accel = None

    def record(self, t, pose, accel):
        """Add the estimate at time ``t``: the base_link pose ``[x, y, yaw]`` and the instantaneous [ax, ay] at the IMU."""
        if self._t and t <= self._t[-1]:
            return
        if self.accel is None:
            self.accel = np.array(accel, dtype=float)
        else:
            dt = t - self._t[-1]
            weight = 1.0 - math.exp(-dt / self.accel_lag) if self.accel_lag > 0.0 else 1.0
            self.accel = self.accel + weight * (np.asarray(accel, dtype=float) - self.accel)
        self._t.append(float(t))
        self._pose.append(np.asarray(pose, dtype=float))
        while len(self._t) > 2 and self._t[1] < t - self.horizon:
            self._t.popleft()
            self._pose.popleft()

    def pose_at(self, t):
        """The estimated ``[x, y, yaw]`` at time ``t``, or None if the history does not reach back to ``t``.

        Linear between recorded ticks (the yaw along the shorter way round); past the newest estimate the
        newest is held, which only matters for a stamp within a tick of the present.
        """
        if len(self._t) < 2 or t < self._t[0]:
            return None
        times = np.asarray(self._t)
        j = int(np.searchsorted(times, t))
        if j >= len(times):
            return self._pose[-1].copy()
        if j == 0:
            return self._pose[0].copy()
        a, b = self._pose[j - 1], self._pose[j]
        f = (t - times[j - 1]) / (times[j] - times[j - 1])
        yaw_step = (b[2] - a[2] + math.pi) % (2.0 * math.pi) - math.pi
        return np.array([a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * yaw_step])
