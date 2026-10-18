"""The filter's recent estimates, for fusing measurements that describe the past."""
import math
from collections import deque

import numpy as np


class PredictionHistory:
    """Recent speed estimates, and the estimated IMU accelerations passed through the accelerometer's lag.

    Two measurements refer to an earlier time than the filter's present. A SLAM window speed is the mean speed
    over a window of past poses, so it is compared with the mean estimated speed over the same window. The
    accelerometer reports the acceleration through a first-order lag (the sensor's own low-pass), so it is
    compared with the estimated acceleration passed through the same lag. The innovation is applied to the
    present state through the present Jacobian, which assumes the estimation error changed little over the
    delay.

    Args:
        accel_lag: time constant of the accelerometer's first-order lag, seconds
        horizon:   how far back speeds are kept, seconds
    """

    def __init__(self, accel_lag, horizon=1.0):
        self.accel_lag = float(accel_lag)
        self.horizon = float(horizon)
        self._t = deque()
        self._vx = deque()
        self.accel = None

    def record(self, t, vx, accel):
        """Add the estimate at time ``t``: forward speed and the instantaneous [ax, ay] at the IMU."""
        if self._t and t <= self._t[-1]:
            return
        if self.accel is None:
            self.accel = np.array(accel, dtype=float)
        else:
            dt = t - self._t[-1]
            weight = 1.0 - math.exp(-dt / self.accel_lag) if self.accel_lag > 0.0 else 1.0
            self.accel = self.accel + weight * (np.asarray(accel, dtype=float) - self.accel)
        self._t.append(float(t))
        self._vx.append(float(vx))
        while len(self._t) > 2 and self._t[1] < t - self.horizon:
            self._t.popleft()
            self._vx.popleft()

    def mean_vx(self, t0, t1):
        """Mean estimated speed over ``[t0, t1]``, or None if the history does not reach back to ``t0``.

        Past the newest estimate the speed is held, which only matters for a window ending within a tick
        of the present.
        """
        if len(self._t) < 2 or t1 <= t0 or t0 < self._t[0]:
            return None
        t = np.asarray(self._t)
        vx = np.asarray(self._vx)
        inner = t[(t > t0) & (t < t1)]
        nodes = np.concatenate(([t0], inner, [t1]))
        values = np.interp(nodes, t, vx)
        return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(nodes)) / (t1 - t0))
