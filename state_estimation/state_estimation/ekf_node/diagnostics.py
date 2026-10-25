"""Health numbers for the estimators (no ROS, so they can be tested offline).

The tuning in ``misc/ekf_tune.py`` rests on one statistic: the normalized innovation squared (NIS) of each sensor,
``y' S^-1 y`` with ``S = H P H' + R``, which averages the number of informative measurement dimensions when the noise
settings match the data. Offline it is read from a replay; here the same number is kept live, so a drive on another
floor, with another IMU or after a config change shows whether the noise still fits.
"""
from collections import deque

import numpy as np

INFORMATIVE_BELOW = 1e5  # a measurement variance above this switches that dimension off (the filters use 1e6)
NIS_HIGH = 2.0  # mean NIS / dof above this: the filter is over-confident (its R or Q too small)
NIS_LOW = 0.1  # below this: the noise settings are far too loose
MIN_SAMPLES = 100
MOVING_SPEED = 0.3  # m/s: below this and below MOVING_YAW_RATE the car is parked and its innovations are all noise floor
MOVING_YAW_RATE = 0.2  # rad/s


def is_moving(vx, yaw_rate):
    """Whether the car is driving. A parked car has no motor vibration and no motion, so every innovation is tiny and
    the NIS reads "noise far too loose" whatever the settings; it is only a calibration check while moving."""
    return abs(vx) > MOVING_SPEED or abs(yaw_rate) > MOVING_YAW_RATE


def informative_dimensions(R):
    """How many measurement dimensions actually constrain the state (the NIS' expected value)."""
    return int(np.sum(np.diag(np.atleast_2d(R)) < INFORMATIVE_BELOW))


def innovation_nis(P, z, hx, H, R, residual):
    """``(innovation, NIS)`` of measurement ``z`` against the prediction ``hx`` with Jacobian ``H``, state covariance ``P``."""
    y = residual(np.array(z, dtype=float), np.array(hx, dtype=float))
    S = H @ P @ H.T + R
    return y, float(y @ np.linalg.solve(S, y))


class SensorHealth:
    """What one input looks like from the filter: arrival rate and lag, age, and the windowed NIS.

    Args:
        name:    sensor name, for reports
        dof:     expected mean NIS (``informative_dimensions`` of its R)
        window:  how many updates the NIS mean covers
    """

    def __init__(self, name, dof, window=200):
        self.name = name
        self.dof = max(int(dof), 1)
        self._nis = deque(maxlen=window)
        self.samples = 0
        self.messages = 0
        self.last_receive = None  # clock seconds
        self.interval = None  # smoothed seconds between messages
        self.stamp_lag = None  # smoothed receive time minus header stamp, seconds

    def message(self, now, stamp=None):
        """A message arrived at clock time ``now`` carrying header time ``stamp`` (both seconds)."""
        if self.last_receive is not None and now > self.last_receive:
            gap = now - self.last_receive
            self.interval = gap if self.interval is None else self.interval + 0.05 * (gap - self.interval)
        if stamp is not None and stamp > 0.0:
            lag = now - stamp
            self.stamp_lag = lag if self.stamp_lag is None else self.stamp_lag + 0.05 * (lag - self.stamp_lag)
        self.last_receive = now
        self.messages += 1

    def record(self, nis):
        self._nis.append(float(nis))
        self.samples += 1

    @property
    def nis_mean(self):
        return float(np.mean(self._nis)) if self._nis else float('nan')

    @property
    def nis_ratio(self):
        return self.nis_mean / self.dof

    def age(self, now):
        return float('inf') if self.last_receive is None else now - self.last_receive

    def nis_verdict(self):
        """'' while the NIS is in its band (or there are too few samples), else what is wrong."""
        if len(self._nis) < MIN_SAMPLES:
            return ''
        if self.nis_ratio > NIS_HIGH:
            return f'{self.name} NIS {self.nis_mean:.1f} vs {self.dof} expected: over-confident'
        if self.nis_ratio < NIS_LOW:
            return f'{self.name} NIS {self.nis_mean:.2f} vs {self.dof} expected: noise far too loose'
        return ''

    def values(self, now):
        """The report as ``{key: number}``."""
        return {
            'rate_hz': 0.0 if not self.interval else 1.0 / self.interval,
            'age_s': self.age(now),
            'stamp_lag_ms': float('nan') if self.stamp_lag is None else 1e3 * self.stamp_lag,
            'nis_mean': self.nis_mean,
            'nis_expected': float(self.dof),
            'nis_ratio': self.nis_ratio,
            'nis_samples': float(self.samples),
        }
