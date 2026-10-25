"""Builds the policy's observation from the car's centerline and state estimate (numpy, no ROS)."""

import numpy as np
import torch

from controller.frames import pose_in_frame  # noqa: F401  (re-exported)
from controller.rl.policy import HISTORY_DIM


def _signed_area(xy):
    return 0.5 * float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]))


class Centerline:
    """A closed centerline with per-point track widths, tracking the nearest point incrementally.

    Mirrors `f1tenth_sim.track.Track.lookahead_waypoints`: the waypoints are the next `n` centerline
    *indices* (the sim's centerline is spaced ~0.1 m, as is the stack's `/centerline_waypoints`),
    expressed in the body frame as (x_rel, y_rel, d_left, d_right).
    """

    def __init__(self, xy, d_left, d_right, window=50, max_window_dist=2.0, direction='ccw'):
        xy = np.asarray(xy, dtype=np.float64)
        d_left = np.asarray(d_left, dtype=np.float64)
        d_right = np.asarray(d_right, dtype=np.float64)
        if direction not in ('ccw', 'cw'):
            raise ValueError("direction must be 'ccw' or 'cw'")
        # Mirrors `Track._orient_centerline`: the centerline is re-ordered to the driving direction the policy
        # was trained for (the sim trains counter-clockwise by default). Reversing also swaps which side is left.
        if (_signed_area(xy) > 0.0) != (direction == 'ccw'):
            xy, d_left, d_right = xy[::-1].copy(), d_right[::-1].copy(), d_left[::-1].copy()
        self.xy, self.d_left, self.d_right = xy, d_left, d_right
        self.n = len(self.xy)
        self.window = window
        self.max_window_dist = max_window_dist
        self.idx = None

    def nearest(self, x, y):
        """Index of the centerline point nearest (x, y); a windowed search around the last match,
        falling back to a global one at start-up or after a localization jump."""
        if self.idx is not None:
            cand = (self.idx + np.arange(-self.window, self.window + 1)) % self.n
            d2 = ((self.xy[cand] - (x, y)) ** 2).sum(axis=1)
            if d2.min() <= self.max_window_dist**2:
                self.idx = int(cand[d2.argmin()])
                return self.idx
        self.idx = int((((self.xy - (x, y)) ** 2).sum(axis=1)).argmin())
        return self.idx

    def lookahead(self, x, y, yaw, n_points):
        idx = self.nearest(x, y)
        fut = (idx + np.arange(1, n_points + 1)) % self.n
        rel = self.xy[fut] - (x, y)
        c, s = np.cos(yaw), np.sin(yaw)
        return np.stack(
            [rel[:, 0] * c + rel[:, 1] * s, -rel[:, 0] * s + rel[:, 1] * c, self.d_left[fut], self.d_right[fut]], axis=1
        ).astype(np.float32)


class HistoryBuffer:
    """Rolling window of [vel, yaw_rate, vel_cmd, steer_cmd] rows, newest last, zero-initialised."""

    def __init__(self, length):
        self.buf = torch.zeros(length, HISTORY_DIM)

    def push(self, vel, yaw_rate, vel_cmd, steer_cmd):
        self.buf = torch.cat([self.buf[1:], torch.tensor([[vel, yaw_rate, vel_cmd, steer_cmd]])])
        return self.buf
