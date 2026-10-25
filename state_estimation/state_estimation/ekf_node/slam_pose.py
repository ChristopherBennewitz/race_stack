"""Fusing the SLAM map-frame pose into the global filter (no ROS, so it can be tested offline)."""
import numpy as np


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class SlamPoseUpdater:
    """Applies SLAM base_link poses ``[x, y, yaw]`` to a filter whose state is at the centre of gravity.

    The localization gets this filter's *local* sibling as its odometry prior, so its pose is not independent
    of the filter's motion estimate; ``R`` is therefore kept generous. A pose is stamped when its scan was
    matched, so it is compared with the pose the filter estimated then (``PredictionHistory``) and the
    innovation is applied to the present state.

    Times: a pose's stamp is when the TF chain was looked up, which trails its arrival by 3-14 ms; the stamp is
    what is used (the arrival only decides when it is applied). ``pose_delay`` is how much older than its stamp the
    pose's content is; ``scripts``/``ekf_replay.py --estimate-delay`` measure it.

    A pose whose normalized innovation exceeds ``gate`` (a chi-square threshold) is dropped, which keeps
    scan-matching glitches out. Poses that keep failing the gate mean the estimate, not the pose, is wrong
    (a kidnapping, a loop closure, a bad initialisation): after ``reseed_after`` in a row the state is
    re-seeded from the pose, a deliberate jump.

    Args:
        model:        a ``SimulatorSingleTrackModel`` (supplies ``Hx_pose`` and its Jacobian)
        R:            3x3 measurement covariance of ``[x, y, yaw]``
        history:      the filter's ``PredictionHistory``
        pose_delay:   how far the pose stamps lag the car, seconds
        gate:         chi-square threshold on the normalized innovation
        reseed_after: consecutive gated poses before the state is re-seeded
    """

    def __init__(self, model, R, history, pose_delay, gate, reseed_after):
        self.model = model
        self.R = np.asarray(R, dtype=float)
        self.history = history
        self.pose_delay = float(pose_delay)
        self.gate = float(gate)
        self.reseed_after = int(reseed_after)
        self.rejected = 0
        self.reseeds = 0
        self._streak = 0
        # The last fused or gated pose's innovation: (innovation, S), for offline scoring and diagnostics.
        self.last_innovation = None

    def seed(self, ekf, pose):
        """Places the state at ``pose`` with the pose's own uncertainty; velocities keep their estimate."""
        x, y, yaw = pose
        l_r = self.model.params.l_r
        self.history.rebase(self.model.Hx_pose(ekf.x), [x, y, yaw])  # the history follows the jump
        ekf.x[0] = x + l_r * np.cos(yaw)
        ekf.x[1] = y + l_r * np.sin(yaw)
        ekf.x[2] = yaw
        ekf.P[:3, :] = 0.0
        ekf.P[:, :3] = 0.0
        ekf.P[:3, :3] = self.R

    def update(self, ekf, pose, stamp):
        """Fuses one pose stamped ``stamp``. Returns 'updated', 'gated', 'reseeded' or 'skipped'."""
        predicted = self.history.pose_at(stamp - self.pose_delay)
        if predicted is None:
            return 'skipped'
        z = np.asarray(pose, dtype=float)
        H = self.model.HJacobian_pose(ekf.x)
        offset = predicted - self.model.Hx_pose(ekf.x)  # the estimate then, not now
        innovation = self._residual(z, predicted)
        S = H @ ekf.P @ H.T + self.R
        self.last_innovation = (innovation, S)
        if float(innovation @ np.linalg.solve(S, innovation)) > self.gate:
            self.rejected += 1
            self._streak += 1
            if self._streak >= self.reseed_after:
                self.seed(ekf, z)
                self.reseeds += 1
                self._streak = 0
                return 'reseeded'
            return 'gated'
        self._streak = 0
        ekf.update(z=z, HJacobian=self.model.HJacobian_pose, Hx=lambda x: self.model.Hx_pose(x) + offset,
                   R=self.R, residual=self._residual)
        return 'updated'

    @staticmethod
    def _residual(a, b):
        y = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        y[2] = wrap_angle(y[2])
        return y
