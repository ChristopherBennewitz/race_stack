"""The smoothly corrected map -> odom (no ROS)."""
import math

import numpy as np

from state_estimation.ekf_node import defaults
from state_estimation.ekf_node.map_odom_correction import MapOdomCorrection, wrap_angle

DT = 0.01  # local filter tick
POSE_DT = 0.02  # SLAM pose period


def make(**overrides):
    c = {**defaults.CORRECTION, **overrides}
    return MapOdomCorrection(c['tau'], c['pose_delay'], c['gate_distance'], c['gate_angle'], c['reseed'], c['smooth'])


def path(seconds, speed=3.0, turn=0.0, bias=0.0):
    """(truth, local): base_link poses at the local rate. Local integrates the same speed with a heading bias rate."""
    n = int(seconds / DT)
    truth, local = np.zeros((n, 3)), np.zeros((n, 3))
    for k in range(1, n):
        for pose, extra in ((truth, 0.0), (local, bias * k * DT)):
            yaw = pose[k - 1, 2]
            pose[k] = [pose[k - 1, 0] + speed * math.cos(yaw + extra) * DT, pose[k - 1, 1] + speed * math.sin(yaw + extra) * DT,
                       yaw + turn * DT]
    return truth, local


def run(local, pose_of, count_pose=lambda k: True, corr=None, noise=None, seed=0, content_delay=None):
    """Feeds the local poses and, every POSE_DT, ``pose_of(k)`` (stamped like the tick, content ``pose_delay`` older).

    Returns (output poses, the correction)."""
    corr = corr or make()
    rng = np.random.default_rng(seed)
    out = np.zeros_like(local)
    delay_ticks = int(round((corr.pose_delay if content_delay is None else content_delay) / DT))
    for k in range(len(local)):
        t = k * DT
        corr.add_local(t, local[k])
        if k % int(round(POSE_DT / DT)) == 0 and k >= delay_ticks and count_pose(k):
            pose = pose_of(k - delay_ticks).copy()
            if noise:
                pose[:2] += rng.normal(0.0, noise, 2)
            corr.add_pose(t, pose)
        out[k] = corr.output(local[k], t)
    return out, corr


def position_error(out, truth, after=3.0):
    k = int(after / DT)
    return np.linalg.norm(out[k:, :2] - truth[k:, :2], axis=1)


def test_a_constant_offset_is_recovered_exactly():
    truth, local = path(5.0, turn=0.3)
    c, s = math.cos(0.7), math.sin(0.7)
    in_map = lambda p: np.array([c * p[0] - s * p[1] + 4.0, s * p[0] + c * p[1] - 2.0, wrap_angle(p[2] + 0.7)])
    out, corr = run(local, lambda k: in_map(local[k]))
    expected = np.array([in_map(p) for p in local])
    settled = slice(5, None)  # before that no pose has arrived
    assert np.abs(out[settled, :2] - expected[settled, :2]).max() < 1e-6
    assert np.abs([wrap_angle(a) for a in out[settled, 2] - expected[settled, 2]]).max() < 1e-6
    assert corr.rejected == 0 and corr.jumps == 0


def test_nothing_is_corrected_before_the_first_pose():
    corr = make()
    corr.add_local(0.0, [1.0, 2.0, 0.3])
    assert np.allclose(corr.output([1.0, 2.0, 0.3], 0.0), [1.0, 2.0, 0.3])
    assert corr.transform is None


def test_steady_drift_leaves_no_lag():
    """A heading bias makes the local odometry drift at a steady rate; a plain low-pass would trail it by rate * tau."""
    truth, local = path(12.0, turn=0.3, bias=0.01)
    out, corr = run(local, lambda k: truth[k])
    # Without the rate it would trail by drift rate * tau: about 0.36 m/s * 0.15 s = 5 cm by the end.
    assert position_error(out, truth, after=4.0).max() < 0.008
    assert corr.rejected == 0


def test_pose_noise_is_attenuated():
    truth, local = path(20.0, turn=0.2, bias=0.01)
    out, _ = run(local, lambda k: truth[k], noise=0.03)
    error = position_error(out, truth, after=3.0)
    assert np.sqrt((error ** 2).mean()) < 0.03  # the raw pose is 3 cm per axis, 4.2 cm in distance


def test_a_step_in_the_pose_is_absorbed_gradually():
    """A loop closure moves the pose by 0.4 m: the output follows at the filter's pace, without a jump or a rejection."""
    truth, local = path(8.0)
    step_at = int(4.0 / DT)
    step = np.array([0.0, 0.4, 0.0])
    out, corr = run(local, lambda k: truth[k] + (step if k >= step_at else 0.0))
    moves = np.linalg.norm(np.diff(out[:, :2], axis=0), axis=1) - 3.0 * DT  # a tick of the car's own motion removed
    after = slice(step_at, step_at + int(2.0 / DT))
    assert np.abs(moves[after]).max() < 0.1 * 0.4  # no tick takes more than a tenth of the step
    assert np.linalg.norm(out[-1, :2] - (truth[-1, :2] + step[:2])) < 0.01
    assert corr.rejected == 0 and corr.jumps == 0


def test_a_kidnapping_is_rejected_then_jumped_to_once():
    truth, local = path(8.0)
    kidnapped_at = int(4.0 / DT)
    out, corr = run(local, lambda k: truth[k] + (np.array([5.0, 0.0, 0.0]) if k >= kidnapped_at else 0.0))
    assert corr.jumps == 1
    assert corr.rejected == defaults.CORRECTION['reseed']
    assert np.linalg.norm(out[-1, :2] - (truth[-1, :2] + [5.0, 0.0])) < 0.01
    k = int((4.0 + defaults.CORRECTION['reseed'] * POSE_DT + 0.5) / DT)
    assert np.abs(np.diff(out[:k, 1])).max() < 0.05  # nothing moved before the jump (the car drives along x)


def test_the_output_lag_removes_the_step_each_pose_makes():
    truth, local = path(15.0, turn=0.2, bias=0.01)
    jitter = lambda out: np.sqrt((np.linalg.norm(np.diff(out[:, :2], 2, axis=0), axis=1) ** 2).mean()) / DT ** 2
    smooth, _ = run(local, lambda k: truth[k], noise=0.02, corr=make(smooth=0.05))
    plain, _ = run(local, lambda k: truth[k], noise=0.02, corr=make(smooth=0.0))
    assert jitter(smooth) < 0.6 * jitter(plain)
    assert position_error(smooth, truth, after=4.0).mean() < 0.02  # and it did not give up the tracking


def test_the_output_lag_adds_no_steady_state_lag_on_drift():
    truth, local = path(12.0, turn=0.3, bias=0.01)
    out, _ = run(local, lambda k: truth[k], corr=make(smooth=0.1))
    assert position_error(out, truth, after=4.0).max() < 0.01


def test_the_pose_delay_is_honoured():
    """The pose's content is older than its stamp by pose_delay; with that told, the offset is zero."""
    truth, local = path(6.0)
    out, _ = run(local, lambda k: truth[k], corr=make(pose_delay=0.02, smooth=0.0), content_delay=0.02)
    assert position_error(out, truth, after=1.0).max() < 1e-6
    wrong, _ = run(local, lambda k: truth[k], corr=make(pose_delay=0.0, smooth=0.0), content_delay=0.02)
    assert position_error(wrong, truth, after=1.0).mean() > 0.04  # 20 ms at 3 m/s is 6 cm


def test_a_republished_stamp_is_ignored():
    corr = make()
    for k in range(50):
        corr.add_local(k * DT, [3.0 * k * DT, 0.0, 0.0])
    assert corr.add_pose(0.40, [1.2, 0.0, 0.0]) == 'jumped'
    assert corr.add_pose(0.40, [9.0, 9.0, 0.0]) == 'skipped'
    assert corr.add_pose(0.30, [9.0, 9.0, 0.0]) == 'skipped'


def test_a_pose_older_than_the_history_is_skipped():
    corr = make()
    corr.add_local(10.0, [0.0, 0.0, 0.0])
    corr.add_local(10.01, [0.03, 0.0, 0.0])
    assert corr.add_pose(2.0, [0.0, 0.0, 0.0]) == 'skipped'
    assert corr.transform is None
