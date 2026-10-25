import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'misc'))
from ekf_fast import FastFilter, FastGlobalFilter, FastGlobalReplay, FastReplay  # noqa: E402
from ekf_replay import Engine, FilterConfig, SlamConfig, load_pose_file, replay  # noqa: E402
from ekf_tune import (GLOBAL_PARAMETERS, PARAMETERS, nominal_vector, nominal_vector_global,  # noqa: E402
                      parse_constraints, to_config, to_global_config)
from state_estimation.ekf_node import defaults  # noqa: E402


def test_the_nominal_vector_reproduces_the_node_defaults():
    config = to_config(np.concatenate([nominal_vector(), np.zeros(2)]), fit_bias=True)
    np.testing.assert_allclose(np.diag(config.Q), defaults.NOISE['Q'])
    np.testing.assert_allclose(np.diag(config.R_imu), defaults.NOISE['R_imu'])
    np.testing.assert_allclose(np.diag(config.R_vesc), defaults.NOISE['R_vesc'])
    np.testing.assert_allclose(config.accel_bias, defaults.NOISE['imu_accel_bias'])


def test_bias_offsets_are_in_units_of_the_search_scale():
    config = to_config(np.concatenate([nominal_vector(), [2.0, -2.0]]), fit_bias=True)
    np.testing.assert_allclose(config.accel_bias, np.array(defaults.NOISE['imu_accel_bias']) + [1.0, -1.0])


def straight_drive_bag(seconds=8.0, speed=1.5):
    """A car at constant speed on a straight: consistent IMU, VESC and reference pose."""
    t = np.arange(0.0, seconds, 0.01)
    bias = defaults.NOISE['imu_accel_bias']
    zeros = np.zeros_like(t)
    # The IMU frame is base_link rotated by 90 degrees: zero acceleration plus the bias the replay removes.
    imu = np.column_stack([t, zeros, zeros, np.full_like(t, bias[1]), np.full_like(t, -bias[0])])
    cmd = np.column_stack([t, np.full_like(t, speed), np.zeros_like(t)])
    vesc = np.column_stack([t, np.zeros_like(t), np.zeros_like(t), np.zeros_like(t),
                            np.full_like(t, speed), np.zeros_like(t), np.zeros_like(t)])
    pose = np.column_stack([t, speed * t, np.zeros_like(t), np.zeros_like(t),
                            np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)])
    return {'imu': imu, 'cmd': cmd, 'odom': pose, 'vesc': vesc, 'live': np.zeros((0, 7))}


def test_replay_of_a_consistent_straight_drive_scores_well_and_tracks_the_speed():
    engine = Engine()
    rows, stats = replay(straight_drive_bag(), FilterConfig(), engine)
    assert np.isfinite(rows[:, :9]).all()  # the last two columns are the live EKF, absent from this bag
    assert abs(rows[-1, 1] - 1.5) < 0.1  # vx ends at the driven speed
    nll = sum(stats['nll'].values()) / sum(stats['count'].values())
    assert np.isfinite(nll)
    assert np.mean(stats['nis']['vesc']) < 6.0  # consistent data: no worse than the degrees of freedom


def test_noisier_noise_settings_score_a_worse_likelihood_on_clean_data():
    engine = Engine()
    bag = straight_drive_bag()
    base = FilterConfig()
    loose = FilterConfig(R_imu=base.R_imu * 1e4, R_vesc=base.R_vesc * 1e4)
    nll = lambda config: (lambda s: sum(s['nll'].values()) / sum(s['count'].values()))(replay(bag, config, engine)[1])
    assert nll(loose) > nll(base)


def with_pose(bag, speed=1.5, jump_at=None, jump=0.15, rate=80.0):
    """The straight drive's SLAM pose at ``rate``: stamped at the car's pose, received 10 ms later."""
    t = np.arange(0.0, bag['imu'][-1, 0], 1.0 / rate)
    x = speed * t + np.where((jump_at is not None) & (t >= (jump_at or 0)), jump, 0.0)
    bag = dict(bag)
    bag['pose'] = np.column_stack([t + 0.01, t, x, np.zeros_like(t), np.zeros_like(t)])
    return bag


def test_global_replay_anchors_the_pose_to_slam_and_is_consistent_on_clean_data():
    bag = with_pose(straight_drive_bag())
    rows, stats = replay(bag, FilterConfig(), Engine(), SlamConfig(delay=0.0))
    slam = stats['slam']
    assert slam['outcomes'].get('updated', 0) > 500 and slam['gated'] == 0 and slam['reseeds'] == 0
    assert np.mean(slam['nis']) < 3.0
    np.testing.assert_allclose(rows[-1, 11:14], [1.5 * rows[-1, 0], 0.0, 0.0], atol=0.1)  # the filter's base_link pose


def test_global_replay_absorbs_a_slam_step_gradually():
    bag = with_pose(straight_drive_bag(seconds=10.0), jump_at=5.0)
    responsive = SlamConfig(R=np.diag([0.01, 0.01, 0.005]), delay=0.0)  # a tight pose variance, not the node's default
    rows, stats = replay(bag, FilterConfig(), Engine(), responsive)
    x = rows[:, 11]
    steps = np.diff(x)[np.searchsorted(rows[:-1, 0], 5.0):]
    assert steps.max() - 1.5 * 0.01 < 0.02  # the 15 cm step is spread over many ticks
    assert abs(x[-1] - 1.5 * rows[-1, 0] - 0.15) < 0.03  # and ends up at the new pose


def test_the_default_global_filter_follows_a_slam_step_smoothly_and_without_a_reseed():
    bag = with_pose(straight_drive_bag(seconds=12.0), jump_at=5.0)
    config = FilterConfig.for_global()
    rows, stats = replay(bag, config, Engine(), SlamConfig(delay=0.0))
    x = rows[:, 11]
    steps = np.diff(x)[np.searchsorted(rows[:-1, 0], 5.0):]
    assert stats['slam']['reseeds'] == 0 and stats['slam']['gated'] == 0
    assert steps.max() - 1.5 * 0.01 < 0.25 * 0.15  # no tick takes more than a quarter of the 15 cm step (about 13 % in the first)
    assert abs(x[-1] - 1.5 * rows[-1, 0] - 0.15) < 0.08  # and it has taken most of the step within seven seconds


def test_pose_files_become_replay_rows(tmp_path):
    path = tmp_path / 'pose.csv'
    np.savetxt(path, np.array([[1.0, 2.0, 3.0, 0.5], [2.0, 4.0, 6.0, 0.6]]), delimiter=',')
    rows = load_pose_file(path, latency=0.02)
    np.testing.assert_allclose(rows, [[1.02, 1.0, 2.0, 3.0, 0.5], [2.02, 2.0, 4.0, 6.0, 0.6]])


def maneuvering_bag(seconds=12.0, seed=1):
    """A noisy drive with changing speed and steering, so the dynamic part of the model and every update matter."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, seconds, 0.02)  # 50 Hz sensors, as on the car
    speed = 1.5 + 1.0 * np.sin(0.7 * t)
    steer = 0.2 * np.sin(1.3 * t)
    bias = defaults.NOISE['imu_accel_bias']
    imu = np.column_stack([t, 0.3 * np.sin(t) + 0.01 * rng.normal(size=t.size), 0.8 * steer + 0.03 * rng.normal(size=t.size),
                           1.0 * rng.normal(size=t.size) + bias[1], 1.0 * rng.normal(size=t.size) - bias[0]])
    tc = np.arange(0.0, seconds, 0.01)
    cmd = np.column_stack([tc, np.interp(tc, t, speed), np.interp(tc, t, steer)])
    vesc = np.column_stack([t, np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), speed + 0.1 * rng.normal(size=t.size),
                            np.zeros_like(t), 0.8 * steer + 0.2 * rng.normal(size=t.size)])
    pose = np.column_stack([t, np.cumsum(speed) * 0.02, np.zeros_like(t), np.zeros_like(t), np.zeros((t.size, 3))])
    return {'imu': imu, 'cmd': cmd, 'odom': pose, 'vesc': vesc, 'live': np.zeros((0, 7))}


def test_the_compiled_evaluator_reproduces_the_python_replay():
    engine = Engine()
    bag = maneuvering_bag()
    fast = FastReplay(bag, FilterConfig(), engine, FastFilter(engine))
    rng = np.random.default_rng(0)
    for config in (FilterConfig(), to_config(nominal_vector() + rng.normal(0, 0.6, len(PARAMETERS)), fit_bias=False)):
        _, stats = replay(bag, config, engine)
        result = fast.run(config)
        for sensor in ('imu', 'vesc'):
            assert result['count'][sensor] == stats['count'][sensor]
            assert result['nll'][sensor] == pytest.approx(stats['nll'][sensor], rel=1e-6)
            assert result['nis_sum'][sensor] == pytest.approx(float(np.sum(stats['nis'][sensor])), rel=1e-6)


def test_constraints_parse_names_and_values():
    fixed, floors = parse_constraints(['R_vesc[vx]', 'Q[vy]=0.01'], ['R_imu[yaw]=0.5'])
    nominal = nominal_vector()
    assert fixed[PARAMETERS.index('R_vesc[vx]')] == pytest.approx(nominal[PARAMETERS.index('R_vesc[vx]')])
    assert fixed[PARAMETERS.index('Q[vy]')] == pytest.approx(-2.0)
    assert floors[PARAMETERS.index('R_imu[yaw]')] == pytest.approx(np.log10(0.5))
    with pytest.raises(SystemExit):
        parse_constraints(['R_vesc[typo]'], [])


def test_the_global_nominal_vector_reproduces_the_global_defaults():
    config, slam = to_global_config(nominal_vector_global(), defaults.SLAM['pose_delay'])
    np.testing.assert_allclose(np.diag(config.Q), defaults.NOISE_GLOBAL['Q'])
    np.testing.assert_allclose(np.diag(config.R_imu)[1:], defaults.NOISE_GLOBAL['R_imu'][1:])
    np.testing.assert_allclose(np.diag(config.R_vesc), defaults.NOISE_GLOBAL['R_vesc'])
    np.testing.assert_allclose(np.diag(slam.R), defaults.SLAM['R_slam_pose'])
    assert len(nominal_vector_global()) == len(GLOBAL_PARAMETERS)


def test_global_constraints_use_the_global_names():
    fixed, floors = parse_constraints(['R_slam[xy]=0.01'], ['R_slam[yaw]=0.001'], GLOBAL_PARAMETERS, nominal_vector_global())
    assert fixed[GLOBAL_PARAMETERS.index('R_slam[xy]')] == pytest.approx(-2.0)
    assert floors[GLOBAL_PARAMETERS.index('R_slam[yaw]')] == pytest.approx(-3.0)
    with pytest.raises(SystemExit):
        parse_constraints(['R_imu[yaw]'], [], GLOBAL_PARAMETERS, nominal_vector_global())  # not a global parameter


def noisy_pose_track(bag, seed=2, rate=40.0, jump_at=6.0):
    """A SLAM pose for the maneuvering bag: noisy, stamped before it arrives, with a step and a glitch to gate."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.5, bag['imu'][-1, 0] - 0.5, 1.0 / rate)
    x = np.interp(t, bag['odom'][:, 0], bag['odom'][:, 1]) + 0.02 * rng.normal(size=t.size) + np.where(t > jump_at, 0.3, 0.0)
    y = 0.02 * rng.normal(size=t.size)
    y[len(t) // 3] += 2.0  # one wild pose
    yaw = 0.01 * rng.normal(size=t.size)
    return np.column_stack([t + 0.012, t, x, y, yaw])


def test_the_compiled_global_evaluator_reproduces_the_python_replay():
    engine = Engine()
    bag = maneuvering_bag()
    pose = noisy_pose_track(bag)
    config = FilterConfig.for_global()
    fast = FastGlobalFilter(engine)
    for slam in (SlamConfig(pose=pose, delay=0.02),
                 SlamConfig(R=np.diag([0.0004, 0.0004, 0.0005]), delay=0.03, gate=5.0, reseed_after=3, pose=pose)):
        result = FastGlobalReplay(bag, config, slam, engine, fast).run(config, slam)
        _, stats = replay(bag, config, engine, slam)
        s = stats['slam']
        assert result['pose_accepted']['count'] == s['count'] and result['pose_nis_count'] == len(s['nis'])
        assert result['pose_accepted']['nll'] == pytest.approx(s['nll'], rel=1e-4)
        assert result['count']['pose'] == s['count_all'] and result['nll']['pose'] == pytest.approx(s['nll_clipped'], rel=1e-4)
        assert result['gated'] == s['gated'] and result['reseeds'] == s['reseeds']
        assert result['nis_sum']['pose'] == pytest.approx(float(np.sum(s['nis'])), rel=1e-4)
        for sensor in ('imu', 'vesc'):
            assert result['count'][sensor] == stats['count'][sensor]
            assert result['nll'][sensor] == pytest.approx(stats['nll'][sensor], rel=1e-4)
    assert result['reseeds'] >= 1 or result['gated'] >= 1  # the harsh setting exercises the gate
