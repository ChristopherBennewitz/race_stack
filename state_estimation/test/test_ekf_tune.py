import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'misc'))
from ekf_fast import FastFilter, FastReplay  # noqa: E402
from ekf_replay import Engine, FilterConfig, load_pose_file, replay  # noqa: E402
from ekf_tune import PARAMETERS, nominal_vector, parse_constraints, to_config  # noqa: E402
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
