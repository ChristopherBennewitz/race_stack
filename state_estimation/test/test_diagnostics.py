"""The estimators' health numbers (no ROS)."""
import numpy as np
import pytest

from state_estimation.ekf_node.diagnostics import (MIN_SAMPLES, SensorHealth, informative_dimensions, innovation_nis)
from state_estimation.ekf_node.map_odom_correction import MapOdomCorrection


def residual(a, b):
    return a - b


def test_dimensions_switched_off_by_a_huge_variance_are_not_counted():
    assert informative_dimensions(np.diag([1e6, 0.01, 1.0, 1.0])) == 3
    assert informative_dimensions(np.diag([1e6, 1e6, 1e6, 0.02, 100.0, 0.8])) == 3
    assert informative_dimensions(np.diag([0.3, 0.3, 0.15])) == 3


def test_the_nis_of_data_that_matches_the_model_averages_the_dimensions():
    rng = np.random.default_rng(1)
    P = np.diag([0.04, 0.09, 0.01])
    R = np.diag([0.02, 0.01, 0.05])
    H = np.eye(3)
    S = H @ P @ H.T + R
    values = [innovation_nis(P, rng.multivariate_normal(np.zeros(3), S), np.zeros(3), H, R, residual)[1] for _ in range(4000)]
    assert np.mean(values) == pytest.approx(3.0, rel=0.08)


def test_the_nis_of_an_over_confident_filter_is_too_large():
    rng = np.random.default_rng(2)
    P, H = np.diag([0.004, 0.004]), np.eye(2)
    true_S = np.diag([0.044, 0.044])  # the data's noise is 0.04 per axis ...
    claimed_R = np.diag([0.004, 0.004])  # ... but the filter claims a tenth of that
    values = [innovation_nis(P, rng.multivariate_normal(np.zeros(2), true_S), np.zeros(2), H, claimed_R, residual)[1]
              for _ in range(2000)]
    assert np.mean(values) > 2.0 * 2 * 2  # over twice the expected value of 2 per dimension, with margin


def test_the_innovation_is_returned_with_the_residual_applied():
    wrap = lambda a, b: np.array([(a[0] - b[0] + np.pi) % (2 * np.pi) - np.pi])
    y, _ = innovation_nis(np.eye(1), [3.1], [-3.1], np.eye(1), np.eye(1), wrap)
    assert y[0] == pytest.approx(6.2 - 2 * np.pi)


def feed(health, mean_nis, count=MIN_SAMPLES):
    rng = np.random.default_rng(3)
    for value in rng.chisquare(health.dof, count) * mean_nis / health.dof:
        health.record(value)


def test_a_sensor_with_few_samples_is_not_judged():
    h = SensorHealth('imu', 4)
    feed(h, 40.0, count=MIN_SAMPLES - 1)
    assert h.nis_verdict() == ''


def test_a_calibrated_sensor_passes():
    h = SensorHealth('imu', 4)
    feed(h, 4.0, count=300)
    assert h.nis_verdict() == ''


def test_an_over_confident_sensor_is_reported():
    h = SensorHealth('vesc', 3)
    feed(h, 3 * 4.0, count=300)  # the old defaults measured 2 to 4 times their dimension
    assert 'over-confident' in h.nis_verdict()


def test_a_far_too_loose_sensor_is_reported():
    h = SensorHealth('imu', 4)
    feed(h, 0.1, count=300)
    assert 'too loose' in h.nis_verdict()


def test_the_window_forgets_old_samples():
    h = SensorHealth('imu', 4, window=50)
    feed(h, 40.0, count=300)
    feed(h, 4.0, count=60)
    assert h.nis_ratio < 2.0


def test_arrival_rate_lag_and_age_are_tracked():
    h = SensorHealth('pose', 3)
    for k in range(400):
        h.message(10.0 + k * 0.02, 10.0 + k * 0.02 - 0.007)  # 50 Hz, stamped 7 ms before it arrives
    v = h.values(10.0 + 399 * 0.02 + 0.3)
    assert v['rate_hz'] == pytest.approx(50.0, rel=0.01)
    assert v['stamp_lag_ms'] == pytest.approx(7.0, abs=0.1)
    assert v['age_s'] == pytest.approx(0.3)


def test_a_sensor_that_never_spoke_is_infinitely_old():
    assert SensorHealth('imu', 4).age(5.0) == float('inf')


def test_the_correction_counts_what_happened_to_each_pose():
    corr = MapOdomCorrection(0.15, 0.0, 2.0, 1.0, 3)
    for k in range(300):
        corr.add_local(k * 0.01, [3.0 * k * 0.01, 0.0, 0.0])
    assert corr.add_pose(1.0, [3.0, 0.1, 0.0]) == 'jumped'  # the first pose seeds it
    assert corr.add_pose(1.0, [3.0, 0.1, 0.0]) == 'skipped'  # a republished stamp
    assert corr.add_pose(1.02, [3.06, 0.1, 0.0]) == 'updated'
    for k in range(3):  # a kidnapping: rejected until it is accepted as a jump
        outcome = corr.add_pose(1.04 + 0.02 * k, [9.0, 5.0, 0.0])
    assert outcome == 'jumped'
    d = corr.diagnostics()
    assert (d['updated'], d['skipped'], d['jumped']) == (1.0, 1.0, 2.0)
    assert d['gated'] == 2.0 and corr.jumps == 1
    assert d['innovation_m'] > 5.0


def test_the_correction_reports_its_state_and_drift():
    corr = MapOdomCorrection(0.15, 0.0, 2.0, 1.0, 15)
    assert np.isnan(corr.diagnostics()['tx'])
    for k in range(300):
        corr.add_local(k * 0.01, [3.0 * k * 0.01, 0.0, 0.0])
    corr.add_pose(1.0, [5.0, 2.0, 0.3])
    d = corr.diagnostics()
    assert np.isnan(d['innovation_m'])
    assert d['theta'] == pytest.approx(0.3)
    assert d['rate_x'] == d['rate_y'] == d['rate_theta'] == 0.0
