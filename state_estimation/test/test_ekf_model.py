import numpy as np
import pytest

from state_estimation.ekf_node.models import CommandDelay, SimulatorSingleTrackModel, VehicleModelParams
from state_estimation.ekf_node.prediction_history import PredictionHistory
from state_estimation.ekf_node.slam_pose import SlamPoseUpdater


def make_model(imu_x=0.07):
    params = VehicleModelParams(
        m=3.718, I_z=0.0561, l_f=0.17334, l_r=0.14766, h_cg=0.074,
        Bf=12.0306, Cf=1.71178, Df=0.402674, Ef=0.0,
        Br=5.06904, Cr=1.64946, Dr=0.572528, Er=0.0,
        steer_gain=1.6746, steer_bias=0.000807634, steer_delay=0.0871629, steer_latency=0.0338089, speed_delay=0.05,
        drive_gain=3.99169, brake_gain=5.51182, imu_x=imu_x,
    )
    return SimulatorSingleTrackModel(params, np.eye(7) * 0.01)


class _FakeEkf:
    """Minimal stand-in for filterpy's ExtendedKalmanFilter, exposing x/P."""

    def __init__(self, x, P):
        self.x = x
        self.P = P


def test_stationary_model_has_finite_transition_and_jacobian():
    model = make_model()
    state = np.zeros(7)
    control = np.array([1.0, 0.2])  # [speed_command, delayed steer_command]
    transitioned = np.array(model.f(state, control, 0.01)).flatten()
    jacobian = np.array(model.F(state, control, 0.01))
    assert np.isfinite(transitioned).all()
    assert np.isfinite(jacobian).all()
    assert jacobian.shape == (7, 7)


def test_imu_measurement_model_is_finite():
    model = make_model()
    state = np.array([0.0, 0.0, 0.0, 2.0, 0.1, 0.05, 0.1])
    control = np.array([0.5, 0.15])
    measurement = model.Hx_imu(state, control)
    jacobian = model.HJacobian_imu(state, control)
    assert np.isfinite(measurement).all()
    assert measurement.shape == (4,)
    assert jacobian.shape == (4, 7)
    assert np.isfinite(jacobian).all()


def test_predict_relaxes_towards_kinematic_yaw_rate_at_low_speed():
    # At low speed the kinematic/dynamic blend weight is close to zero, so the
    # yaw rate should settle near the pure kinematic bicycle relation.
    model = make_model()
    ekf = _FakeEkf(np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]), model.P_0.copy())
    control = np.array([0.1, 0.1])  # hold a small steering command and the current speed

    for _ in range(200):
        model.predict(ekf, dt=0.01, u=control)

    wheelbase = 0.17334 + 0.14766
    steer = 1.6746 * control[1] + 0.000807634  # steady-state road-wheel angle
    expected_yaw_rate = ekf.x[3] * np.tan(steer) / wheelbase
    assert abs(ekf.x[6] - steer) < 1e-3
    assert np.isfinite(ekf.x).all()
    assert abs(ekf.x[5] - expected_yaw_rate) < 0.01


def test_speed_loop_accelerates_towards_commanded_speed():
    model = make_model()
    ekf = _FakeEkf(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), model.P_0.copy())
    for _ in range(100):
        model.predict(ekf, dt=0.01, u=np.array([3.0, 0.0]))
    assert 1.0 < ekf.x[3] < 3.0


def test_steering_follows_a_single_lag_toward_the_delayed_command():
    model = make_model()
    ekf = _FakeEkf(np.zeros(7), model.P_0.copy())
    for _ in range(400):  # 4 s, many lag time constants
        model.predict(ekf, dt=0.01, u=np.array([0.0, 0.1]))
    assert abs(ekf.x[6] - (1.6746 * 0.1 + 0.000807634)) < 1e-3


def test_command_delay_holds_the_last_command_issued_before_the_delayed_time():
    d = CommandDelay(0.1)
    assert d.at(1.0) == 0.0  # nothing received yet
    for k in range(11):  # command steps 0 -> 1 in tenths, each held for 0.1 s
        d.push(k * 0.1, k * 0.1)
    assert abs(d.at(0.6) - 0.5) < 1e-9  # the command issued at 0.5, 0.1 s earlier
    assert abs(d.at(0.65) - 0.5) < 1e-9  # still held, not interpolated toward 0.6
    assert abs(d.at(5.0) - 1.0) < 1e-9  # held at the newest
    assert abs(d.at(0.05) - 0.0) < 1e-9  # before the first sample


def test_command_delay_forgets_old_samples():
    d = CommandDelay(0.1, horizon=0.5)
    for k in range(1000):
        d.push(k * 0.01, k * 0.01)
    assert len(d._t) < 100


def _integrate(model, x0, u, total, dt):
    x = np.asarray(x0, dtype=float)
    for _ in range(round(total / dt)):
        x = np.array(model.f(x, u, dt)).flatten()
    return x


def test_prediction_converges_at_fourth_order():
    model = make_model()
    x0 = [0.0, 0.0, 0.1, 2.0, 0.05, 0.3, 0.1]
    u = np.array([3.0, 0.15])
    reference = _integrate(model, x0, u, 0.2, 0.2 / 256)
    coarse = np.linalg.norm(_integrate(model, x0, u, 0.2, 0.2 / 4) - reference)
    fine = np.linalg.norm(_integrate(model, x0, u, 0.2, 0.2 / 8) - reference)
    assert coarse / fine > 10.0  # Euler would give about 2


def test_imu_accelerations_are_moved_from_the_cog_to_the_imu():
    at_cog = make_model(imu_x=0.14766)  # IMU on the CoG: no lever arm
    at_imu = make_model(imu_x=0.07)
    state = np.array([0.0, 0.0, 0.0, 2.0, 0.1, 1.5, 0.2])
    u = np.array([2.0, 0.1])
    lever = 0.07 - 0.14766
    yaw_accel = np.array(at_cog.f(state, u, 1e-6)).flatten()[5] - state[5]
    difference = at_imu.Hx_imu(state, u) - at_cog.Hx_imu(state, u)
    assert abs(difference[2] - (-state[5] ** 2 * lever)) < 1e-9
    assert abs(difference[3] - yaw_accel / 1e-6 * lever) < 1e-4


def test_history_interpolates_the_pose_at_a_past_time():
    history = PredictionHistory(accel_lag=0.05)
    for k in range(101):  # 2 m/s along x while turning at 1 rad/s, recorded at 100 Hz
        history.record(k * 0.01, [2.0 * k * 0.01, 0.0, k * 0.01], [0.0, 0.0])
    np.testing.assert_allclose(history.pose_at(0.305), [0.61, 0.0, 0.305], atol=1e-12)  # between ticks
    np.testing.assert_allclose(history.pose_at(5.0), [2.0, 0.0, 1.0])  # past the newest: held
    assert history.pose_at(-0.1) is None  # older than the history


def test_history_interpolates_yaw_the_short_way_round_the_wrap():
    history = PredictionHistory(accel_lag=0.05)
    history.record(0.0, [0.0, 0.0, np.pi - 0.1], [0.0, 0.0])
    history.record(1.0, [0.0, 0.0, -np.pi + 0.1], [0.0, 0.0])
    yaw = history.pose_at(0.5)[2]
    assert abs(abs(yaw) - np.pi) < 1e-12


def test_history_lags_the_predicted_acceleration_like_the_accelerometer():
    history = PredictionHistory(accel_lag=0.05)
    history.record(0.0, 0.0, [0.0, 0.0])
    for k in range(1, 6):  # a step to 1 m/s^2, followed for exactly one time constant
        history.record(k * 0.01, 0.0, [1.0, -2.0])
    np.testing.assert_allclose(history.accel, [1.0 - np.exp(-1.0), -2.0 * (1.0 - np.exp(-1.0))], atol=1e-9)


def test_pose_measurement_is_the_rear_axle_behind_the_cog():
    model = make_model()
    state = np.array([1.0, 2.0, np.pi / 2, 3.0, 0.1, 0.2, 0.05])
    np.testing.assert_allclose(model.Hx_pose(state), [1.0, 2.0 - model.params.l_r, np.pi / 2], atol=1e-12)
    eps = 1e-6
    numeric = np.column_stack([(model.Hx_pose(state + eps * e) - model.Hx_pose(state - eps * e)) / (2 * eps)
                               for e in np.eye(7)])
    np.testing.assert_allclose(model.HJacobian_pose(state), numeric, atol=1e-6)


class _EkfWithUpdate(_FakeEkf):
    """The textbook EKF update, standing in for filterpy's (same signature, Joseph form)."""

    def update(self, z, HJacobian, Hx, R, residual=np.subtract):
        H = HJacobian(self.x)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ residual(z, Hx(self.x))
        I_KH = np.eye(len(self.x)) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T


def make_updater(history, **overrides):
    kwargs = dict(R=np.diag([0.01, 0.01, 0.005]), pose_delay=0.0, gate=16.0, reseed_after=3)
    kwargs.update(overrides)
    return SlamPoseUpdater(make_model(), history=history, **kwargs)


def history_at_rest(pose, until=1.0):
    history = PredictionHistory(accel_lag=0.05)
    for k in range(int(until * 100) + 1):
        history.record(k * 0.01, pose, [0.0, 0.0])
    return history


def test_slam_pose_seeds_the_cog_state_ahead_of_the_rear_axle():
    updater = make_updater(history_at_rest([0.0, 0.0, 0.0]))
    ekf = _EkfWithUpdate(np.zeros(7), np.eye(7) * 10.0)
    updater.seed(ekf, [5.0, 6.0, np.pi / 2])
    np.testing.assert_allclose(ekf.x[:3], [5.0, 6.0 + updater.model.params.l_r, np.pi / 2], atol=1e-12)
    np.testing.assert_allclose(ekf.P[:3, :3], updater.R)
    assert np.all(ekf.P[:3, 3:] == 0.0) and ekf.P[3, 3] == 10.0


def test_slam_pose_pulls_the_estimate_towards_it_and_shrinks_the_uncertainty():
    updater = make_updater(history_at_rest([0.0, 0.0, 0.0]))
    ekf = _EkfWithUpdate(np.zeros(7), np.diag([0.04, 0.04, 0.02, 1.0, 1.0, 1.0, 1.0]))
    updater.model.Hx_pose(ekf.x)
    ekf.x[0] = updater.model.params.l_r  # base_link at the origin
    outcome = updater.update(ekf, [0.1, 0.0, 0.0], 0.5)
    assert outcome == 'updated'
    assert 0.0 < ekf.x[0] - updater.model.params.l_r < 0.1  # moved towards the pose, not onto it
    assert ekf.P[0, 0] < 0.04


def test_slam_pose_is_compared_with_the_estimate_at_its_stamp_not_the_present():
    # The car moved from x=0 to x=1 over the second; a pose at the start must not drag the present back.
    history = PredictionHistory(accel_lag=0.05)
    for k in range(101):
        history.record(k * 0.01, [k * 0.01, 0.0, 0.0], [0.0, 0.0])
    updater = make_updater(history)
    ekf = _EkfWithUpdate(np.array([1.0 + updater.model.params.l_r, 0.0, 0.0, 1.0, 0, 0, 0]), np.eye(7) * 0.01)
    before = ekf.x.copy()
    assert updater.update(ekf, [0.3, 0.0, 0.0], 0.3) == 'updated'  # agrees with where the car was at 0.3 s
    np.testing.assert_allclose(ekf.x, before, atol=1e-9)


def test_slam_pose_that_disagrees_is_gated_then_reseeds_after_a_streak():
    updater = make_updater(history_at_rest([0.0, 0.0, 0.0]), reseed_after=3)
    ekf = _EkfWithUpdate(np.zeros(7), np.eye(7) * 1e-3)
    far = [5.0, 0.0, 0.0]
    assert updater.update(ekf, far, 0.5) == 'gated'
    assert updater.update(ekf, far, 0.5) == 'gated'
    assert ekf.x[0] == 0.0  # untouched while gated
    assert updater.update(ekf, far, 0.5) == 'reseeded'
    assert ekf.x[0] == pytest.approx(5.0 + updater.model.params.l_r)
    assert updater.rejected == 3 and updater.reseeds == 1


def test_slam_pose_without_history_is_skipped():
    updater = make_updater(PredictionHistory(accel_lag=0.05))
    ekf = _EkfWithUpdate(np.zeros(7), np.eye(7))
    assert updater.update(ekf, [0.0, 0.0, 0.0], 0.5) == 'skipped'


def test_slam_yaw_residual_wraps():
    updater = make_updater(history_at_rest([0.0, 0.0, np.pi - 0.01]))
    ekf = _EkfWithUpdate(np.array([updater.model.params.l_r * np.cos(np.pi - 0.01), updater.model.params.l_r * np.sin(np.pi - 0.01),
                                   np.pi - 0.01, 0, 0, 0, 0]), np.eye(7) * 0.01)
    assert updater.update(ekf, [0.0, 0.0, -np.pi + 0.01], 0.5) == 'updated'  # 0.02 rad apart, not 6.26


def test_a_slam_jump_reaches_the_estimate_gradually():
    """A map->odom style step in the SLAM pose is absorbed over many ticks, not passed through as a jump."""
    model = make_model()
    dt, speed = 0.01, 2.0
    history = PredictionHistory(accel_lag=0.05)
    updater = SlamPoseUpdater(model, np.diag([0.01, 0.01, 0.005]), history, 0.0, 16.0, 15)
    ekf = _EkfWithUpdate(np.array([0.0, 0.0, 0.0, speed, 0.0, 0.0, 0.0]), np.eye(7) * 0.01)
    ekf.P[3:, 3:] = np.eye(4) * 1e-6  # the speed is known; only the position is in question
    u = np.array([speed, 0.0])
    xs, outcomes = [], []
    for k in range(1, 401):
        t = k * dt
        model.predict(ekf, dt, u)
        truth_x = speed * t
        jump = 0.15 if t >= 2.0 else 0.0  # the localization steps by 15 cm at t = 2 s
        outcomes.append(updater.update(ekf, [truth_x + jump, 0.0, 0.0], t))
        history.record(t, model.Hx_pose(ekf.x), [0.0, 0.0])
        xs.append(ekf.x[0] - model.params.l_r)
    xs = np.array(xs)
    step = np.diff(xs)[199:]  # ticks from the jump on
    assert step.max() - speed * dt < 0.02, "the 15 cm step is spread over many ticks (about 1 cm in the first)"
    assert abs(xs[-1] - (speed * 4.0 + 0.15)) < 0.03, "it ends up at the new pose"
    assert 'gated' not in outcomes and 'reseeded' not in outcomes


def test_history_rebase_moves_the_stored_trajectory_rigidly():
    history = PredictionHistory(accel_lag=0.05)
    for k in range(11):  # 1 m/s along x, then the filter is re-seeded 5 m away and turned a quarter turn
        history.record(k * 0.1, [k * 0.1, 0.0, 0.0], [0.0, 0.0])
    history.rebase([1.0, 0.0, 0.0], [10.0, 20.0, np.pi / 2])
    np.testing.assert_allclose(history.pose_at(1.0), [10.0, 20.0, np.pi / 2], atol=1e-12)
    np.testing.assert_allclose(history.pose_at(0.5), [10.0, 19.5, np.pi / 2], atol=1e-12)  # half a metre earlier


def test_after_a_reseed_the_next_poses_are_compared_with_the_new_trajectory():
    model = make_model()
    dt, speed = 0.01, 2.0
    history = PredictionHistory(accel_lag=0.05)
    updater = SlamPoseUpdater(model, np.diag([0.01, 0.01, 0.005]), history, 0.03, 16.0, 5)
    # The filter believes the car is 5 m ahead of where SLAM says it is.
    ekf = _EkfWithUpdate(np.array([5.0 + model.params.l_r, 0.0, 0.0, speed, 0.0, 0.0, 0.0]), np.eye(7) * 0.01)
    ekf.P[3:, 3:] = np.eye(4) * 1e-6
    u = np.array([speed, 0.0])
    outcomes = []
    for k in range(1, 301):
        t = k * dt
        model.predict(ekf, dt, u)
        outcomes.append(updater.update(ekf, [speed * (t - 0.03), 0.0, 0.0], t))  # content 30 ms older than its stamp
        history.record(t, model.Hx_pose(ekf.x), [0.0, 0.0])
    assert updater.reseeds == 1, outcomes[:40]
    after = outcomes[outcomes.index('reseeded') + 1:]
    # The poses right after the jump look up the filter's history from before it; rebased, they are not gated.
    assert after and set(after) == {'updated'}, after[:10]
    assert abs(ekf.x[0] - model.params.l_r - speed * 3.0) < 0.1
