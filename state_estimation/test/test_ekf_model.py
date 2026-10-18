import numpy as np

from state_estimation.ekf_node.models import CommandDelay, SimulatorSingleTrackModel, VehicleModelParams
from state_estimation.ekf_node.prediction_history import PredictionHistory
from state_estimation.ekf_node.speed_window import TrailingSpeedWindow


def make_model(imu_x=0.07):
    params = VehicleModelParams(
        m=3.718, I_z=0.0561, l_f=0.17334, l_r=0.14766, h_cg=0.074,
        Bf=12.0306, Cf=1.71178, Df=0.402674, Ef=0.0,
        Br=5.06904, Cr=1.64946, Dr=0.572528, Er=0.0,
        steer_gain=1.6746, steer_bias=0.000807634, steer_delay=0.0871629, steer_latency=0.0338089,
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


def test_history_mean_speed_over_a_past_window():
    history = PredictionHistory(accel_lag=0.05)
    for k in range(101):  # speed ramps 1 m/s per second, recorded at 100 Hz
        history.record(k * 0.01, k * 0.01, [0.0, 0.0])
    assert abs(history.mean_vx(0.3, 0.5) - 0.4) < 1e-9
    assert abs(history.mean_vx(0.305, 0.495) - 0.4) < 1e-9  # between recorded ticks
    assert history.mean_vx(-0.1, 0.1) is None  # older than the history


def test_history_lags_the_predicted_acceleration_like_the_accelerometer():
    history = PredictionHistory(accel_lag=0.05)
    history.record(0.0, 0.0, [0.0, 0.0])
    for k in range(1, 6):  # a step to 1 m/s^2, followed for exactly one time constant
        history.record(k * 0.01, 0.0, [1.0, -2.0])
    np.testing.assert_allclose(history.accel, [1.0 - np.exp(-1.0), -2.0 * (1.0 - np.exp(-1.0))], atol=1e-9)


def test_trailing_speed_window_measures_mean_forward_speed():
    tracker = TrailingSpeedWindow(0.2)
    samples = [tracker.push(k * 0.05, 2.0 * k * 0.05, 0.0, 0.0) for k in range(9)]
    completed = [s for s in samples if s is not None]
    assert len(completed) == 2  # windows [0, 0.2] and [0.2, 0.4] share only the pose at 0.2
    assert all(abs(speed - 2.0) < 1e-9 and abs(duration - 0.2) < 1e-9 for speed, duration in completed)


def test_trailing_speed_window_ignores_repeated_stamps_and_drops_stretched_windows():
    tracker = TrailingSpeedWindow(0.2)
    assert tracker.push(0.0, 0.0, 0.0, 0.0) is None
    assert tracker.push(0.0, 5.0, 0.0, 0.0) is None  # republished stamp
    assert tracker.push(1.0, 2.0, 0.0, 0.0) is None  # a 1 s gap is not a 0.2 s window
    speed, duration = tracker.push(1.25, 2.5, 0.0, 0.0)
    assert abs(speed - 2.0) < 1e-9 and abs(duration - 0.25) < 1e-9


def test_trailing_speed_window_projects_on_the_mean_heading():
    tracker = TrailingSpeedWindow(0.2)
    tracker.push(0.0, 0.0, 0.0, np.pi / 2)
    speed, _ = tracker.push(0.2, 0.0, 0.5, np.pi / 2)  # driving along +y with the car facing +y
    assert abs(speed - 2.5) < 1e-9
