import numpy as np

from state_estimation.ekf_node.models import SimulatorSingleTrackModel, VehicleModelParams


def make_model():
    params = VehicleModelParams(
        m=3.718, I_z=0.0561, l_f=0.17334, l_r=0.14766, h_cg=0.074,
        Bf=12.0306, Cf=1.71178, Df=0.402674, Ef=0.0,
        Br=5.06904, Cr=1.64946, Dr=0.572528, Er=0.0,
        steer_gain=1.50936, steer_bias=0.00332491, steer_latency=0.111489,
        drive_gain=3.59866, brake_gain=5.00361,
    )
    return SimulatorSingleTrackModel(params, np.eye(8) * 0.01)


class _FakeEkf:
    """Minimal stand-in for filterpy's ExtendedKalmanFilter, exposing x/P."""

    def __init__(self, x, P):
        self.x = x
        self.P = P


def test_stationary_model_has_finite_transition_and_jacobian():
    model = make_model()
    state = np.zeros(8)
    control = np.array([1.0, 0.2])  # [speed_command, steer_command]
    transitioned = np.array(model.f(state, control, 0.01)).flatten()
    jacobian = np.array(model.F(state, control, 0.01))
    assert np.isfinite(transitioned).all()
    assert np.isfinite(jacobian).all()
    assert jacobian.shape == (8, 8)


def test_imu_measurement_model_is_finite():
    model = make_model()
    state = np.array([0.0, 0.0, 0.0, 2.0, 0.1, 0.05, 0.1, 0.1])
    control = np.array([0.5, 0.15])
    measurement = model.Hx_imu(state, control)
    jacobian = model.HJacobian_imu(state, control)
    assert np.isfinite(measurement).all()
    assert measurement.shape == (4,)
    assert jacobian.shape == (4, 8)
    assert np.isfinite(jacobian).all()


def test_predict_relaxes_towards_kinematic_yaw_rate_at_low_speed():
    # At low speed the kinematic/dynamic blend weight is close to zero, so the
    # yaw rate should settle near the pure kinematic bicycle relation.
    model = make_model()
    ekf = _FakeEkf(np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]), model.P_0.copy())
    control = np.array([0.1, 0.1])  # hold a small steering command and the current speed

    for _ in range(200):
        model.predict(ekf, dt=0.01, u=control)

    wheelbase = 0.17334 + 0.14766
    steer = 1.50936 * control[1] + 0.00332491  # steady-state road-wheel angle
    expected_yaw_rate = ekf.x[3] * np.tan(steer) / wheelbase
    assert abs(ekf.x[6] - steer) < 1e-3
    assert np.isfinite(ekf.x).all()
    assert abs(ekf.x[5] - expected_yaw_rate) < 0.01


def test_speed_loop_accelerates_towards_commanded_speed():
    model = make_model()
    ekf = _FakeEkf(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), model.P_0.copy())
    for _ in range(100):
        model.predict(ekf, dt=0.01, u=np.array([3.0, 0.0]))
    assert 1.0 < ekf.x[3] < 3.0
