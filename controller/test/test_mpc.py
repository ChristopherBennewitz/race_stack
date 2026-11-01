"""Tests for the ROS-free parts of the MPC controller."""

import logging

import numpy as np
import pytest

from controller.mpc.mpc_controller import MPC_Controller, base_link_to_cog, oriented_centerline, track_envelope, extrapolate_state


def circle(n=200, r=5.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def test_counter_clockwise_centerline_is_kept_for_ccw():
    xy, left, right = oriented_centerline(circle(), np.full(200, 1.0), np.full(200, 2.0), 'ccw')
    assert np.allclose(xy, circle())
    assert (left == 1.0).all() and (right == 2.0).all()


def test_clockwise_driving_reverses_the_centerline_and_swaps_the_widths():
    xy, left, right = oriented_centerline(circle(), np.full(200, 1.0), np.full(200, 2.0), 'cw')
    assert np.allclose(xy, circle()[::-1])
    assert (left == 2.0).all() and (right == 1.0).all()


def test_unknown_direction_is_rejected():
    with pytest.raises(ValueError):
        oriented_centerline(circle(), np.ones(200), np.ones(200), 'up')


def test_base_link_to_cog_moves_ahead_along_the_heading():
    x, y, vy = base_link_to_cog(1.0, 2.0, np.pi / 2, 0.1, 2.0, 0.15)
    assert (x, y) == pytest.approx((1.0, 2.15))
    assert vy == pytest.approx(0.1 + 0.15 * 2.0)


def test_controller_drives_a_ccw_circle():
    pytest.importorskip("progress_mpc")
    controller = MPC_Controller('ccw', logging.getLogger('test'))
    controller.set_centerline(circle(r=8.0), np.full(200, 1.0), np.full(200, 1.0))
    assert controller.has_track
    # On the circle at angle 0, heading along the ccw tangent (+y), at rest.
    speed, steer = controller.main_loop(8.0, 0.0, np.pi / 2, 0.5, 0.0, 0.0)
    assert speed > 0.5, "it should ask for more speed"
    assert controller.last_cmd == (steer, speed)
    d = controller.diagnostics()  # real solver output: the keys the stub imitates must exist and make sense
    assert d['cycles'] == 1 and 0.0 < d['solve_ms_max'] < 100.0 and d['iterations_max'] >= 1
    assert abs(d['lateral_error']) < 0.5 and d['failures'] == 0
    assert controller.verdict(d)[0] in ('ok', 'warn')
    controller.reset()
    assert controller.last_cmd is None


# --- diagnostics (a stub solver, so these run without acados) ---------------------------------------------------------

class StubMpc:
    """Just enough of ProgressMpc for the controller's bookkeeping: plans to be 5 cm further along x next period."""
    control_dt = 0.025
    horizon = 40
    speed_max = 10.0
    steer_max = 0.4
    cog_offset = 0.15
    has_track = True
    steer_estimate = 0.01

    def __init__(self, spacing=0.1):
        self.stats = {'solve_seconds': 0.004, 'solver_only_seconds': 0.003, 'iterations': 1, 'status': 'Solve_Succeeded',
                      'slack_max': 0.0, 'failures': 0}
        self.plan_world = np.zeros((41, 2))
        self.plan_frenet = np.zeros((41, 2))
        self.resets = 0

    def step(self, state):
        self.plan_world = np.array([[state[0] + 0.05 * k, state[1]] for k in range(41)])
        self.plan_frenet[0, 1] = 0.1
        return 0.02, 3.0

    def reset(self):
        self.resets += 1


@pytest.fixture
def stubbed(monkeypatch):
    import sys
    import types

    module = types.ModuleType('progress_mpc')
    module.ProgressMpc = StubMpc
    monkeypatch.setitem(sys.modules, 'progress_mpc', module)
    times = [0.0]
    controller = MPC_Controller('ccw', logging.getLogger('test'), clock=lambda: times[0])
    return controller, times


def test_the_prediction_error_is_the_distance_from_where_the_plan_put_the_car(stubbed):
    controller, times = stubbed
    controller.main_loop(0.0, 0.0, 0.0, 2.0, 0.0, 0.0)
    times[0] += 0.025
    controller.main_loop(0.06, 0.0, 0.0, 2.0, 0.0, 0.0)  # the plan said 0.05
    d = controller.diagnostics()
    assert d['prediction_error_mean'] == pytest.approx(0.01)
    assert d['cycles'] == 2
    assert d['period_mean_ms'] == pytest.approx(25.0)
    assert d['lateral_error'] == pytest.approx(0.1)


def test_a_late_cycle_is_not_compared_with_the_old_prediction(stubbed):
    controller, times = stubbed
    controller.main_loop(0.0, 0.0, 0.0, 2.0, 0.0, 0.0)
    times[0] += 0.3  # the control loop stalled
    controller.main_loop(0.6, 0.0, 0.0, 2.0, 0.0, 0.0)
    assert np.isnan(controller.diagnostics()['prediction_error_mean'])


def test_the_window_restarts_after_each_report(stubbed):
    controller, times = stubbed
    controller.mpc.stats['solve_seconds'] = 0.02
    controller.main_loop(0.0, 0.0, 0.0, 2.0, 0.0, 0.0)
    assert controller.diagnostics()['solve_ms_max'] == pytest.approx(20.0)
    again = controller.diagnostics()
    assert again['cycles'] == 0 and np.isnan(again['solve_ms_max'])


def test_verdicts_name_the_first_thing_wrong(stubbed):
    controller, times = stubbed
    assert controller.verdict(controller.diagnostics())[0] == 'warn'  # nothing has run yet
    controller.main_loop(0.0, 0.0, 0.0, 2.0, 0.0, 0.0)
    assert controller.verdict(controller.diagnostics()) == ('ok', 'solving')

    controller.mpc.stats['solve_seconds'] = 0.05  # over the 25 ms period
    controller.main_loop(0.0, 0.0, 0.0, 2.0, 0.0, 0.0)
    level, message = controller.verdict(controller.diagnostics())
    assert level == 'warn' and 'solve took' in message

    controller.mpc.stats['solve_seconds'] = 0.004
    controller.mpc.stats['failures'] = 1
    controller.mpc.stats['status'] = 'acados_status_4'
    controller.main_loop(0.0, 0.0, 0.0, 2.0, 0.0, 0.0)
    level, message = controller.verdict(controller.diagnostics())
    assert level == 'error' and 'acados_status_4' in message
    controller.main_loop(0.0, 0.0, 0.0, 2.0, 0.0, 0.0)  # the same failure is not reported again
    assert controller.verdict(controller.diagnostics())[0] == 'ok'


def test_stale_odometry_and_track_violations_are_flagged(stubbed):
    controller, times = stubbed
    controller.note_odom_age(0.09)
    controller.main_loop(0.0, 0.0, 0.0, 2.0, 0.0, 0.0)
    assert 'odometry' in controller.verdict(controller.diagnostics())[1]
    controller.mpc.stats['slack_max'] = 0.12
    controller.main_loop(0.0, 0.0, 0.0, 2.0, 0.0, 0.0)
    assert 'track limit' in controller.verdict(controller.diagnostics())[1]


def test_an_estimate_jump_drops_the_plan_and_is_counted(stubbed):
    controller, times = stubbed
    controller.estimate_jumped()  # no plan yet: nothing to drop
    assert controller.diagnostics()['jump_resets'] == 0
    controller.main_loop(0.0, 0.0, 0.0, 2.0, 0.0, 0.0)
    controller.estimate_jumped()
    d = controller.diagnostics()
    assert controller.last_cmd is None and controller.mpc.resets == 1
    assert d['jump_resets'] == 1 and d['resets'] == 1
    times[0] += 0.025
    controller.main_loop(5.0, 0.0, 0.0, 2.0, 0.0, 0.0)  # the old prediction is forgotten: no spurious error
    assert np.isnan(controller.diagnostics()['prediction_error_mean'])


def test_the_envelope_is_the_track_narrowed_by_the_clearance():
    # A counter-clockwise circle of radius 8: left of the driving direction is towards the centre.
    left, right = track_envelope(circle(r=8.0), np.full(200, 1.0), np.full(200, 2.0), 0.5)
    assert np.allclose(np.linalg.norm(left, axis=1), 8.0 - (1.0 - 0.5), atol=0.01)
    assert np.allclose(np.linalg.norm(right, axis=1), 8.0 + (2.0 - 0.5), atol=0.01)


def test_the_envelope_follows_the_driving_direction():
    # Clockwise driving: the centerline is reversed and the widths swapped, so "left" is now towards the outside.
    xy, d_left, d_right = oriented_centerline(circle(r=8.0), np.full(200, 1.0), np.full(200, 2.0), 'cw')
    left, right = track_envelope(xy, d_left, d_right, 0.5)
    assert np.allclose(np.linalg.norm(left, axis=1), 8.0 + (2.0 - 0.5), atol=0.01)
    assert np.allclose(np.linalg.norm(right, axis=1), 8.0 - (1.0 - 0.5), atol=0.01)


def test_the_controller_reports_the_envelope_once_it_has_a_track(stubbed):
    controller, _ = stubbed
    controller.mpc.set_track = lambda *args: None
    controller.mpc.clearance_margin = 0.484
    assert controller.envelope() is None
    controller.set_centerline(circle(r=8.0), np.full(200, 1.0), np.full(200, 1.0))
    left, right = controller.envelope()
    assert left.shape == right.shape == (200, 2)


def test_extrapolating_a_straight_run_moves_the_pose_along_the_heading():
    x, y, yaw = extrapolate_state(1.0, 2.0, np.pi / 2, 3.0, 0.0, 0.0, 0.1)
    assert (x, y, yaw) == pytest.approx((1.0, 2.3, np.pi / 2))


def test_extrapolating_with_a_sideways_velocity_uses_the_body_frame():
    # Heading along +x, moving left (+y in the body frame) at 1 m/s.
    x, y, yaw = extrapolate_state(0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.5)
    assert (x, y, yaw) == pytest.approx((1.0, 0.5, 0.0))


def test_extrapolating_a_turn_follows_the_circle():
    # 2 m/s at 1 rad/s is a circle of radius 2; a quarter turn takes pi/2 s, from (0, 0) heading +x it ends at (2, 2).
    dt = np.pi / 2 / 1000
    x, y, yaw = 0.0, 0.0, 0.0
    for _ in range(1000):
        x, y, yaw = extrapolate_state(x, y, yaw, 2.0, 0.0, 1.0, dt)
    assert (x, y, yaw) == pytest.approx((2.0, 2.0, np.pi / 2), abs=1e-4)
    # one big step uses the heading at the middle of the interval (45 degrees): speed * cos(45 deg) * dt along x
    x, y, yaw = extrapolate_state(0.0, 0.0, 0.0, 2.0, 0.0, 1.0, np.pi / 2)
    assert (x, y, yaw) == pytest.approx((2.0 * np.cos(np.pi / 4) * np.pi / 2, 2.0 * np.sin(np.pi / 4) * np.pi / 2, np.pi / 2))


def test_zero_time_leaves_the_pose_alone():
    assert extrapolate_state(1.0, 2.0, 0.3, 4.0, 0.5, 1.0, 0.0) == pytest.approx((1.0, 2.0, 0.3))


def test_the_compute_lag_is_the_fallback_until_a_solve_has_been_timed(stubbed):
    controller, _ = stubbed
    assert controller.compute_lag(0.0012, 0.0075) == 0.0075


def test_the_compute_lag_follows_the_solve_time_and_stays_within_one_period(stubbed):
    controller, times = stubbed
    controller.main_loop(0.0, 0.0, 0.0, 2.0, 0.0, 0.0)  # the stub solves in 4 ms
    assert controller.compute_lag(0.0012, 0.0075) == pytest.approx(0.0052)
    controller.mpc.stats['solve_seconds'] = 0.020  # the CPU gets busy
    lags = []
    for _ in range(12):
        times[0] += 0.025
        controller.main_loop(0.0, 0.0, 0.0, 2.0, 0.0, 0.0)
        lags.append(controller.compute_lag(0.0012, 0.0075))
    assert lags == sorted(lags) and lags[0] < 0.01 < lags[-1]  # smoothed: it follows, not jumps
    # 12 updates of weight 0.2 move the smoothed 4 ms to 20 - 16 * 0.8**12 = 18.9 ms
    assert lags[-1] == pytest.approx(0.0012 + 0.0189, abs=0.0005)
    controller.mpc.stats['solve_seconds'] = 0.200  # a solve that overran by far
    for _ in range(40):
        times[0] += 0.025
        controller.main_loop(0.0, 0.0, 0.0, 2.0, 0.0, 0.0)
    assert controller.compute_lag(0.0012, 0.0075) == pytest.approx(controller.mpc.control_dt)


def test_a_reset_keeps_the_timed_solves(stubbed):
    controller, _ = stubbed
    controller.main_loop(0.0, 0.0, 0.0, 2.0, 0.0, 0.0)
    controller.reset()
    assert controller.compute_lag(0.0012, 0.0075) == pytest.approx(0.0052)


# --- the multi-point turn ----------------------------------------------------------------------------------------------

import math

from controller.mpc.mpc_controller import track_geometry


def test_the_track_geometry_measures_the_distance_to_the_nearer_wall_and_the_driving_direction():
    wall_distance, heading = track_geometry(circle(r=20.0), np.full(200, 0.7), np.full(200, 0.9))
    on_centerline, near_left, near_right, outside = (20.0, 0.0), (19.5, 0.0), (20.8, 0.0), (21.5, 0.0)
    distances = wall_distance(np.array([on_centerline, near_left, near_right, outside]))
    # a ccw circle: the left wall is the inner one, 0.7 m from the centerline, the right one 0.9 m
    assert distances == pytest.approx([0.7, 0.2, 0.1, -0.6], abs=0.02)
    assert heading(20.0, 0.0) == pytest.approx(np.pi / 2, abs=0.03)  # driving in +y at angle 0
    assert heading(0.0, 20.0) == pytest.approx(np.pi, abs=0.03)


class IdealCar:
    """A kinematic bicycle that does what it is told at once, standing on the circle track, for the maneuver's sake."""

    def __init__(self, x, y, yaw, wheelbase=0.258, rear_to_cog=0.15, steer_gain=1.16):
        self.l, self.r, self.gain = wheelbase, rear_to_cog, steer_gain
        self.rear = np.array([x, y]) - rear_to_cog * np.array([math.cos(yaw), math.sin(yaw)])
        self.yaw, self.speed = yaw, 0.0

    @property
    def cog(self):
        return (*(self.rear + self.r * np.array([math.cos(self.yaw), math.sin(self.yaw)])), self.yaw)

    def drive(self, speed, steer, dt=0.025):
        self.speed = speed
        self.rear = self.rear + speed * dt * np.array([math.cos(self.yaw), math.sin(self.yaw)])
        self.yaw += speed * math.tan(self.gain * steer) / self.l * dt


def _drive_until_handed_back(controller, times, car, seconds, requested_stub_speed=0.0):
    for _ in range(int(seconds / 0.025)):
        x, y, yaw = car.cog
        speed, steer = controller.main_loop(x, y, yaw, car.speed, 0.0, 0.0)
        car.drive(speed, steer)
        times[0] += 0.025
        if controller.recovery is not None and controller.recovery.maneuvers and not controller.recovery.active:
            return True
    return False


@pytest.fixture
def stubbed_with_recovery(stubbed):
    controller, times = stubbed
    controller.mpc.clearance_margin = 0.3
    controller.mpc.set_track = lambda *args: None
    controller.mpc.step = lambda state: (0.0, 0.0)  # a planner that is stuck: asks for no movement
    controller.set_centerline(circle(r=20.0), np.full(200, 0.9), np.full(200, 0.9))
    return controller, times


def test_without_the_recovery_a_stuck_planner_leaves_the_car_where_it_is(stubbed_with_recovery):
    controller, times = stubbed_with_recovery
    car = IdealCar(20.0, 0.0, -np.pi / 2)  # on the centerline of the ccw circle, facing against the driving direction
    assert controller.recovery is None
    assert not _drive_until_handed_back(controller, times, car, 10.0)
    assert car.cog[:2] == pytest.approx((20.0, 0.0), abs=1e-9)
    assert controller.diagnostics()['recovering'] == 0.0


def test_the_recovery_turns_a_car_that_the_planner_cannot_move_around(stubbed_with_recovery):
    controller, times = stubbed_with_recovery
    controller.configure_recovery(wheelbase=0.258, steer_gain=1.16)
    car = IdealCar(20.0, 0.0, -np.pi / 2)
    assert _drive_until_handed_back(controller, times, car, 30.0)
    x, y, yaw = car.cog
    tangent = np.pi / 2  # ccw circle at angle 0: driving in +y
    assert abs((yaw - tangent + np.pi) % (2 * np.pi) - np.pi) < 0.7
    d = controller.diagnostics()
    assert d['recoveries'] == 1.0 and d['recovery_failures'] == 0.0 and d['recovering'] == 0.0
    # the planner has the car again and starts from a clean plan
    assert controller.last_cmd is not None


def test_a_lost_odometry_cancels_the_maneuver(stubbed_with_recovery):
    controller, times = stubbed_with_recovery
    controller.configure_recovery(wheelbase=0.258, steer_gain=1.16)
    car = IdealCar(20.0, 0.0, -np.pi / 2)
    for _ in range(int(2.0 / 0.025)):  # long enough for it to take over
        x, y, yaw = car.cog
        speed, steer = controller.main_loop(x, y, yaw, car.speed, 0.0, 0.0)
        car.drive(speed, steer)
        times[0] += 0.025
    assert controller.recovery.active
    controller.reset()  # what the manager does when the odometry is stale
    assert not controller.recovery.active
    assert controller.diagnostics()['recovering'] == 0.0


def test_the_verdict_says_when_the_car_is_being_turned_around(stubbed_with_recovery):
    controller, times = stubbed_with_recovery
    controller.configure_recovery(wheelbase=0.258, steer_gain=1.16)
    car = IdealCar(20.0, 0.0, -np.pi / 2)
    for _ in range(int(2.0 / 0.025)):
        x, y, yaw = car.cog
        speed, steer = controller.main_loop(x, y, yaw, car.speed, 0.0, 0.0)
        car.drive(speed, steer)
        times[0] += 0.025
    level, message = controller.verdict(controller.diagnostics())
    assert level == 'warn' and 'turning the car around' in message
