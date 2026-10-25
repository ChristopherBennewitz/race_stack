"""Tests for the C++ progress MPC through its Python bindings; the plant is the MPC's own model."""
import math

import numpy as np
import pytest

progress_mpc = pytest.importorskip("progress_mpc")


def stadium(straight=6.0, radius=2.5, spacing=0.1):
    """Counter-clockwise stadium centerline starting at the bottom straight's start, with its length."""
    pieces = []
    n_straight = int(straight / spacing)
    n_arc = int(math.pi * radius / spacing)
    for i in range(n_straight):
        pieces.append((i * spacing, 0.0))
    for i in range(n_arc):
        a = -math.pi / 2 + math.pi * i / n_arc
        pieces.append((straight + radius * math.cos(a), radius + radius * math.sin(a)))
    for i in range(n_straight):
        pieces.append((straight - i * spacing, 2 * radius))
    for i in range(n_arc):
        a = math.pi / 2 + math.pi * i / n_arc
        pieces.append((radius * math.cos(a), radius + radius * math.sin(a)))
    return np.array(pieces)


@pytest.fixture
def mpc():
    track = stadium()
    controller = progress_mpc.ProgressMpc()
    controller.set_track(track, np.full(len(track), 1.0), np.full(len(track), 1.0))
    return controller


def initial_state(mpc, speed=1.0):
    return np.concatenate([[1.0, 0.0, 0.0, speed, 0.0, 0.0, 0.0], np.zeros(mpc.history_length)])


def drive(mpc, steps, speed=1.0):
    """Closed loop on the model. Returns the commands and the arc length / lateral offset of the car."""
    z = initial_state(mpc, speed)
    mpc.reset()
    commands, frenet = [], []
    for _ in range(steps):
        steer, speed_cmd = mpc.step(z[:6], z[6])
        z = mpc.model_step(z, steer, speed_cmd)
        mpc.report_applied(steer, speed_cmd)
        commands.append((steer, speed_cmd))
        frenet.append(mpc.plan_frenet[0])
    return np.array(commands), np.array(frenet), z


def test_step_needs_a_track():
    with pytest.raises(RuntimeError, match="set_track"):
        progress_mpc.ProgressMpc().step(np.zeros(6))


@pytest.mark.parametrize("n", [3])
def test_a_degenerate_track_is_rejected(n):
    with pytest.raises(ValueError):
        progress_mpc.ProgressMpc().set_track(np.zeros((n, 2)), np.ones(n), np.ones(n))


def test_widths_must_match_the_centerline():
    with pytest.raises(ValueError):
        progress_mpc.ProgressMpc().set_track(stadium(), np.ones(5), np.ones(5))


def test_track_length_is_the_closed_perimeter(mpc):
    track = stadium()
    expected = np.linalg.norm(np.roll(track, -1, axis=0) - track, axis=1).sum()
    assert mpc.track_length == pytest.approx(expected)


def test_drives_forward_inside_the_track_and_the_command_limits(mpc):
    commands, frenet, _ = drive(mpc, 160)
    assert mpc.stats["failures"] == 0
    assert mpc.stats["status"] == "Solve_Succeeded"
    s = np.unwrap(frenet[:, 0], period=mpc.track_length)
    assert s[-1] - s[0] > 3.0, "the car should make progress"
    assert np.abs(frenet[:, 1]).max() < 1.0, "the car should stay between the walls"
    assert np.abs(commands[:, 0]).max() <= mpc.steer_max + 1e-9
    assert commands[:, 1].min() >= -1e-9 and commands[:, 1].max() <= mpc.speed_max + 1e-9


def test_commands_respect_the_per_period_rate_limits(mpc):
    commands, _, _ = drive(mpc, 80)
    steps = np.abs(np.diff(commands, axis=0))
    assert steps[:, 0].max() <= 0.06 + 1e-6  # 3 rad/s over the manager's two 10 ms ticks per 25 ms period
    assert steps[:, 1].max() <= 0.10 + 1e-6  # 5 m/s^2 likewise


def test_a_plan_spans_the_horizon(mpc):
    drive(mpc, 5)
    assert mpc.plan_world.shape == (mpc.horizon + 1, 2)
    assert mpc.plan_frenet.shape == (mpc.horizon + 1, 2)
    assert np.all(np.diff(np.unwrap(mpc.plan_frenet[:, 0], period=mpc.track_length)) > 0), "the plan moves forward"


def test_internal_steering_estimate_tracks_the_models_steering_state(mpc):
    z = initial_state(mpc)
    mpc.reset()
    worst = 0.0
    for _ in range(120):
        steer, speed = mpc.step(z[:6], z[6])
        z = mpc.model_step(z, steer, speed)
        mpc.report_applied(steer, speed)
        worst = max(worst, abs(mpc.steer_estimate - z[6]))
    # The estimate is the exact first-order lag; the model integrates it with RK4.
    assert worst < 5e-4


def test_reset_restarts_from_scratch(mpc):
    first, _, _ = drive(mpc, 30)
    second, _, _ = drive(mpc, 30)
    assert np.allclose(first, second)


def test_the_track_clearance_is_exposed(mpc):
    # collision radius + margin of the exported solver: positive, and well under a metre on a 30 cm wide car
    assert 0.2 < mpc.clearance_margin < 1.0
