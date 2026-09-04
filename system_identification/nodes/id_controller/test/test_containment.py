import math

import numpy as np

from id_controller import containment


def test_straight_reference_integrates_forward_and_reverse():
    commands = np.array([[0.0, 1.0]] * 11)
    forward = containment.integrate_reference(
        commands, commands[:, 0], 0.1, 0.32)
    reverse_commands = np.array([[0.0, -1.0]] * 11)
    reverse = containment.integrate_reference(
        reverse_commands, reverse_commands[:, 0], 0.1, 0.32)
    assert np.allclose(forward[-1], [1.0, 0.0, 0.0])
    assert np.allclose(reverse[-1], [-1.0, 0.0, 0.0])


def test_transform_path_handles_rotation():
    path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    transformed = containment.transform_path(path, 2.0, 3.0, math.pi / 2.0)
    assert np.allclose(transformed[:, :2], [[2.0, 3.0], [2.0, 4.0]])
    assert np.allclose(transformed[:, 2], math.pi / 2.0)


def test_steering_correction_returns_toward_path_in_both_directions():
    forward = containment.steering_correction(0.2, 0.0, 1.0, 0.35, 0.35, 0.6)
    reverse = containment.steering_correction(0.2, 0.0, -1.0, 0.35, 0.35, 0.6)
    assert forward < 0.0
    assert reverse < 0.0


def test_reverse_heading_correction_changes_sign():
    forward = containment.steering_correction(0.0, 0.2, 1.0, 0.35, 0.35, 0.6)
    reverse = containment.steering_correction(0.0, 0.2, -1.0, 0.35, 0.35, 0.6)
    assert forward > 0.0
    assert reverse < 0.0


def test_distance_field_treats_unknown_obstacles_and_boundary_as_unsafe():
    grid = np.zeros((7, 7), dtype=int)
    grid[3, 3] = 100
    grid[2, 2] = -1
    field = containment.DistanceField.from_grid(
        grid.ravel(), 7, 7, 0.1, 1.0, 2.0, 0.0)
    assert field.clearance(1.35, 2.35) == 0.0
    assert field.clearance(1.25, 2.25) == 0.0
    assert math.isclose(field.clearance(1.45, 2.35), 0.1)
    assert field.clearance(50.0, 50.0) == 0.0


def test_predicted_clearance_detects_approaching_wall():
    grid = np.zeros((20, 20), dtype=int)
    grid[:, 15] = 100
    field = containment.DistanceField.from_grid(
        grid.ravel(), 20, 20, 0.1, 0.0, 0.0, 0.0)
    here = field.clearance(0.5, 1.0)
    ahead = containment.predicted_clearance(
        field, 0.5, 1.0, 0.0, 1.0, 0.0, 0.8, 0.32)
    assert ahead < here


def test_twist_prediction_includes_lateral_slip():
    grid = np.zeros((20, 40), dtype=int)
    grid[15, :] = 100
    field = containment.DistanceField.from_grid(
        grid.ravel(), 40, 20, 0.1, 0.0, 0.0, 0.0)
    no_slip = containment.predicted_twist_clearance(
        field, 1.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.8)
    sliding_left = containment.predicted_twist_clearance(
        field, 1.0, 0.5, 0.0, 1.0, 1.0, 0.0, 0.8)
    assert sliding_left < no_slip
