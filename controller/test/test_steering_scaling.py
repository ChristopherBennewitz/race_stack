"""Tests for explicit controller-side steering scaling."""

import pytest

from controller.steering_scaling import scale_steering


def scaled(steer, lookup_speed, measured_speed):
    return scale_steering(
        steer=steer,
        lookup_speed=lookup_speed,
        measured_speed=measured_speed,
        start_scale_speed=7.0,
        end_scale_speed=8.0,
        downscale_factor=0.2,
        boost_per_mps=0.1,
        boost_max=1.25,
    )


def test_low_speed_steering_is_unchanged():
    assert scaled(0.2, 0.0, 0.0) == pytest.approx(0.2)


def test_explicit_parameters_preserve_previous_scaling():
    assert scaled(0.2, 2.0, 2.0) == pytest.approx(0.24)


def test_boost_and_downscale_are_both_applied():
    # The previous defaults produce 1.25 boost * 0.8 downscale = 1.0.
    assert scaled(0.2, 8.0, 8.0) == pytest.approx(0.2)
