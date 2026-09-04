import numpy as np

from id_controller import takes


def test_manifest_has_expected_groups():
    assert len(takes.TAKES) == 19
    assert len(takes.CORRIDOR) == 5
    assert set(takes.TAKES).isdisjoint(takes.CORRIDOR)


def test_all_takes_are_finite_and_within_command_bounds():
    for name in takes.ALL:
        commands = takes.build(name)
        assert commands.ndim == 2
        assert commands.shape[1] == 2
        assert len(commands) > 0
        assert np.isfinite(commands).all()
        assert np.abs(commands[:, 0]).max() <= takes.S_MAX
        assert commands[:, 1].max() <= takes.V_CEILING


def test_no_take_requests_reverse_motion():
    for name in takes.ALL:
        assert takes.build(name)[:, 1].min() >= 0.0


def test_fast_steering_excitation_is_not_in_containment_reference():
    for name, expected in (("M4_chirp_on_circle", 0.24),
                           ("M7_doublets_on_circle", 0.17),
                           ("C5_chirp_highband", 0.0)):
        commands = takes.build(name)
        reference = takes.reference_steering(name, commands)
        assert np.allclose(reference, expected)


def test_closed_circle_tracking_policy():
    for name in takes.ALL:
        expected = (name.startswith(("M1_circle_", "M2_skidpad_", "M5_speed_steps_"))
                    or name in {"M4_chirp_on_circle", "M7_doublets_on_circle"})
        assert takes.has_phase_independent_reference(name) is expected
        assert takes.permits_radial_departure(name) is name.startswith("M2_skidpad_")
