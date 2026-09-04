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
                    or name in {"M3_figure_eight", "M4_chirp_on_circle",
                                "M7_doublets_on_circle"})
        assert takes.has_phase_independent_reference(name) is expected
        assert takes.completes_at_radial_limit(name) is name.startswith("M2_skidpad_")


def test_figure_eight_switches_lobes_from_measured_yaw():
    sequence = takes.FigureEightSequencer(
        rate_hz=10.0, laps=1, lead_sec=0.0)

    assert sequence.next_command(0.0) == (takes.M3_DELTA, takes.M3_V)
    assert sequence.next_command(2.0 * np.pi - 0.01)[0] > 0.0
    assert sequence.next_command(2.0 * np.pi + 0.01)[0] < 0.0
    assert sequence.lobe_index == 1
    assert sequence.next_command(0.0) is None
    assert sequence.complete


def test_figure_eight_leaves_feedback_headroom():
    commands = takes.build("M3_figure_eight")
    assert np.abs(commands[:, 0]).max() == takes.M3_DELTA
    assert takes.M3_DELTA + 0.10 <= takes.S_MAX


def test_figure_eight_lobe_has_a_timeout():
    sequence = takes.FigureEightSequencer(
        rate_hz=1.0, laps=1, lead_sec=0.0)
    for _ in range(sequence.max_lobe_steps + 1):
        sequence.next_command(0.0)
    assert sequence.timed_out
    assert not sequence.complete
