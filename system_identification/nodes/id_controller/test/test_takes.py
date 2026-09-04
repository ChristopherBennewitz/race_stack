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
