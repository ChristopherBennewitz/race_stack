"""Tests for the ROS-free parts of rl_controller."""

import numpy as np
import pytest
import torch

from rl_controller.observation import Centerline, HistoryBuffer
from rl_controller.policy import Policy, PolicyShape, load_policy

ENC = {'hidden': [8], 'embed_dim': 4, 'conv': True, 'conv_channels': 3, 'kernel_size': 5}
MODEL_CFG = {'mode': 'history', 'pi_layers': [16], 'waypoint': ENC, 'history': ENC}


def circle(n=200, r=5.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def test_lookahead_is_in_body_frame():
    c = Centerline(circle(), np.full(200, 1.0), np.full(200, 2.0))
    wp = c.lookahead(5.0, 0.0, np.pi / 2, 3)  # at angle 0, heading along the ccw tangent
    assert c.idx == 0
    assert wp.shape == (3, 4)
    assert wp[0, 0] > 0 and abs(wp[0, 1]) < 0.2  # next point is ahead, not to the side
    assert (wp[:, 2] == 1.0).all() and (wp[:, 3] == 2.0).all()


def test_clockwise_centerline_is_reversed_and_widths_swapped():
    c = Centerline(circle()[::-1], np.full(200, 1.0), np.full(200, 2.0))
    wp = c.lookahead(5.0, 0.0, np.pi / 2, 3)
    assert wp[0, 0] > 0
    assert (wp[:, 2] == 2.0).all() and (wp[:, 3] == 1.0).all()


def test_nearest_recovers_from_a_jump():
    c = Centerline(circle(), np.ones(200), np.ones(200))
    c.nearest(5.0, 0.0)
    assert c.nearest(-5.0, 0.0) == 100


def test_history_newest_last():
    h = HistoryBuffer(3)
    h.push(1, 2, 3, 4)
    buf = h.push(5, 6, 7, 8)
    assert buf.shape == (3, 4)
    assert buf[-1].tolist() == [5, 6, 7, 8] and buf[0].tolist() == [0, 0, 0, 0]


def test_policy_output_stays_in_action_bounds():
    policy = Policy(PolicyShape(10, 6, 40.0), MODEL_CFG, [0.0, -0.3], [4.0, 0.3])
    for p in policy.parameters():
        torch.nn.init.normal_(p, std=3.0)
    vel_cmd, steer_cmd = policy(3.0, 0.2, torch.randn(10, 4), torch.randn(6, 4)).tolist()
    assert -1e-6 <= vel_cmd <= 4.0 + 1e-6 and abs(steer_cmd) <= 0.3 + 1e-6


def test_privileged_checkpoint_is_rejected():
    with pytest.raises(ValueError):
        Policy(PolicyShape(10, 6, 40.0), {**MODEL_CFG, 'mode': 'privileged'}, [0, -1], [1, 1])


def test_clockwise_direction_keeps_a_clockwise_centerline():
    cw = circle()[::-1]
    c = Centerline(cw, np.full(200, 1.0), np.full(200, 2.0), direction='cw')
    wp = c.lookahead(5.0, 0.0, -np.pi / 2, 3)  # at angle 0, heading along the cw tangent
    assert wp[0, 0] > 0 and abs(wp[0, 1]) < 0.2
    assert (wp[:, 2] == 1.0).all() and (wp[:, 3] == 2.0).all()  # not reversed, widths not swapped


def test_a_counter_clockwise_centerline_is_reversed_for_cw():
    c = Centerline(circle(), np.full(200, 1.0), np.full(200, 2.0), direction='cw')
    wp = c.lookahead(5.0, 0.0, -np.pi / 2, 3)
    assert wp[0, 0] > 0
    assert (wp[:, 2] == 2.0).all() and (wp[:, 3] == 1.0).all()


def test_invalid_direction_is_rejected():
    with pytest.raises(ValueError):
        Centerline(circle(), np.ones(200), np.ones(200), direction='left')


def test_checkpoint_round_trips_with_weights_only_loading(tmp_path):
    policy = Policy(PolicyShape(10, 6, 40.0), MODEL_CFG, [0.0, -0.3], [4.0, 0.3])
    path = tmp_path / 'ckpt.pt'
    torch.save(
        {
            'model': policy.state_dict(),
            'config': {'env': {'n_waypoints': 10, 'history_len': 6, 'control_hz': 40}, 'model': MODEL_CFG},
            'action_bounds': {'low': [0.0, -0.3], 'high': [4.0, 0.3]},
        },
        path,
    )
    loaded = load_policy(str(path))
    wp, hist = torch.randn(10, 4), torch.randn(6, 4)
    assert torch.allclose(policy(2.0, 0.1, wp, hist), loaded(2.0, 0.1, wp, hist), atol=1e-6)
    assert loaded.shape.control_hz == 40.0
