"""Inference-only port of f1tenth_parallel_sim's `ppo.model.ActorCritic` (torch only, no ROS).

Only the deterministic action path is kept: the waypoint encoder, the history encoder, and the
policy MLP. The critic, auxiliary head, and privileged `env_encoder` are not needed on the car;
a checkpoint trained in "privileged" mode (which feeds ground-truth dynamics) cannot be deployed.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

# Layout of one history row, matching f1tenth_sim.env: [vel, yaw_rate, vel_cmd, steer_cmd].
HISTORY_DIM = 4
WAYPOINT_DIM = 4
ACTION_DIM = 2


def _mlp(in_dim, layers):
    mods = []
    last = in_dim
    for width in layers:
        mods += [nn.Linear(last, width), nn.Tanh()]
        last = width
    return nn.Sequential(*mods), last


class WindowEncoder(nn.Module):
    def __init__(self, n_steps, n_features, cfg):
        super().__init__()
        self.conv_enabled = cfg['conv']
        if self.conv_enabled:
            k = cfg['kernel_size']
            self.conv = nn.Conv1d(n_features, cfg['conv_channels'], k, padding=k // 2)
            self.conv_act = nn.Tanh()
            flat_dim = cfg['conv_channels'] * n_steps
        else:
            flat_dim = n_steps * n_features
        self.net, self.embed_dim = _mlp(flat_dim, [*cfg['hidden'], cfg['embed_dim']])

    def forward(self, window):
        if self.conv_enabled:
            window = self.conv_act(self.conv(window.transpose(1, 2)))
        return self.net(window.flatten(1))


@dataclass
class PolicyShape:
    """Observation sizes and rates the checkpoint was trained with."""

    n_waypoints: int
    history_len: int
    control_hz: float


class Policy(nn.Module):
    """Maps (vel, yaw_rate, waypoints, history) to a normalized action in [-1, 1]^2."""

    def __init__(self, shape: PolicyShape, model_cfg: dict, action_low, action_high):
        super().__init__()
        if model_cfg['mode'] == 'privileged':
            raise ValueError("a 'privileged' checkpoint needs ground-truth dynamics; deploy a 'history' or 'adapt' one")
        self.shape = shape
        self.waypoint_encoder = WindowEncoder(shape.n_waypoints, WAYPOINT_DIM, model_cfg['waypoint'])
        self.history_encoder = WindowEncoder(shape.history_len, HISTORY_DIM, model_cfg['history'])
        in_dim = 2 + ACTION_DIM + self.waypoint_encoder.embed_dim + self.history_encoder.embed_dim
        self.policy_mlp, pi_out = _mlp(in_dim, model_cfg['pi_layers'])
        self.action_net = nn.Linear(pi_out, ACTION_DIM)
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer('action_scale', 0.5 * (high - low))
        self.register_buffer('action_bias', 0.5 * (high + low))

    @torch.no_grad()
    def forward(self, vel, yaw_rate, waypoints, history):
        """Returns the env-scale [vel_cmd, steer_cmd] for one observation.

        Args:
            vel, yaw_rate: scalars.
            waypoints: (n_waypoints, 4) body-frame (x, y, d_left, d_right).
            history: (history_len, 4) rows of [vel, yaw_rate, vel_cmd, steer_cmd], newest last.
        """
        scalars = torch.tensor([[vel, yaw_rate]], dtype=torch.float32)
        history = history.unsqueeze(0)
        prev_action = (history[:, -1, 2:4] - self.action_bias) / self.action_scale
        features = torch.cat(
            [scalars, prev_action, self.waypoint_encoder(waypoints.unsqueeze(0)), self.history_encoder(history)], dim=-1
        )
        a_norm = self.action_net(self.policy_mlp(features)).clamp(-1.0, 1.0)
        return (self.action_bias + self.action_scale * a_norm)[0]


def load_policy(path: str, device: str = 'cpu') -> Policy:
    """Builds a `Policy` from a checkpoint saved by f1tenth_parallel_sim's `scripts/train.py`."""
    # weights_only: a checkpoint is plain tensors and python containers, so never unpickle arbitrary objects.
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ckpt['config']
    env, model_cfg = cfg['env'], cfg['model']
    state = ckpt['model']

    if 'action_bounds' in ckpt:
        low, high = ckpt['action_bounds']['low'], ckpt['action_bounds']['high']
    elif 'action_scale' in state and 'action_bias' in state:
        scale, bias = state['action_scale'].float().cpu(), state['action_bias'].float().cpu()
        low, high = bias - scale, bias + scale
    else:
        raise ValueError(f"{path}: no action bounds recorded; the previous-action input cannot be normalized")

    shape = PolicyShape(env['n_waypoints'], env['history_len'], float(env['control_hz']))
    policy = Policy(shape, model_cfg, low, high)
    buffers = {'action_scale', 'action_bias'}
    own = {k: v for k, v in state.items() if k in policy.state_dict() and k not in buffers}
    missing = set(policy.state_dict()) - set(own) - buffers
    if missing:
        raise ValueError(f"{path}: checkpoint is missing weights {sorted(missing)}")
    try:
        policy.load_state_dict(own, strict=False)
    except RuntimeError as exc:
        raise ValueError(
            f"{path}: weights do not fit the current observation layout (history rows are "
            f"{HISTORY_DIM} features, waypoints {WAYPOINT_DIM}); the checkpoint predates it. {exc}"
        ) from exc
    return policy.to(device).eval()
