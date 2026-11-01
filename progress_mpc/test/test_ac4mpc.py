"""Tests for the C++ AC4MPC through its Python bindings; the plant is the MPC's own model.

The networks are a small random TorchScript module with the exporter's interface (`act`, `value`, `meta`), so the
tests need torch and the package built with libtorch, but not the sim repo. `test_matches_the_sim_controller` runs the
sim repo's Python AC4MPC next to it, when that repo (and its environment) is importable.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

progress_mpc = pytest.importorskip("progress_mpc")
if progress_mpc.Ac4Mpc is None:  # built without libtorch or without generated_ac4mpc/
    pytest.skip("progress_mpc was built without AC4MPC", allow_module_level=True)
torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

N_WAYPOINTS, HISTORY = 10, 4
CONTROL_DT = 0.025


def stadium(straight=6.0, radius=2.5, spacing=0.1):
    """Counter-clockwise stadium centerline starting at the bottom straight's start."""
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


class TinyPolicy(nn.Module):
    """A random actor-critic over the exporter's observation (the layout of `scripts/export_policy.py`)."""

    def __init__(self, control_dt=CONTROL_DT):
        super().__init__()
        self.waypoints = nn.Sequential(nn.Flatten(), nn.Linear(N_WAYPOINTS * 4, 16), nn.Tanh())
        self.history = nn.Sequential(nn.Flatten(), nn.Linear(HISTORY * 4, 16), nn.Tanh())
        features = 2 + 2 + 16 + 16
        self.pi = nn.Sequential(nn.Linear(features, 32), nn.Tanh(), nn.Linear(32, 2))
        self.vf = nn.Sequential(nn.Linear(features, 32), nn.Tanh(), nn.Linear(32, 1))
        self.register_buffer("action_scale", torch.tensor([2.5, 0.3], dtype=torch.float64))
        self.register_buffer("action_bias", torch.tensor([2.5, 0.0], dtype=torch.float64))
        meta = [N_WAYPOINTS, HISTORY, control_dt, 0.95, 1.0, 2.5, 0.3, 2.5, 0.0]
        self.register_buffer("meta", torch.tensor(meta, dtype=torch.float64))

    def _features(self, scalars, waypoints, history):
        previous = (history[:, -1, 2:4] - self.action_bias) / self.action_scale
        return torch.cat([scalars, previous, self.waypoints(waypoints), self.history(history)], dim=-1)

    def act(self, scalars, waypoints, history):
        return self.pi(self._features(scalars, waypoints, history))

    def value(self, scalars, waypoints, history):
        return self.vf(self._features(scalars, waypoints, history))

    def forward(self, scalars, waypoints, history):
        return self.act(scalars, waypoints, history)


def export_tiny(path, control_dt=CONTROL_DT, seed=0):
    torch.manual_seed(seed)
    module = TinyPolicy(control_dt).double().eval()
    example = (
        torch.zeros(1, 2, dtype=torch.float64),
        torch.zeros(1, N_WAYPOINTS, 4, dtype=torch.float64),
        torch.zeros(1, HISTORY, 4, dtype=torch.float64),
    )
    traced = torch.jit.trace_module(module, {"act": example, "value": example, "forward": example}, strict=False)
    traced.save(str(path))
    return str(path)


@pytest.fixture(scope="module")
def policy(tmp_path_factory):
    return export_tiny(tmp_path_factory.mktemp("policy") / "policy.ts")


def make(policy, mode="full", **kwargs):
    track = stadium()
    controller = progress_mpc.Ac4Mpc(policy, mode=mode, **kwargs)
    controller.set_track(track, np.full(len(track), 1.0), np.full(len(track), 1.0))
    return controller


def initial_state(mpc, speed=1.0):
    return np.concatenate([[1.0, 0.0, 0.0, speed, 0.0, 0.0, 0.0], np.zeros(mpc.history_length)])


def drive(mpc, steps, speed=1.0):
    """Closed loop on the model. Returns the commands, the arc length of the car and the last state."""
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


def test_step_needs_a_track(policy):
    with pytest.raises(RuntimeError, match="set_track"):
        progress_mpc.Ac4Mpc(policy).step(np.zeros(6))


def test_a_policy_for_another_control_period_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="acts every"):
        progress_mpc.Ac4Mpc(export_tiny(tmp_path / "slow.ts", control_dt=0.05))


def test_a_missing_policy_is_an_error(tmp_path):
    with pytest.raises((RuntimeError, ValueError)):
        progress_mpc.Ac4Mpc(str(tmp_path / "nothing.ts"))


def test_the_mode_is_checked(policy):
    with pytest.raises(ValueError, match="mode"):
        progress_mpc.Ac4Mpc(policy, mode="both")


@pytest.mark.parametrize("mode", ["critic", "actor", "full"])
def test_every_mode_drives_forward_within_the_command_limits(policy, mode):
    mpc = make(policy, mode, eval_correction=0.5 if mode == "full" else 0.0, eval_rollout=5 if mode == "full" else 0)
    commands, frenet, _ = drive(mpc, 120)
    assert mpc.stats["failures"] == 0
    s = np.unwrap(frenet[:, 0], period=mpc.track_length)
    assert s[-1] - s[0] > 2.0
    steps = np.diff(commands, axis=0)
    steer_step, speed_up, speed_down = mpc.step_limits
    assert np.all(np.abs(steps[:, 0]) <= steer_step + 1e-6)
    assert np.all(steps[:, 1] <= speed_up + 1e-6) and np.all(steps[:, 1] >= -speed_down - 1e-6)
    assert np.all(commands[:, 1] >= -1e-9) and np.all(commands[:, 1] <= mpc.speed_max + 1e-9)
    expected = {"mpc_active"} if mode == "critic" else {"mpc_active", "mpc_parallel", "actor"}
    assert set(mpc.candidate_wins) <= expected
    assert sum(mpc.candidate_wins.values()) == 120


def test_threads_give_the_same_commands_as_one_thread(policy):
    threaded, single = make(policy, threads=True), make(policy, threads=False)
    a, _, _ = drive(threaded, 60)
    b, _, _ = drive(single, 60)
    assert np.allclose(a, b, atol=1e-9)
    assert threaded.candidate_wins == single.candidate_wins


def test_the_actor_wins_some_steps_and_the_solvers_others(policy):
    mpc = make(policy)
    drive(mpc, 100)
    assert len(mpc.candidate_wins) >= 2


def test_the_critic_gradient_matches_finite_differences(policy):
    mpc = make(policy)
    rng = np.random.default_rng(0)
    nz = 7 + mpc.history_length
    xa = np.zeros(nz + 1)
    xa[:2] = [3.0, 0.2]
    xa[2:6] = [0.1, 1.5, 0.05, -0.2]
    xa[7:] = rng.normal(0.0, 0.1, nz + 1 - 7)
    past = rng.normal(0.0, 1.0, (HISTORY - 1, 4))
    idx = mpc.policy_nearest(xa[0], xa[1], -1)
    value, grad = mpc.policy_value(xa, idx, past)
    for i in [0, 1, 2, 3, 5, 7, nz]:
        h = 1e-6
        up, down = xa.copy(), xa.copy()
        up[i] += h
        down[i] -= h
        numeric = (mpc.policy_value(up, idx, past)[0] - mpc.policy_value(down, idx, past)[0]) / (2 * h)
        assert grad[i] == pytest.approx(numeric, rel=1e-5, abs=1e-8), i
    assert value == mpc.policy_value(xa, idx, past)[0]


def test_the_actor_stays_inside_the_action_bounds(policy):
    mpc = make(policy)
    xa = np.zeros(7 + mpc.history_length + 1)
    xa[:6] = [3.0, 0.0, 0.0, 2.0, 0.0, 0.0]
    steer, speed = mpc.policy_act(xa, mpc.policy_nearest(3.0, 0.0, -1), np.zeros((HISTORY - 1, 4)))
    assert -0.3 <= steer <= 0.3 and 0.0 <= speed <= 5.0


def test_reset_restarts_from_scratch(policy):
    mpc = make(policy)
    a, _, _ = drive(mpc, 25)
    b, _, _ = drive(mpc, 25)
    assert np.allclose(a, b, atol=1e-9)


def test_the_track_clearance_is_exposed(policy):
    mpc = make(policy)
    assert mpc.clearance_margin > 0.2 and mpc.horizon == 40 and mpc.control_dt == pytest.approx(CONTROL_DT)


def _sim_repo():
    root = os.environ.get("F1TENTH_SIM_REPO")
    path = Path(root) if root else Path(__file__).resolve().parents[3] / "f1tenth_parallel_sim"
    return path if (path / "src" / "mpc" / "ac4mpc.py").exists() else None


@pytest.mark.parametrize("mode", ["critic", "full"])
def test_matches_the_sim_controller(tmp_path, mode):
    """The same checkpoint, track and measurements through the sim's Python AC4MPC and the C++ one."""
    repo = _sim_repo()
    if repo is None:
        pytest.skip("f1tenth_parallel_sim is not next to this repo (set F1TENTH_SIM_REPO)")
    sys.path[:0] = [str(repo / "src"), str(repo / "scripts"), str(repo / "tests")]
    cwd = os.getcwd()
    os.chdir(repo)
    try:
        pytest.importorskip("acados_template")
        from export_policy import export
        from mpc.ac4mpc import AC4MPC, AC4MPCConfig
        from mpc.rollout import VEHICLE_STATE_DIM, build_env
        from ppo.obs import build_action_td
        from test_mpc_ac4mpc import _checkpoint

        checkpoint = _checkpoint(tmp_path / "policy.pt")
        module = export(checkpoint, tmp_path / "policy.ts")

        class Sim(AC4MPC):
            """The sim controller with the exported OCP's QP settings."""

            def _ocp(self):
                ocp = super()._ocp()
                ocp.solver_options.qp_solver_cond_N = 8
                ocp.solver_options.qp_tol = 1e-4
                return ocp

        config = AC4MPCConfig.from_toml("configs/ac4mpc.toml").model_copy(
            update=dict(
                output_dir=tmp_path / "sim",
                checkpoint=checkpoint,
                mode=mode,
                worker_processes=False,
                track_margin=0.1,
                value_weight=0.2,
                reinit_period=5,
                track_dir=repo / "configs/maps/generated/oval",
                track_name="oval",
            )
        )
        if not (config.track_dir / "oval.yaml").exists():
            pytest.skip("run scripts/generate_maps.py in the sim repo first")
        reference = Sim(config)
        track = reference.track
        mpc = progress_mpc.Ac4Mpc(str(module), mode=mode, value_weight=0.2, reinit_period=5, track_spacing=0.0)
        mpc.set_track(
            track.centerline.numpy().astype(float),
            track._w_left.numpy().astype(float),
            track._w_right.numpy().astype(float),
        )
        env = build_env(config)
        env.reset()
        state = env.sim.state[0, :VEHICLE_STATE_DIM].numpy().copy()
        reference.reset(state)
        mpc.reset()
        # The two solvers start from the same state and agree to rounding for the first steps; the closed loop
        # then separates as any two RTI runs do (the solver's tolerance is 1e-4).
        for k in range(8):
            a = np.array(reference.step(state))
            b = np.array(mpc.step(state[:6], float(state[6])))
            assert np.abs(a - b).max() < 1e-6, k
            assert mpc.stats["candidate"] == reference.last_candidate, k
            env.step(build_action_td(env, torch.tensor([[b[1], b[0]]], dtype=torch.float32)))
            speed, steer = env.applied_command[0].tolist()
            reference.report_applied(steer, speed)
            mpc.report_applied(steer, speed)
            state = env.sim.state[0, :VEHICLE_STATE_DIM].numpy().copy()
    finally:
        os.chdir(cwd)
