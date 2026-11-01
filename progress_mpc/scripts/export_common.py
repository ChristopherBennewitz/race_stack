"""What the two exporters (`export_acados.py`, `export_ac4mpc.py`) share: options, the sim config, the car constants.

Both export the sim repo's OCP with the same QP solver settings, track margin and weights, so the plain MPC and the
AC4MPC solve the same problem apart from the critic's terminal term.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]

_IGNORED = (
    "*.o",
    "*.so",
    "*.a",
    "main_*",
    "Makefile",
    "*.pxd",
    "acados_sim_solver_*",
    "libacados_ocp_solver_*",
    "CMakeCache*",
    "CMakeFiles",
    "build",
)


def add_arguments(
    parser: argparse.ArgumentParser,
    default_out: Path,
    config: str = "configs/progress_mpc.toml",
    q_steer_rate: float = 30.0,
    levenberg_marquardt: float = 0.03,
) -> None:
    """The options both exporters take, with the defaults of the one that calls."""
    parser.add_argument("--sim-repo", type=Path, required=True, help="path to f1tenth_parallel_sim")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(config),
        help="ProgressMPCConfig TOML, relative to the sim repo",
    )
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("--qp-cond-n", type=int, default=8, help="partial condensing blocks (0: sim default)")
    parser.add_argument("--qp-tol", type=float, default=1e-4, help="HPIPM tolerance (0: sim default)")
    parser.add_argument(
        "--track-margin",
        type=float,
        default=0.10,
        help="clearance beyond the collision radius (m); negative keeps the sim config's 0.2",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=0.0,
        help="speed command cap (m/s); 0 (default) keeps the actuation manager's limit",
    )
    parser.add_argument(
        "--q-steer-rate",
        type=float,
        default=q_steer_rate,
        help="steering-rate weight; negative keeps the sim config's. For the plain MPC 30 (config: 1) halves the "
        "steering reversals in the hangar sim",
    )
    parser.add_argument(
        "--levenberg-marquardt",
        type=float,
        default=levenberg_marquardt,
        help="Levenberg-Marquardt regularization; negative keeps the sim config's. For the plain MPC 0.03 "
        "(config: 0.01) steadies the plan a little",
    )


def sim_config(args: argparse.Namespace, work: Path, ac4mpc: bool = False):
    """The sim repo's config (`ProgressMPCConfig`, or `AC4MPCConfig`) with this package's overrides, generating into `work`."""
    sim_repo = args.sim_repo.resolve()
    sys.path.insert(0, str(sim_repo / "src"))
    if ac4mpc:
        from mpc.ac4mpc import AC4MPCConfig as ConfigClass
    else:
        from mpc.progress_mpc import ProgressMPCConfig as ConfigClass

    config = ConfigClass.from_toml(sim_repo / args.config)
    overrides = {}
    if args.track_margin >= 0.0:
        overrides["track_margin"] = args.track_margin
    if args.max_speed > 0.0:
        overrides["max_speed"] = args.max_speed
    if args.q_steer_rate >= 0.0:
        overrides["q_steer_rate"] = args.q_steer_rate
    if args.levenberg_marquardt >= 0.0:
        overrides["levenberg_marquardt"] = args.levenberg_marquardt
    return config.model_copy(update={**overrides, "output_dir": work})


def set_qp_options(ocp, args: argparse.Namespace) -> None:
    """Applies this package's QP solver settings to an acados OCP."""
    if args.qp_cond_n > 0:
        ocp.solver_options.qp_solver_cond_N = args.qp_cond_n
    if args.qp_tol > 0.0:
        ocp.solver_options.qp_tol = args.qp_tol


def copy_acados(generated: Path, out: Path) -> None:
    """Copies acados' generated solver into `out/acados`, without build products."""
    shutil.copytree(generated, out / "acados", ignore=shutil.ignore_patterns(*_IGNORED))


def constants(mpc, config, args: argparse.Namespace) -> dict[str, tuple[object, str]]:
    """The sizes and car constants the C++ wrapper needs, as `name: (value, c++ type)`."""
    from f1tenth_sim.state import STATE_DIM
    from mpc.progress_mpc import command_step_limits

    car, act = mpc.car, mpc.car.actuation
    step_lo, step_hi = command_step_limits(act, config.control_dt)
    speed_max = act.speed_max if config.max_speed is None else min(config.max_speed, act.speed_max)
    return {
        "kHorizon": (config.horizon_nodes, "int"),
        "kVehicleDim": (STATE_DIM, "int"),
        "kHistory": (mpc._n_history, "int"),
        "kSteerHistory": (mpc._n_steer, "int"),
        "kControlDt": (config.control_dt, "double"),
        "kSteerGain": (car.vehicle.steer_gain, "double"),
        "kSteerCubic": (car.vehicle.steer_cubic, "double"),
        "kSteerBias": (car.vehicle.steer_bias, "double"),
        "kSteerLatency": (car.vehicle.steer_latency, "double"),
        "kSteerDelay": (car.vehicle.steer_delay, "double"),
        "kLr": (car.vehicle.l_r, "double"),
        "kSteerStep": (float(step_hi[0]), "double"),
        "kSpeedStepUp": (float(step_hi[1]), "double"),
        "kSpeedStepDown": (float(-step_lo[1]), "double"),
        "kSteerMax": (act.steering_max, "double"),
        "kSpeedMax": (speed_max, "double"),
        "kClearanceMargin": (car.vehicle.collision_radius + config.track_margin, "double"),
        "kQpCondN": (args.qp_cond_n, "int"),
        "kQpTol": (args.qp_tol, "double"),
        "kTrackMargin": (config.track_margin, "double"),
    }


def write_header(path: Path, values: dict[str, tuple[object, str]], namespace: str = "progress_mpc::generated") -> None:
    """Writes the constants as a C++ header."""
    lines = ["// Generated by the exporter in scripts/; do not edit.", "#pragma once", "", f"namespace {namespace} {{", ""]
    lines += [f"constexpr {kind} {name} = {value!r};" for name, (value, kind) in values.items()]
    lines += ["", f"}}  // namespace {namespace}", ""]
    path.write_text("\n".join(lines))
