#!/usr/bin/env python3
"""Exports the sim repo's progress MPC as C code for the `progress_mpc` ROS package.

Needs the f1tenth_parallel_sim environment (CasADi, acados_template, ACADOS_SOURCE_DIR set):

    uv run --project ../f1tenth_parallel_sim python progress_mpc/scripts/export_acados.py \
        --sim-repo ../f1tenth_parallel_sim --config configs/progress_mpc.toml

Writes into `progress_mpc/generated/`:
    acados/         acados' generated solver (model, cost and constraint functions, capsule API)
    mpc_step.{c,h}  CasADi's C code for one control period of the model, used for warm starts
    mpc_config.h    the sizes and car constants the C++ wrapper needs

Everything the OCP bakes in (car, horizon, weights, command limits) is fixed by this export; the track
is a runtime input. Re-run it, and commit the result, whenever the sim repo's MPC or car config changes.

QP solver settings. The solve is one SQP-RTI iteration, and almost all of its time (97 %) is the QP inside it: the
model evaluation is ~0.2 ms of ~7.6 ms. Two settings that leave the solution and the closed-loop behaviour unchanged
cut it 5.5x (timed on the sim's track with the same model as the plant, 8 start points x 250 steps, 0 solver failures):

    --qp-cond-n 8     partial condensing: the 40 stages are condensed into 8 blocks. The same QP solved with
                      different linear algebra: commands agree to 1e-11, ~35 % faster.
    --qp-tol 1e-4     HPIPM stops at a looser tolerance (acados default 1e-6). With one RTI iteration the plan is
                      approximate anyway; path length 17.34 m against 17.37 m, no track violations.

Not used, because they changed the behaviour: capping the QP iterations at 20 (track violations of 0.4 m) and HPIPM
mode SPEED_ABS (breaks); SPEED failed more often. The sim repo's defaults stay in place when a flag is 0.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim-repo", type=Path, required=True, help="path to f1tenth_parallel_sim")
    parser.add_argument("--config", type=Path, default=Path("configs/progress_mpc.toml"),
                        help="ProgressMPCConfig TOML, relative to the sim repo")
    parser.add_argument("--out", type=Path, default=PACKAGE / "generated")
    parser.add_argument("--qp-cond-n", type=int, default=8, help="partial condensing blocks (0: sim default)")
    parser.add_argument("--qp-tol", type=float, default=1e-4, help="HPIPM tolerance (0: sim default)")
    args = parser.parse_args()

    sim_repo = args.sim_repo.resolve()
    sys.path.insert(0, str(sim_repo / "src"))

    import casadi as ca

    from mpc.progress_mpc import ProgressMPC, ProgressMPCConfig, command_step_limits
    from f1tenth_sim.state import STATE_DIM

    config = ProgressMPCConfig.from_toml(sim_repo / args.config)
    with tempfile.TemporaryDirectory() as work:
        config = config.model_copy(update={"output_dir": Path(work) / "mpc"})

        class Exported(ProgressMPC):
            """The sim's OCP with this package's QP solver settings."""

            def _ocp(self):
                ocp = super()._ocp()
                if args.qp_cond_n > 0:
                    ocp.solver_options.qp_solver_cond_N = args.qp_cond_n
                if args.qp_tol > 0.0:
                    ocp.solver_options.qp_tol = args.qp_tol
                return ocp

        mpc = Exported(config)

        out = args.out
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        shutil.copytree(
            config.output_dir / "c_generated_code", out / "acados",
            ignore=shutil.ignore_patterns("*.o", "*.so", "*.a", "main_*", "Makefile", "*.pxd",
                                          "acados_sim_solver_*", "libacados_ocp_solver_*", "CMakeCache*",
                                          "CMakeFiles", "build"))

        generator = ca.CodeGenerator("mpc_step.c", {"with_header": True})
        step = mpc._step_fn
        generator.add(step)
        generator.generate(f"{out}/")

        car, act = mpc.car, mpc.car.actuation
        step_lo, step_hi = command_step_limits(act, config.control_dt)
        speed_max = act.speed_max if config.max_speed is None else min(config.max_speed, act.speed_max)
        values = {
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
        }
        lines = ["// Generated by scripts/export_acados.py; do not edit.", "#pragma once", "",
                 "namespace progress_mpc::generated {", ""]
        lines += [f"constexpr {kind} {name} = {value!r};" for name, (value, kind) in values.items()]
        lines += ["", "}  // namespace progress_mpc::generated", ""]
        (out / "mpc_config.h").write_text("\n".join(lines))

    print(f"Exported to {out}")


if __name__ == "__main__":
    main()
