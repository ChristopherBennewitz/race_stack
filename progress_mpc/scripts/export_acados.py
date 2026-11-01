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

Track clearance and speed. The sim's MPC keeps the centre of gravity `collision radius + track_margin` (0.284 + 0.2 = 0.484 m)
inside the track limits, tuned for wider sim tracks. On the 1.7 m wide hangar track that leaves a corridor of 0.7 m on
average (0.13 m in the narrowest places) and the MPC crashed in 5 of 6 runs, 3 of 6 even without any delay. With the car's
measured delays, 10 starts x 60 s in the sim (`docs/track_clearance.md`): margin 0.10 -> 1/10 crashes, 0.15 -> 8/10, 0.07 -> 1/10.

    --track-margin 0.10   clearance 0.384 m: 6 cm of worst-case wall clearance (the rectangle's corners are 0.32 m from
                          the centre of gravity), 24 cm when the car is aligned with the track
    --max-speed 0         no cap of its own: the speed limit is the actuation manager's (5 m/s). A cap such as
                          `--max-speed 4.0` is a cheap safety margin for first drives (sim: no crash, slightly less
                          progress), not a requirement: margin 0.10 + latency compensation at 5 m/s drove 10 of 10
                          sim runs without contact

Not used, because they changed the behaviour: capping the QP iterations at 20 (track violations of 0.4 m) and HPIPM
mode SPEED_ABS (breaks); SPEED failed more often. The sim repo's defaults stay in place when a flag is 0.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import export_common as common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_arguments(parser, common.PACKAGE / "generated")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as work:
        config = common.sim_config(args, Path(work) / "mpc")

        import casadi as ca

        from mpc.progress_mpc import ProgressMPC

        class Exported(ProgressMPC):
            """The sim's OCP with this package's QP solver settings."""

            def _ocp(self):
                ocp = super()._ocp()
                common.set_qp_options(ocp, args)
                return ocp

        mpc = Exported(config)

        out = args.out
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        common.copy_acados(config.output_dir / "c_generated_code", out)

        generator = ca.CodeGenerator("mpc_step.c", {"with_header": True})
        generator.add(mpc._step_fn)
        generator.generate(f"{out}/")
        common.write_header(out / "mpc_config.h", common.constants(mpc, config, args))

    print(f"Exported to {out}")


if __name__ == "__main__":
    main()
