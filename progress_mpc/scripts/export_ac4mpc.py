#!/usr/bin/env python3
"""Exports the sim repo's AC4MPC OCP as C code for the `Ac4Mpc` class of the `progress_mpc` ROS package.

Needs the f1tenth_parallel_sim environment (CasADi, acados_template, ACADOS_SOURCE_DIR set):

    uv run --project ../f1tenth_parallel_sim python progress_mpc/scripts/export_ac4mpc.py \
        --sim-repo ../f1tenth_parallel_sim

Takes the options of `export_acados.py` and the sim's `configs/ac4mpc.toml`: the track margin and the QP settings are
the plain MPC's, but the cost weights stay the sim's AC4MPC ones (a small steering-rate weight: the actor and the critic
supply the smoothness the plain MPC gets from its large one). It writes into `progress_mpc/generated_ac4mpc/`:
    acados/               the OCP with the critic's terminal term: its model is `ac4mpc_critic`, and the parameters
                          are the stage reference followed by the critic's linearization point and gradient
    ac4mpc_casadi.{c,h}   CasADi's C code for the model step and the OCP's cost pieces, which `ac4eval` sums
    ac4mpc_config.h       the sizes and car constants the C++ class needs

The critic's weight is not baked in: the C++ side scales the gradient it passes (the paper's beta times gamma^N, in
the units of the progress cost), so the export does not depend on the checkpoint. Nor does it need one.

The exported functions are renamed `ac4mpc_*`, so this code and `export_acados.py`'s can live in one binary.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import export_common as common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_arguments(
        parser,
        common.PACKAGE / "generated_ac4mpc",
        config="configs/ac4mpc.toml",
        q_steer_rate=-1.0,
        levenberg_marquardt=-1.0,
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as work:
        config = common.sim_config(args, Path(work) / "ac4mpc", ac4mpc=True)

        import casadi as ca

        from mpc.ac4mpc import add_critic_terminal, stage_functions
        from mpc.progress_mpc import ProgressMPC

        class Exported(ProgressMPC):
            """The sim's OCP with the critic's terminal term, this package's QP settings, and no policy."""

            def _build_solver(self):
                ocp = self._ocp()
                common.set_qp_options(ocp, args)
                self.cost_functions = stage_functions(ocp, self.config.slack_weight)
                add_critic_terminal(ocp, self._aug_dim, 1.0)
                return self._create_solver(ocp)

        mpc = Exported(config)

        out = args.out
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        common.copy_acados(config.output_dir / "c_generated_code", out)

        rate, stage, terminal = mpc.cost_functions
        step = mpc._step_fn
        generator = ca.CodeGenerator("ac4mpc_casadi.c", {"with_header": True})
        for name, fn in (("ac4mpc_step", step), ("ac4mpc_rate", rate), ("ac4mpc_stage", stage),
                         ("ac4mpc_terminal", terminal)):  # fmt: skip
            generator.add(_renamed(fn, name))
        generator.generate(f"{out}/")

        values = common.constants(mpc, config, args)
        values["kAugDim"] = (mpc._aug_dim, "int")
        values["kQProgress"] = (config.q_progress, "double")
        common.write_header(out / "ac4mpc_config.h", values, namespace="ac4mpc::generated")

    print(f"Exported to {out}")


def _renamed(fn, name: str):
    """`fn` under another name, which is also the C function's."""
    import casadi as ca

    args = [ca.SX.sym(fn.name_in(i), fn.sparsity_in(i)) for i in range(fn.n_in())]
    return ca.Function(name, args, fn.call(args), fn.name_in(), fn.name_out())


if __name__ == "__main__":
    main()
