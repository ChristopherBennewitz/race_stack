#!/usr/bin/env python3
"""Times the progress MPC's solver settings on the sim's track and checks that the driving does not change.

Needs the f1tenth_parallel_sim environment (CasADi, acados_template, ACADOS_SOURCE_DIR set). Each variant is a set
of acados solver options applied on top of the sim's OCP; every variant is built (about 15 s), then driven in
closed loop with the model as the plant from several start points on the track.

    uv run --project ../f1tenth_parallel_sim python progress_mpc/scripts/benchmark_solver.py \
        --sim-repo ../f1tenth_parallel_sim
    ... --profile                      # where one solve spends its time (acados' own timers)
    ... --variants baseline "cond_N 8 + tol 1e-4" --starts 10 130

Per variant it prints the solver-only time, the number of solver failures, and the outcome of the closed loops: path
length driven in 250 steps and the largest track violation. Judge a setting by those outcomes, not by comparing command
traces: the closed loop is chaotic, and the sim's controller diverges from a copy of itself nudged by 1e-7 (path
lengths of the same start differ by up to 2.5 m). A setting is acceptable if the path lengths stay within the baseline's
spread, there is no track violation and no failures. Results and the decision: progress_mpc/docs/solve_time.md.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

VARIANTS = {
    "baseline": {},
    "qp_tol 1e-4": {"qp_tol": 1e-4},
    "iter_max 20": {"qp_solver_iter_max": 20},
    "hpipm SPEED": {"hpipm_mode": "SPEED"},
    "hpipm SPEED_ABS": {"hpipm_mode": "SPEED_ABS"},
    "cond_N 4": {"qp_solver_cond_N": 4},
    "cond_N 5": {"qp_solver_cond_N": 5},
    "cond_N 8": {"qp_solver_cond_N": 8},
    "cond_N 10": {"qp_solver_cond_N": 10},
    "cond_N 20": {"qp_solver_cond_N": 20},
    "cond_N 5 + tol 1e-4": {"qp_solver_cond_N": 5, "qp_tol": 1e-4},
    "cond_N 8 + tol 1e-4": {"qp_solver_cond_N": 8, "qp_tol": 1e-4},  # the one exported
    "cond_N 5 + tol 1e-3": {"qp_solver_cond_N": 5, "qp_tol": 1e-3},
    "cond_N 5 + tol 1e-4 + SPEED": {"qp_solver_cond_N": 5, "qp_tol": 1e-4, "hpipm_mode": "SPEED"},
}


def build(options, config_path):
    from mpc.progress_mpc import ProgressMPC, ProgressMPCConfig

    class Variant(ProgressMPC):
        def _ocp(self):
            ocp = super()._ocp()
            for name, value in options.items():
                setattr(ocp.solver_options, name, value)
            return ocp

    config = ProgressMPCConfig.from_toml(config_path).model_copy(update={"output_dir": Path(tempfile.mkdtemp()) / "m"})
    return Variant(config)


def initial_state(mpc, index):
    p, t = mpc._center[index], mpc._tangent[index]
    return np.concatenate([[p[0], p[1], np.arctan2(t[1], t[0]), 0.5, 0, 0, 0], np.zeros(mpc._n_history)])


def closed_loop(mpc, start, steps=250):
    """Drives the model with the controller; returns (solver seconds, path length, largest track violation)."""
    z = initial_state(mpc, start)
    mpc.reset(None)
    seconds, path, violation, previous = [], 0.0, 0.0, z[:2].copy()
    for _ in range(steps):
        command = mpc.step(z[:7])
        seconds.append(mpc.last_solver_only_seconds)
        z = np.asarray(mpc._step_fn(z, np.array(command))).ravel()
        mpc.report_applied(*command)
        path += float(np.linalg.norm(z[:2] - previous))
        previous = z[:2].copy()
        i = int(np.argmin(np.sum((mpc._center - z[:2]) ** 2, axis=1)))
        offset = z[:2] - mpc._center[i]
        lateral = -mpc._tangent[i][1] * offset[0] + mpc._tangent[i][0] * offset[1]
        violation = max(violation, lateral - mpc._w_left[i], -lateral - mpc._w_right[i], 0.0)
    return np.array(seconds[20:]), path, violation


def profile(mpc, steps=250):
    """acados' own timers averaged over a closed loop: where one solve spends its time."""
    z = initial_state(mpc, 10)
    mpc.reset(None)
    keys = ["time_tot", "time_lin", "time_qp", "time_qp_solver_call", "time_qp_xcond", "time_reg", "time_sim"]
    sums = {k: [] for k in keys}
    for _ in range(steps):
        command = mpc.step(z[:7])
        for key in keys:
            sums[key].append(float(mpc._solver.get_stats(key)))
        z = np.asarray(mpc._step_fn(z, np.array(command))).ravel()
        mpc.report_applied(*command)
    for key in keys:
        print(f"  {key[5:]:16s} {1e3 * np.mean(sums[key][20:]):7.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim-repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/progress_mpc.toml"), help="relative to the sim repo")
    parser.add_argument("--variants", nargs="*", default=None, help=f"names from: {', '.join(VARIANTS)} (default: all)")
    parser.add_argument("--starts", type=int, nargs="*", default=[10, 50, 90, 130, 170, 210, 250, 290],
                        help="centerline indices to start the closed loops from")
    parser.add_argument("--profile", action="store_true", help="only print the baseline's time breakdown")
    args = parser.parse_args()

    sim_repo = args.sim_repo.resolve()
    sys.path.insert(0, str(sim_repo / "src"))
    config_path = sim_repo / args.config

    if args.profile:
        print("baseline solve, acados timers (mean over a closed loop):")
        profile(build({}, config_path))
        return

    names = args.variants or list(VARIANTS)
    print(f"{'variant':30s} {'solver ms':>9s} {'max':>6s} {'fails':>5s}  {'path m mean (min-max)':>24s} {'max viol m':>10s}")
    for name in names:
        mpc = build(VARIANTS[name], config_path)
        seconds, paths, violation = [], [], 0.0
        for start in args.starts:
            s, path, v = closed_loop(mpc, start)
            seconds.append(s)
            paths.append(path)
            violation = max(violation, v)
        ms = 1e3 * np.concatenate(seconds)
        print(f"{name:30s} {ms.mean():9.2f} {ms.max():6.1f} {mpc.total_solver_failures:5d}  "
              f"{np.mean(paths):8.2f} ({min(paths):5.1f}-{max(paths):5.1f}) {violation:10.3f}", flush=True)


if __name__ == "__main__":
    main()
