# Making the MPC solve fast enough for the car (2026-10-25)

## The problem

On the car's NUC a solve took **26 ms on average (max 47 ms)** against the 25 ms control period. Consequences seen in the
first drive bag (`/controller/diagnostics`, 14 s run):

| | measured on the car |
|---|---|
| solve (references + solver) | mean 26.4 ms, p90 31 ms, max 47 ms |
| control loop period | 29.6 ms mean (the timer is 25 ms) |
| `/drive` rate | 33.3 Hz instead of 40 Hz |
| odometry age when a cycle starts | median 41 ms (the executor was behind) |
| steering response | the car oscillated, growing in amplitude |

The MPC model assumes one command every 25 ms from a state that is current. At a 30 ms period with ~26 ms of that spent
solving, the plan was made from an old state and applied late.

## Where the time goes

acados' own timers over a closed loop on the sim's track (laptop, baseline settings):

| part | time |
|---|---|
| total solve | 7.69 ms |
| model linearization (CasADi functions) | 0.26 ms |
| **QP solver (HPIPM)** | **7.42 ms (97 %)** |
| condensing, regularization, integration | < 0.02 ms |

The SQP side is already one RTI iteration per cycle (`iterations_max` is 1 in the diagnostics), so there is nothing to
cut there. The time is spent inside that single iteration: HPIPM is an interior-point method and needs many of its own
inner iterations at the default tolerance of 1e-6. Reproduce with
`benchmark_solver.py --profile`.

## What was tried

Method: each variant is an acados solver option applied on top of the sim's OCP (`scripts/benchmark_solver.py`), built,
and driven in closed loop with the model as the plant from several start points, 250 steps each. Timed on a 16-thread
laptop; the NUC is about 3.4 to 4.3x slower. **Closed-loop command traces cannot be compared:** the sim's own controller
diverges from a copy of itself with its state nudged by 1e-7 (the same start ends up 0.1 to 2.5 m of path apart), so a
setting is judged by outcomes: path length within the baseline's spread, no track violation, no solver failures.

### First sweep (5 starts; wall time includes ~2.5 ms of Python overhead the C++ controller does not have)

| variant | wall ms (max) | path m, 5 starts (min-max) | max track violation | note |
|---|---|---|---|---|
| baseline | 10.20 (12.3) | 17.17 (15.4-19.2) | 0 | |
| `qp_tol` 1e-4 | 4.95 (6.8) | 17.25 (13.4-19.5) | 0 | half the time |
| `qp_solver_cond_N` 8 | 6.60 (7.6) | 16.90 (14.9-19.4) | 0 | commands identical to the baseline to 3.6e-11 |
| `hpipm_mode` SPEED | 8.34 (177.6) | 17.16 (15.4-19.2) | 0 | commands identical to 1.9e-11; one 177 ms outlier |
| **`qp_solver_iter_max` 20** | 4.17 (4.8) | 13.22 (4.4-17.3) | **0.403 m** | **rejected: the QP needs its iterations at the default tolerance** |
| **`hpipm_mode` SPEED_ABS** | 2.55 (7.7) | 4.61 (0.4-11.8) | **0.432 m** | **rejected: breaks the driving** |
| tol 1e-4 + SPEED + iter_max 30 | 3.90 (4.8) | 17.72 (15.6-19.6) | 0 | |

### Second sweep (8 starts; solver-only time; failures counted over all ~2,200 steps of the variant)

| variant | solver ms (max) | failures | path m, 8 starts (min-max) | max violation |
|---|---|---|---|---|
| baseline | 7.61 (8.2) | 1 | 17.37 (14.2-19.8) | 0 |
| `cond_N` 4 | 3.85 (4.2) | 7 | 17.14 (14.8-19.7) | 0 |
| `cond_N` 5 | 3.86 (5.8) | 2 | 17.50 (14.1-19.7) | 0 |
| `cond_N` 8 | 4.09 (4.8) | 3 | 17.25 (14.4-19.7) | 0 |
| `cond_N` 10 | 4.10 (4.2) | 0 | 17.40 (14.5-19.8) | 0 |
| `cond_N` 20 | 5.04 (5.3) | 3 | 17.29 (14.4-19.7) | 0 |
| `cond_N` 5 + tol 1e-4 | 1.34 (2.1) | 0 | 17.49 (14.4-19.7) | 0 |
| **`cond_N` 8 + tol 1e-4** | **1.38 (2.2)** | **0** | **17.34 (14.7-19.2)** | **0** |
| `cond_N` 5 + tol 1e-3 | 1.30 (1.8) | 0 | 17.46 (14.8-19.6) | 0 |
| `cond_N` 5 + tol 1e-4 + SPEED | 0.81 (1.5) | 5 | 17.52 (15.1-19.7) | 0 |

Reading it:
- Partial condensing (`cond_N`) alone gives the **same QP solution** (commands agree to 1e-11) in about half the time; it
  is pure linear algebra. The failure counts of 2 to 7 for some `cond_N` values are within the noise of a 2,200-step
  count (the baseline had 1) but are why 8 and 10 were preferred over 4 and 5.
- Loosening the QP tolerance to 1e-4 (all four HPIPM tolerances) roughly halves the time again. With one RTI iteration
  the plan is approximate anyway, and the outcomes do not change. 1e-3 gains nothing more.
- `hpipm_mode` SPEED is not worth it: the extra gain brought failures (5) back.

## What was chosen

`qp_solver_cond_N = 8` and `qp_tol = 1e-4`, set in `scripts/export_acados.py` (`--qp-cond-n`, `--qp-tol`; 0 keeps the sim's
default). Only the generated solver file and `mpc_config.h` change in the export; the large model files stay
byte-identical, so a `git pull` followed by an incremental build on the car takes about a minute.

Check of the C++ controller with the new solver (8 starts, 250 steps, model as the plant): solve (references + solver)
**1.5 ms** (it was 7.9 ms), path 18 to 23 m per start, no track violations, 0 solver failures; the Python controller on
the same starts gave the same range.

## Result on the car

Second drive bag (`mpc_run2`, 34 s), same diagnostics:

| | before | after |
|---|---|---|
| solve mean (median over windows) | 26.4 ms (max 47) | **6.3 ms** (max 13) |
| control loop period | 29.6 ms | **25.0 ms** |
| `/drive` rate | 33.3 Hz | **40 Hz** |
| odometry age at cycle start (median) | 41 ms | **19 ms** |
| solver failures | | 0 |

## What this did not touch

Considered but not changed, in case the solve time is ever a problem again:
- **Horizon length** (40 stages). The solve time scales roughly with it; not needed now.
- **`integration_substeps`** (4): the model linearization is only 0.26 ms, so reducing it gains nothing.
- **CPU power mode**: the NUC was in "balanced"; not measured separately, because the settings above were enough.
- **Compile flags**: the large generated file `progress_mpc_dyn_disc_phi_fun_jac_hess.c` (1.3 MB, an exact dynamics
  Hessian that the OCP is configured not to use, `exact_hess_dyn = False`) makes a clean build take ~6.5 min on the NUC (22 s with `-O3` on the
  laptop, 11 s at `-O1`, 1 s at `-O0`). Incremental builds do not recompile it.
