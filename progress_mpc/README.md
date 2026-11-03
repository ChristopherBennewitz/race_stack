# progress_mpc

C++ port of the progress-maximizing MPC in `f1tenth_parallel_sim` (`src/mpc/progress_mpc.py`), with
pybind11 bindings. The controller manager runs it with `mode:=MPC`.

The OCP is solved by acados SQP-RTI (HPIPM, soft track limits) in C, exactly as the Python version does;
the C++ wraps the generated solver and ports the rest of the controller: centerline projection per stage,
warm-start shift and re-simulation, rate-limit anchoring and the steering and speed transport-delay histories.

## Using it

```python
from progress_mpc import ProgressMpc

mpc = ProgressMpc()                          # centerline resampled to 0.1 m
mpc.set_track(centerline_xy, d_left, d_right)  # closed, in driving direction; (n, 2) and (n,)
steer, speed = mpc.step([x, y, yaw, vx, vy, yaw_rate])  # CoG state, map frame, body velocities
mpc.report_applied(steer_applied, speed_applied)        # once per period, what the car actually got
```

`step` takes about 8 ms (horizon 40 x 25 ms) and releases the GIL. The road-wheel angle is not measured, so
`step` takes it from a steering model advanced by `report_applied`; pass `steer=` to override it.
`mpc.stats`, `mpc.plan_world` and `mpc.plan_frenet` expose the last solve and plan.

The state is at the centre of gravity (`mpc.cog_offset` ahead of the rear axle); the controller manager
converts from the EKF's `base_link` odometry.

## Building

Needs acados v0.4.4 (`.install_utils/acados_setup.sh`, or the sim repo's `install/install_acados.sh`),
`ACADOS_SOURCE_DIR` pointing at it, and pybind11. `colcon build --packages-select progress_mpc` builds it; without
acados the package warns and builds nothing, so the rest of the workspace is unaffected.
Plain CMake works too, with `PYTHONPATH=<build>/python` and `LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib`.

## Changing the MPC

The car, horizon, weights and command limits are baked into `generated/` (committed). After changing them in
the sim repo, regenerate it from there:

```bash
export ACADOS_SOURCE_DIR=~/acados LD_LIBRARY_PATH=~/acados/lib:$LD_LIBRARY_PATH
uv run --project ../f1tenth_parallel_sim python progress_mpc/scripts/export_acados.py --sim-repo ../f1tenth_parallel_sim
```

Only the track is a runtime input. `scripts/export_acados.py` writes the acados solver, CasADi's C code for
the model step, and `mpc_config.h` with the sizes and car constants the wrapper uses.

### Command smoothing

The cost penalizes the change of each command per period (`q_steer_rate`, `q_vel_rate`) and its second difference,
`(u_k - 2 u_(k-1) + u_(k-2))^2` (`q_steer_accel` for the steering, `q_vel_accel` for the speed command). The OCP state therefore
holds the last speed command and the command before the last one (steer, speed), three values after the model state and
its histories, and `report_applied` keeps that previous command. The weights are in the sim's `configs/progress_mpc.toml`
(300 and 50; 0 switches the terms off) and are baked into the export; `--q-steer-accel` and `--q-vel-accel` override them
when exporting. In the sim (1200 steps, the exported settings) the second difference of the applied commands fell by 57 %
(steering) and 50 % (speed) with no loss of progress; that was a single run, not a tuned optimum. `configs/ac4mpc.toml` has
the terms at 0.

## Solve time

One SQP-RTI iteration costs ~7.6 ms on a fast laptop, ~26 ms on the car's NUC (the control period is 25 ms). 97 % of it is
the QP inside that iteration, not the model. The export therefore sets `qp_solver_cond_N = 8` (the same QP with partial
condensing, identical commands) and `qp_tol = 1e-4` (HPIPM stops at a looser tolerance). Measured on the sim's track with
the model as the plant over 8 starts x 250 steps: solve 7.6 -> 1.4 ms, 0 solver failures, same path length and no track
violations; the C++ controller solves in ~1.5 ms. Changing these is an export flag (`--qp-cond-n`, `--qp-tol`, 0 keeps the
sim's default); capping the QP iterations or HPIPM `SPEED_ABS` changed the behaviour and are not used. Every variant that
was timed, the profile, the rejected ones and the before/after on the car are in [docs/solve_time.md](docs/solve_time.md);
`scripts/benchmark_solver.py` reproduces them.

## Track clearance and speed cap

The export defaults to `--track-margin 0.10` (clearance 0.384 m around the centre of gravity; the sim config has 0.2, 0.484 m)
and no speed cap of its own (the actuation manager's 5 m/s; `--max-speed 4.0` adds one). On the 1.7 m wide hangar track the
original margin left a 0.7 m corridor and the MPC crashed in 5 of 6 simulated runs (3 of 6 without any delay); with 0.10 m
and the controller's latency compensation it crashed in 0 of 10 at 5 m/s, and 0.15 m crashed in 8 of 10. A horizon of 40
stages is the optimum. Details, the three test rounds
and the harness (`scripts/benchmark_track.py`): [docs/track_clearance.md](docs/track_clearance.md).

## Getting unstuck

The MPC drives forward only and keeps a margin from the walls, so a car that stands still with its heading turned against the
track (after a spin or a hairpin taken too slowly) has no plan that makes progress, and it stays there. The controller can then turn
it around with a multi-point turn in reverse and forward gear (`controller/controller/mpc/recovery.py`, the same file as
`f1tenth_parallel_sim/src/mpc/recovery.py`; a test keeps the two identical): it picks the gear and steering whose arc turns the heading most
toward the track without the car's outline touching a wall, drives it until the walls get close, stops, and repeats until the heading is
within 34 degrees of the track and the car is out of the MPC's keep-out band. It is off on the car by default (`mpc_recovery`, see the
controller README). In the sim (`scripts/run_mpc.py`, on by default, `--no-recovery`), the progress MPC with estimator delays that left it
stuck for the rest of the run (0 laps in 60 s) completes 3 to 5 laps.

## AC4MPC (actor- and critic-guided MPC)

`progress_mpc.Ac4Mpc` is the C++ port of the sim repo's AC4MPC-RTI (`src/mpc/ac4mpc.py`, Reiter et al., "AC4MPC", Alg. 2/3): the
same progress OCP with a trained PPO actor and critic around it. The controller manager runs it with `mode:=AC4MPC`
(see the controller README). Per control step:

- the actor rolls the model out from the measured state (one candidate plan);
- the *active* solver continues from the last chosen plan, with the actor's step appended where the horizon moved on, and the
  *parallel* solver does the same from its own plan, re-seeded with the actor's rollout every `reinit_period` steps;
- the critic enters the solvers' terminal cost as its linearization at the plan's end (`value_weight` is the paper's beta), so
  the QP sees what lies beyond the horizon;
- `ac4eval` scores the three candidates on the model (the OCP's cost with the track-limit slacks, plus the critic at the end)
  and the cheapest one is applied. Its plan seeds the next step.

`mode` picks the parts: `critic` (one solver, the critic's terminal term), `actor` (the three branches, no critic) or `full`.
All of it is C++: the rollout, the track projections, the observation, the candidate scoring and the two acados solvers. The networks run
in libtorch (a TorchScript module, double precision, one thread per branch), the critic's gradient by autograd chained through
the observation's geometry. The three branches run on their own threads, the actor's rollout on the caller's, each with its own
copy of the networks and its own solver, so a step takes about as long as its longest branch.

### Building and exporting

Needs everything the plain MPC does, plus libtorch (the `torch` Python package the RL mode uses, `make build RL=cpu|gpu`) and an
export of the AC4MPC's OCP:

```bash
export ACADOS_SOURCE_DIR=~/acados LD_LIBRARY_PATH=~/acados/lib:$LD_LIBRARY_PATH
uv run --project ../f1tenth_parallel_sim python progress_mpc/scripts/export_ac4mpc.py --sim-repo ../f1tenth_parallel_sim
colcon build --packages-select progress_mpc        # TORCH_ROOT=<torch package dir> if `python3` has no torch
```

`scripts/export_ac4mpc.py` writes `generated_ac4mpc/` (commit it, like `generated/`): the OCP with the critic's terminal term (its
parameters are the stage reference followed by the critic's point and gradient, so one export serves every checkpoint), CasADi's C
code for the model step and the cost pieces `ac4eval` sums, and the car constants. It takes the sim's `configs/ac4mpc.toml`, so the cost weights are the
AC4MPC ones (small steering-rate weight, speed cap = the policy's action range), with this package's track margin and QP settings.
Without libtorch or without the export the package builds as before and `progress_mpc.Ac4Mpc` is `None`.

The policy comes from a PPO checkpoint, trained in the sim repo at the control rate of this export (40 Hz), in `history` or `adapt`
mode (a `privileged` one reads plant parameters the car does not have):

```bash
uv run python scripts/export_policy.py --checkpoint checkpoints/ppo_f1tenth_ac4mpc.pt --out policy.ts   # in the sim repo
```

```python
from progress_mpc import Ac4Mpc
mpc = Ac4Mpc("policy.ts", mode="full")          # value_weight, reinit_period, eval_correction, eval_rollout, threads, torch_threads
mpc.set_track(centerline_xy, d_left, d_right)   # as ProgressMpc
steer, speed = mpc.step([x, y, yaw, vx, vy, yaw_rate])
mpc.report_applied(steer_applied, speed_applied)
mpc.stats["candidate"], mpc.candidate_wins      # which branch won this step / how often each did
```

The car, the horizon and the weights are baked into the export, so the policy must have been trained for the same car (its
action range, latencies and body): the export and the checkpoint are not checked against each other beyond the control period.

### Timing and agreement with the sim

Measured on a 16-core laptop with a randomly initialized network of the training size (50 waypoints, history 40, 256-wide
layers, conv encoders), per control step over 60 steps of the closed loop:

| mode | threads | step | longest branch |
| --- | --- | --- | --- |
| critic | one solver | 1.9 ms | solver 1.8 ms |
| actor | 3 | 2.9 ms | solvers 2.2 / 2.7 ms, actor rollout 1.0 ms |
| full | 3 | 3.3 ms | solvers 2.6 / 3.2 ms, actor rollout 1.1 ms |
| full | none | 5.0 ms | |

The actor's 40 network evaluations cost 1 ms; the solvers dominate. The NUC is several times slower than the laptop for the
solver (the plain MPC's solve is 1.4 ms here and ~6 ms on the car), so expect roughly 10 to 15 ms in `full` mode, inside the 25 ms
period; measure it with `/controller/diagnostics` (`solve_ms_*`, and `wins_*` for the branches).

`test/test_ac4mpc.py` runs the sim's Python AC4MPC and this one on the same checkpoint, track and measurements: the actor, the
critic's value and gradient and the centerline projection agree to 1e-16, and the commands and the winning branch agree step by step
(about 1e-15 at first; two RTI runs drift apart in the closed loop as any two do, so compare outcomes over many steps). The threaded and the single-threaded run
give identical commands.

## Tests

`pytest progress_mpc/test` runs the closed loop on a synthetic track with the MPC's own model as the plant.
Against the Python controller on the sim's track and with the points unresampled (`ProgressMpc(0.0)`), checked
after the sim gained a speed-command delay (history of 2 steering + 1 speed command):

- the generated solver is identical to a fresh export from the sim, and one model step agrees exactly (max difference 0.0
  over random states and commands);
- the first steps agree to ~1e-7 in the plan and the command. The seed is the Python side's float32 track (arc length
  and track length differ by ~1e-7 from the same points in double precision);
- a closed loop then diverges, as it does for the Python controller against itself: with its state nudged by 1e-7 it
  ends up 0.1 to 2.5 m of path away after 300 steps. Over five start points the C++ port's path length, mean speed and
  track violation (none) fall inside that spread, so compare outcomes rather than command traces.
