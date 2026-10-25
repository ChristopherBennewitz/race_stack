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
and `--max-speed 4.0`. On the 1.7 m wide hangar track the original margin left a 0.7 m corridor and the MPC crashed in 5 of
6 simulated runs (3 of 6 without any delay); with 0.10 m it crashed in 1 of 10 (0 of 10 with the cap or the controller's
latency compensation), and 0.15 m crashed in 8 of 10. A horizon of 40 stages is the optimum. Details, the three test rounds
and the harness (`scripts/benchmark_track.py`): [docs/track_clearance.md](docs/track_clearance.md).

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
