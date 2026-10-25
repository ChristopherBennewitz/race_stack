# Why the MPC crashed on the hangar track, and what fixed it (2026-10-25)

## What the car showed

Two drives on `hangar_25_10_v` (a 25.6 m track, 1.70 m wide on average, 1.10 m at its narrowest) with the loop already at
its real rate (40 Hz, 6 ms solves): the car completed a lap at up to 4.6 m/s and then went wide into a wall on a tight left
corner (radius about 1.1 m). In the moving part of the second bag the plan used more than 5 cm of track-limit slack in
27 % of the 0.1 s windows (more than 20 cm in 14 %, up to 0.88 m), the steering sat at its +-0.3 rad lock a good part of
the time, and the lateral error had an rms of 0.26 m.

The sim's MPC keeps the centre of gravity `collision radius + track_margin = 0.284 + 0.2 = 0.484 m` inside the track
limits (a margin tuned on wider sim tracks). On this track that leaves a corridor of **0.73 m on average, 0.13 m in the
narrowest places**, for a 0.30 m wide car doing 3 to 4 m/s.

## The experiment

`scripts/benchmark_track.py` runs the MPC in closed loop on the track in the sim, with what the car has:

- plant: the sim's vehicle model (fitted car parameters of 2026-10-24) at 400 Hz, commands through the sim's actuation
  manager (100 Hz tick with a random phase, the car's rate limits), 40 Hz controller;
- the controller sees the state delayed per channel by the new EKF's measured lag (position 15 ms, yaw 20, speed 16, yaw
  rate 18) plus a message age uniform in 0..10 ms, and its command is available 7.5 ms after the controller tick (the solve);
- a run ends at the first contact of the car's rectangle (0.568 x 0.296 m, centred on the centre of gravity) with a track
  limit; "crashes" below are runs that ended that way within the time limit.

The sim reproduces the car: with the original settings it used more than 5 cm of slack in 27 % of the steps (the car:
27 %), the lateral error rms was 0.27 m (0.26 m) and the steering sat at lock 40 % of the time.

## Results

**Round 1** (6 starts x 40 s; the sim's default QP settings):

| variant | crashes | time until crash or end | progress | v mean / max | slack > 5 cm | lateral rms |
|---|---|---|---|---|---|---|
| ideal, no delays at all | 3/6 | 30.6 s | 105 m | 3.36 / 4.98 | 10 % | 0.22 m |
| **base (original settings)** | **5/6** | 25.6 s | 83 m | 3.26 / 4.98 | 19 % | 0.24 m |
| base + latency compensation | 5/6 | 26.2 s | 91 m | 3.39 / 4.96 | 9 % | 0.20 m |
| **margin 0.07** | **0/6** | 40.0 s | 138 m | 3.38 / 4.97 | 3 % | 0.19 m |
| speed cap 3.0 | 0/6 | 40.0 s | 120 m | 2.70 / 3.00 | 2 % | 0.24 m |
| margin 0.07 + cap 3.5 | 0/6 | 40.0 s | 134 m | 2.92 / 3.50 | 2 % | 0.31 m |
| horizon 30 | 6/6 | 12.8 s | 41 m | 3.14 / 4.99 | 29 % | 0.27 m |
| margin 0.07 + horizon 30 | 6/6 | 7.1 s | 23 m | 3.18 / 4.98 | 19 % | 0.31 m |

**Round 2** (10 starts x 60 s; "QP" = the exported settings `cond_N` 8, `qp_tol` 1e-4):

| variant | crashes | progress | slack > 5 cm | lateral rms | solver failures |
|---|---|---|---|---|---|
| margin 0.07 (sim default QP) | 1/10 | 195 m | 3 % | 0.19 m | 22 |
| margin 0.07 + QP | 1/10 | 191 m | 3 % | 0.19 m | 1 |
| **margin 0.10 + QP** | **1/10** | **197 m** | 3 % | 0.18 m | 0 |
| margin 0.15 + QP | **8/10** | 89 m | 14 % | 0.23 m | 0 |
| margin 0.10 + horizon 50 + QP | 8/10 | 85 m | 13 % | 0.29 m | 0 |
| margin 0.10 + horizon 60 + QP | 5/10 | 132 m | 8 % | 0.29 m | 2 |
| margin 0.10 + speed cap 4.0 + QP | **0/10** | 211 m | 2 % | 0.25 m | 0 |
| margin 0.10 + latency compensation + QP | **0/10** | **219 m** | 2 % | 0.20 m | 0 |

**Round 3** (10 starts x 60 s, the deployed C++ controller: margin 0.10, speed cap 4.0, exported QP; the compensation is
the controller's own `extrapolate_state` with its default 12 ms + measured age + 7.5 ms):

| variant | crashes | progress | slack > 5 cm | solver failures |
|---|---|---|---|---|
| C++, compensation on | 0/10 | 216 m | 2 % | 1 |
| C++, compensation off | 0/10 | 210 m | 2 % | 0 |
| Python reference, same settings | 0/10 | 217 m | 2 % | 0 |
| C++, no delays at all | 0/10 | 216 m | 2 % | 0 |

Raw per-start results: `track_benchmark_round{1,2,3}.json`.

## What the results say

- **The clearance was the problem, not the delays.** With no delays at all the original settings still crashed in 3 of 6
  runs. The car's delays made it worse (5/6, 83 m instead of 105 m), but removing them did not fix it.
- **The margin has a sharp edge.** 0.10 m (clearance 0.384 m) drives cleanly; 0.15 m (0.434 m) crashes in 8 of 10 runs,
  because the narrowest parts of the track leave almost no corridor. With 0.07 and 0.10 m the results are the same within
  noise. 0.10 was chosen: the rectangle's corners are 0.32 m from the centre of gravity, so the worst-case clearance to a
  wall is 0.384 - 0.32 = 6.4 cm (0.354 - 0.32 = 3.4 cm with 0.07), and 23.6 cm when the car is aligned with the track.
  The car's localization error (about 5 cm) comes on top, so do not go lower without a reason.
- **40 stages is the right horizon.** 30 stages crash in every run, 50 and 60 crash more than 40. The solver's weights were
  tuned at 40; the horizon length is not what was wrong.
- **With the right clearance the delays hardly matter** (216 m with them, 216 m without), and latency compensation adds
  about 3 % progress and removed the last crash in the 10-start test. A speed cap of 4 m/s also removed it, at a small
  cost in progress. Both are cheap safety margins for the first drives.
- **The looser QP settings do not change the driving**, and the solver failed less often with them (1 against 22).

## What changed

- `scripts/export_acados.py` defaults to `--track-margin 0.10` and `--max-speed 4.0` (the sim config has 0.2 and the
  actuation manager's 5 m/s). `--track-margin -1` / `--max-speed 0` restore those. `mpc_config.h` records
  `kTrackMargin`; `kClearanceMargin` (0.384) is what the RViz envelope draws.
- The controller extrapolates the state by the estimator's lag (12 ms), the message's measured age and the compute time
  (7.5 ms) before each solve: parameters `mpc_latency_comp`, `mpc_state_lag_s`, `mpc_compute_lag_s`.

## Caveats

- This is the sim: the plant is the same model the MPC uses, so model mismatch is not tested here (the model check against
  the car is in `state_estimation/docs/measured_latencies.md`: comparable to its fit on the fitted data). There is no
  estimator noise, only delays and the message age.
- The crash check is a rectangle around the centre of gravity, an approximation of the body.
- Ten starts of 60 s: a difference of one crash (1/10 against 0/10) is not significant; 8/10 against 1/10 is.
- The numbers are for the hangar track. A wider track tolerates a larger margin; a narrower one would need a smaller one.
- First drives on the car should still keep the speed low and the deadman ready: the sim cannot show what the car's
  real localization error does at the walls.

## Reproducing

```bash
# the track must be in the sim's configs/maps/<name>/ (git-ignored data) without the repeated closing point
uv run --project ../f1tenth_parallel_sim python progress_mpc/scripts/benchmark_track.py --round 2
PYTHONPATH=<build>/python uv run --project ../f1tenth_parallel_sim python progress_mpc/scripts/benchmark_track.py --round 3
```
