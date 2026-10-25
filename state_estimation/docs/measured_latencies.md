# Measured latencies of the estimation and control chain (2026-10-25)

Measured on NUC2 with the simulator-model EKF as the odometry prior, `map_odom_node` for the map-frame output and the C++
MPC driving closed loop (bags `mpc_run2`, 34 s, and `mpc_run3`, 110 s, with 28 s of continuous driving). Tool:
`f1tenth_parallel_sim/scripts/measure_latency.py <bag> --verbose`, which fits every stage as
`y(t) = k * lag_tau(u(t - delay))` on the bag's own signals (no hardware timestamps). `d+lag` is the well-determined part.

## Estimator stages (software only; what an observation delay in the sim is built from)

| stage | run 2 | run 3 | old chain (2026-10-18 teleop, 6 bags) |
|---|---|---|---|
| gyro to EKF yaw rate | 14.4 +- 0.7 ms | 12.0 +- 3.1 ms | 12.0 ms |
| gyro to EKF d(yaw)/dt | 15.4 +- 2.1 ms | 14.3 +- 1.2 ms | 11.3 ms |
| VESC speed to EKF v | 13.5 +- 46.7 ms | 8.1 +- 12.1 ms | 39.2 ms |
| VESC speed to EKF d(xy)/dt | 13.9 +- 26.7 ms | 6.5 +- 9.3 ms | 43.4 ms |
| VESC speed to SLAM window speed (mid-stamp) | 28.6 ms | 2.0 +- 3.1 ms | 34.1 ms |

The new filter's position and speed lag far less than the old chain's (the spreads are large, but both bags agree they are
well below the old 39 to 43 ms). The SLAM pose stamps lag the car by 2 +- 3 ms on the long run 3 and 29 +- 24 ms on the short
run 2 (34 ms on the old chain); run 2 is too short to pin it down, and the model-based scan below (run 3) puts it at 0 to
10 ms. That is plausible: the live pose is `map -> odom` composed with the local filter's pose at the stamp, and the
odometry part is exact there.

Used in the sim: `configs/train_*.toml` `[perturb] delay_*`: each stage's mean over the two bags plus the ~5 ms the
controller tick waits for the EKF's next 100 Hz message: position 15 ms, velocity 16 ms, yaw 20 ms, yaw rate 18 ms (were
48, 44, 16, 17). `delay_jitter` (5 ms) and `delay_extra_max` (30 ms) are unchanged, so a policy still trains across roughly
15 to 50 ms of position and velocity delay.

## Controller side (not part of the plant, so not in the sim's actuator delays)

| | mean | 95th percentile |
|---|---|---|
| estimator age when `/drive` is published | 7.0 ms | 15.1 ms |
| `/drive` to applied command (actuation manager, 100 Hz) | 5.1 ms | 9.9 ms |
| MPC solve (references + solver), after the QP change | 6.3 ms | 8 ms |
| `/car_state/pose` stamp age | 8.2 ms | 23.7 ms |

State to applied command is about 18 ms on average. The MPC does not compensate for it; an RL policy sees it as the
observation delay above. See `progress_mpc/docs/solve_time.md` for how the solve time went from 26 ms to 6 ms.

## Plant (command to car)

| stage | MPC runs (run 2, run 3) | teleop takes the car parameters were fitted on |
|---|---|---|
| steer command to gyro, delay + lag | 127.4 and 121.4 ms (R2 0.82 to 0.84) | 161.4 ms (R2 0.90) |
| speed command to VESC speed | 147.9 and 175.1 ms | 168.4 ms |

The fitted actuator parameters are `steer_delay` 45.8 ms, `steer_latency` 44.1 ms, `speed_delay` 10.2 ms; the steering
response above also includes the yaw dynamics, so it is longer than the actuator parameters alone.

## Does the sim model predict the car? (open loop, restarted every 0.5 s from the measured state)

| horizon | MPC bag: position / yaw / speed / yaw rate | teleop takes (in-sample reference) |
|---|---|---|
| 0.25 s | 0.046 m / 3.3 deg / 0.126 m/s / 0.289 rad/s | 0.040 m / 2.3 deg / 0.170 m/s / 0.346 rad/s |
| 0.5 s | 0.087 m / 5.2 deg / 0.157 m/s / 0.310 rad/s | 0.066 m / 3.8 deg / 0.215 m/s / 0.343 rad/s |
| 1.0 s | 0.189 m / 6.9 deg / 0.161 m/s / 0.324 rad/s | 0.121 m / 5.5 deg / 0.241 m/s / 0.337 rad/s |

Speed and yaw rate are predicted better on the MPC data than on the fitted data; position and yaw are 1.2 to 1.6 times worse
(higher speeds, tighter turns). No large mismatch. A scan of `steer_delay`, `steer_latency` and `speed_delay` on both data
sets does not support a change: the gyro puts the best steering latency at 0.044 s, and so do the pose channels once the
pose stamps are shifted correctly. The best steering latency does not shrink with speed in the teleop data either.

## The pose-stamp shift of the sysid extractor

`sysid.rosbag.SLAM_POSE_DELAY = 0.045` is right for the old chain (34 to 44 ms measured) and wrong for this one. Model error
on the MPC bag against the shift (position / 0.08 m + yaw / 4 deg, lower is better, current model): 1.91 at -20 ms, 1.92 at
0 ms, 1.98 at +10 ms, 2.08 at +20 ms, 2.49 at +45 ms. With 45 ms the position and yaw channels wanted 0.005 s of steering
latency while the gyro wanted 0.044 s; at 0 ms they agreed. `extract_bags.py` now takes `--pose-delay` (default unchanged)
and `--glitch-scale`; use `--pose-delay 0.005 --glitch-scale 2.5 --max-gap 0.2` for bags like these.

## Reproducing

```bash
cd f1tenth_parallel_sim
uv run python scripts/measure_latency.py ../race_stack/mpc_run3 --verbose
uv run python scripts/extract_bags.py ../race_stack/mpc_run3 --out /tmp/csv --control-hz 100 \
    --pose-delay 0.005 --glitch-scale 2.5 --max-gap 0.2 --min-seconds 2
```

The bags need `/commands/servo/position` and `/commands/motor/speed` besides the usual topics. Recording command:

```bash
ros2 bag record -o mpc_runN \
  /tf /tf_static /sensors/imu/raw /imu /odom /ackermann_cmd_applied /drive \
  /commands/servo/position /commands/motor/speed \
  /state_estimation/odom /state_estimation/odom_map /car_state/pose \
  /state_estimation/diagnostics /controller/diagnostics /controller/mpc/markers
```

Stand still for at least 1 s at the start (the speed offset is calibrated from it).
