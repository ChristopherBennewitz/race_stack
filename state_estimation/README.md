# State Estimation
![State Estimation Architecture](./misc/state_estimation_arch.png)

This directory contains the all required files to launch the state estimation submodule, including Google's *cartographer* localization framework, the Extended Kalan Filter for velocity estimation. This includes the relevant launchers, configuration files, and scripts. 

The main launcher for the state estimation components can be found in `state_estimation/launch/state_estimation_launch.xml`. 

The resulting car state is exposed as:
- TF `map -> base_link` (and the alias `car_state/base_link`): the car pose from localization.
- `/car_state/pose`: the same pose as `PoseStamped`, published by `carstate_node` on the physical car (the simulator publishes it directly).
- `/state_estimation/odom`: the local EKF's estimate in the `odom` frame (twist in the `base_link` frame). It never sees SLAM; Cartographer starts from it.
- `/state_estimation/odom_map`: the estimate for the controllers, in the `map` frame, smooth where `map -> odom` jumps (`odom_topic:=/state_estimation/odom_map`). It is published by `map_odom_node`: the local odometry moved into the map frame by a slowly corrected `map -> odom`, so position, heading and the whole twist are as smooth as the local filter's. Turn it off with `global_ekf:=False`.
- `/car_state/frenet/odom`: pose and velocity in the Frenet frame, published by the `frenet_odom_republisher` node (`utilities/nodes/frenet_odom_republisher`), which combines the TF pose with `/state_estimation/odom`. Launched for both simulation and the physical car by `stack_master/launch/base_system_launch.xml`.

## Kalman Filter ##

The standard launch now runs `state_estimation/ekf_node`: a `filterpy`
`ExtendedKalmanFilter` with sequential IMU and VESC odometry updates. Its
prediction is a CasADi-symbolic mirror of the blended, Pacejka single-track
model in `f1tenth_parallel_sim`; it estimates
`[x, y, yaw, vx, vy, yaw_rate, steer]` and takes Ackermann steering/speed
commands as inputs. The transition, its Jacobian, and both measurement
Jacobians are all derived from one CasADi graph (`ca.jacobian`), so the
Jacobians used by the filter can never drift out of sync with the model that
is actually stepped forward. VIO is intentionally not integrated. The nominal
vehicle and tire values are ROS parameters, defaulting to the simulator's
`f1tenth_car.toml`; override them per car in a launch file rather than relying
on an external configuration package.

`casadi` and `filterpy` are already part of the repo-wide Python requirements
(`.install_utils/python_req.txt`), so no extra install step is needed beyond
the usual environment setup.

`ekf_launch.py` launches the new filter and continues to publish
`/early_fusion/odom`, so existing Cartographer and car-state launch wiring does
not change. The old `robot_localization` configuration remains in `config/ekf.yaml`
solely to support controlled comparisons.

For the simulator-equivalence tests and shared simulator utilities, install the
CPU-only PyTorch build (the deployed RaceStack never requires CUDA):

```
python3 -m pip install -r .install_utils/torch_cpu_req.txt
```

### Tuning the local filter's noise from bags

`misc/ekf_tune.py` fits the local filter's Q, the IMU R and the VESC speed / yaw-rate R to recorded bags (the sysid
takes in `sysid_bags/` have everything it needs: `/odom`, `/sensors/imu/raw`, `/ackermann_cmd_applied`,
`/car_state/pose`). It minimizes the negative log-likelihood of the filter's own innovations with CMA-ES, replaying
every candidate over the training bags in parallel, so it needs no ground truth. Bags held out of the search check
the result (NLL, NIS against its degrees of freedom, velocity error against the SLAM-derived reference):

```
f1tenth_parallel_sim/.venv/bin/python state_estimation/misc/ekf_tune.py sysid_bags/sysid_* \
    --holdout figure_8 free_drive_3 --out tuned.json
```

It prints the new `defaults.NOISE` entries to paste into `state_estimation/ekf_node/defaults.py`, which is what the
node and the replay both start from. `--evaluate-only` reports the current defaults, `--fit-bias` adds the
accelerometer bias. The sysid bags' SLAM pose came from the live estimator's odometry, so the velocity reference is
not independent of it. Parameters that end up on the search bound are not determined by the data.

### An independent SLAM pose

The takes' own `/car_state/pose` was computed from the live estimator's odometry, so it is not independent of the filter.
`misc/offline_slam_poses.py` re-runs Cartographer on a take's `/scan` (and `/tf_static`) alone, with no odometry and no IMU,
inside the race_stack container. Its poses are the reference for `ekf_replay.py --reference-dir` and the input of
`ekf_correction_sweep.py --pose-dir`:

```
python3 state_estimation/misc/offline_slam_poses.py sysid_bags/sysid_* --out-dir sysid_poses       # add --map X.pbstream to localize
```

`--dry-run` prints the Cartographer configuration and the three commands. It is untested against a live
`cartographer_ros` here, so treat the first run as a smoke test (it prints the pose count, duration and path length;
the path length should match the take's driven distance). See the docstring for what the resulting pose does and does
not represent.

### The smooth map-frame output

`map_odom_node` (`ekf_node/map_odom_correction.py`, no ROS) keeps a rigid transform `map <- odom` and its rate, follows
the SLAM pose with a critically damped alpha-beta filter (`tau`, bandwidth 1/tau) and a further short lag on the output
(`smooth`), and publishes the local odometry through it. Parameters are in `defaults.CORRECTION`;
`misc/ekf_correction_sweep.py` replays the local filter on the bags and reports the corrected output's jitter, distance from the SLAM pose, yaw
smoothness and velocity accuracy, also over `tau` and `smooth`. On the six takes with independent poses: position jitter
32 m/s^2 against the local filter's 16, at 5 cm from the SLAM pose (a global EKF that fused the pose had 88 at the same distance, and was removed). Details and the trade-offs: `docs/ekf_formulation.md`, section 5d.

### Observability

All estimators publish `diagnostic_msgs/DiagnosticArray` on `/state_estimation/diagnostics` (PlotJuggler plots the numeric
values; `ros2 topic echo` shows the messages), and the MPC on `/controller/diagnostics`.

| status name | rate | what to look at |
|---|---|---|
| `ekf_local/<sensor>` | 1 Hz | `nis_mean` against `nis_expected`: the filter's live calibration, the same statistic `ekf_tune.py` optimises. `nis_ratio` > 2 means over-confident (noise too small), < 0.1 far too loose; the status turns WARN. The NIS is only counted while the car is moving (above 0.3 m/s or 0.2 rad/s): a parked car has no vibration, so its innovations are all noise floor and read "too loose" whatever the settings. Optional inputs that never deliver (the VIO) are reported as not used. `age_s`, `rate_hz`, `stamp_lag_ms`: arrival of the input. |
| `ekf_local` | 1 Hz | `std_*` (uncertainty), `predict_*`/`update_*` times, `loop_dt_max_ms` (WARN above 3 periods), `skipped_ticks`. |
| `map_odom` | 2 Hz | `tx, ty, theta` the correction and `rate_*` its drift (the local odometry's drift per second); `innovation_m` how far the last SLAM pose was from the output (about 5 cm normally; WARN above 0.3 m); `updated/gated/jumped/skipped` pose outcomes; `pose_age_s` (WARN above 0.5 s, ERROR above 2 s). The pose covariance of `/state_estimation/odom_map` is the correction's alone (`std`, growing without a SLAM pose) plus the local heading variance: the local filter's position covariance grows without bound and says nothing about where the car is on the map. |
| `mpc` (`/controller/diagnostics`, 10 Hz) | 10 Hz | `solve_ms_*` against `control_dt_ms`; `iterations_*`, `failures`; `slack_max` (track-limit violation in the plan, m); `prediction_error_*` (how far the car was from where the plan put it one period earlier: model mismatch or a command the car did not follow); `lateral_error`; `odom_age_max_ms`; `resets`, `jump_resets`. |

`/state_estimation/odom_map_jumps` (`std_msgs/UInt32`, latched) increments whenever the map-frame output stepped because
the correction was set from a SLAM pose after it disagreed repeatedly. The controller drops the MPC plan on a change.

### Comparing the old and new filters without motion capture

Record one representative bag containing `/odom`, `/sensors/imu/raw`,
and `/ackermann_cmd_applied`, then replay it with:

```
ros2 launch state_estimation ekf_compare_launch.py
```

This runs both filters against exactly the same data and prints a report every
10 seconds. It is deliberately a *consistency* benchmark, not a claim of
absolute pose accuracy:

- `held_out_imu_acceleration_residual` compares each output's finite-difference
  body acceleration with the IMU. The default new filter does not correct on
  acceleration, making this the most useful quasi-independent metric.
- `imu_yaw_rate_residual` and `vesc_twist_residual` check agreement between
  independent inertial and drivetrain channels.
- `vesc_twist_nis` tests whether each filter's reported covariance matches its
  observed odometry discrepancy (a calibrated NIS should be near its 3-D
  degrees of freedom on average).
- `velocity_jerk` catches a numerically good-looking estimate that is too noisy
  to be useful to the controller.

Use multiple runs with straight-line acceleration, steady circles, braking, and
slaloms. Tune only on a subset, then compare the held-out maneuvers. Because
the outputs and sensors share information, these scores cannot prove global
position accuracy; they are a practical way to choose a model and tune noise
without an external reference.

### Comparing the old and new filters via SLAM scan-matching agreement

`cartographer_ros` does not expose its internal scan-matcher cost/residual
over ROS, so there is no direct "matching difficulty" number to read off.
`slam_compare_launch.py` instead runs both EKFs (via `ekf_compare_launch.py`)
alongside two independent, unmodified `f110_2d.lua` mapping-mode Cartographer
instances -- one per candidate odometry, each in its own namespace so they
don't collide on topics or the transform tree -- and `slam_match_benchmark_node`
compares each instance's own scan-matched trajectory (`tracked_pose`) against
its candidate's odometry:

```
ros2 launch state_estimation slam_compare_launch.py racecar_version:=NUC6 scan_topic:=/scan
```

- `cartographer_speed_residual` / `cartographer_yaw_rate_residual`: how much
  Cartographer's own scan-matched motion disagrees with what the candidate
  odometry predicted. A small, steady gap means the odometry was a good prior
  for the scan matcher's search window -- i.e. matching was easier. A large
  one means the matcher had to substantially override the prior.
- `jump_events`: count of tracked_pose steps implying a speed or yaw rate far
  outside the car's physical envelope -- a proxy for the matcher losing track
  and re-converging elsewhere, excluded from the residual metrics above.

This has not been run end-to-end against real `cartographer_ros`/hardware in
this environment; treat the first run as a supervised smoke test.

### Parameters

The parameters associated with the EKF can be modified in the configuration file located in `state_estimation/config/ekf.yaml`.

The parameters relevant to the EKF, responsible for generating the velocity estimates, can in this case be subdivided into two categories: the covariances $\sigma_{i}^2 \in [0, \infty)$ associated with observations (zero elements are handled by the *robot_localization* package) and the configuration parameters which dictate what sources are to be considered by the EKF. 
    
The `robot_localization` sensor fusion EKF configuration parameters are defined according to the associated publication as:
    
$$ 
\verb|config| =
\begin{bmatrix}
    x & y & z \\
    roll & pitch & yaw \\
    vx & vy & vz \\
    vax & vay & vaz \\
    alx & aly & alz \\
\end{bmatrix} 
$$

where each entry is of type boolean, indicating whether or not the associated data source should be considered by the filter.

The EKF parameters intended for tuning are summarized as follows:

- $[\sigma_{VESC,x}, \sigma_{VESC,y}, \sigma_{VESC,\psi}] = [0.25, 0.5, 0.4]$ - the covariances associated with VESC control odometry x position, y position, and yaw angle, respectively.
- $[\sigma_{VESC,vx}, \sigma_{VESC,vy}, \sigma_{VESC,vaz}] = [0.02, 0.05, 0.0]$ - the covariances associated with VESC control odometry longitudinal, lateral, and angular yaw velocities, respectively.
- $[\sigma_{IMU,al}, \sigma_{IMU,va}, \sigma_{IMU,q}] = [0.0,0.0,0.0]$ - the covariances associated with IMU linear acceleration, angular velocity, and orientation measurements, respectively.
- The EKF configuration matrix for the VESC control odometry is defined as:
$$
\verb|config_odom| = 
\begin{bmatrix}
    \verb|false| & \verb|false| & \verb|false| \\
    \verb|false| & \verb|false| & \verb|false| \\
    \verb|true| & \verb|true| & \verb|false| \\
    \verb|false| & \verb|false| & \verb|false| \\
    \verb|false| & \verb|false| & \verb|false| \\
\end{bmatrix}
$$
- The EKF configuration matrix for the IMU is defined as:
$$
\verb|config_imu| = 
\begin{bmatrix}
    \verb|false| & \verb|false| & \verb|false| \\
    \verb|false| & \verb|false| & \verb|true| \\
    \verb|false| & \verb|false| & \verb|false| \\
    \verb|false| & \verb|false| & \verb|true| \\
    \verb|false| & \verb|false| & \verb|false| \\
\end{bmatrix}
$$

## EKF / UKF Velocity Estimation Nodes ##

In addition to the `robot_localization`-based EKF above, the package ships its own `ekf_node` and `ukf_node` executables (Extended/Unscented Kalman Filter, using [FilterPy](https://github.com/rlabbe/filterpy)). They fuse IMU and VESC odometry (with Ackermann drive commands as control input for some models) into a local pose/odometry estimate. Sensors can be de-weighted by inflating their covariance instead of removing them, keeping the node structure generic.

Motion models are implemented as interchangeable classes with a common interface:
- `state_estimation/ekf_node/models.py` — point-mass, kinematic bicycle, and single-track models
- `state_estimation/ukf_node/models.py` — single-track model

Each node is launched standalone via its own launch file, both part of this package:
```
ros2 launch state_estimation ekf_node_launch.py
ros2 launch state_estimation ukf_node_launch.py
```

## Localization ##

With the ROS2 race-stack, you can currently only use Google's *cartographer* localization framework outlined [here](https://google-cartographer-ros.readthedocs.io/en/latest/) [see our paper for reference]. 

The localization package is launched via the aformentioned main state estimation launch file located under `state_estimation/launch/state_estimation_launch.xml`. 

### Parameters

The parameters associated with SLAM localization can be modified in the configuration files located under `stack_master/config/[racecar_version]/slam/`, where `[racecar_version]` is the subfolder corresponding to the vehicle in question. When mapping, edit the `f110_2d.lua` (for Mapping) or `f110_2d_loc.lua` (for Localization) file.

Based on our trust on Odometry values:
- `TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight`
- `TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight`

These weight the cost of rotating/translating incoming scans (from the odometry reference). If we trust our odometry, we can weight these values a little higher, but if not (our IMU and `vesc/odom` values aren't great at the moment) the default values are `0.2`.

- `POSE_GRAPH.optimization_problem.odometry_rotation_weight`
- `POSE_GRAPH.optimization_problem.odometry_translation_weight`

These weight how much the pose graph weights odometry (over scan matching). Similarly to above, but when performing localization of the car. Default values are `0`.

Based on our trust of LIDAR scans:
- `TRAJECTORY_BUILDER_2D.submaps.range_data_inserter.probability_grid_range_data_inserter.hit_probability`
`TRAJECTORY_BUILDER_2D.submaps.range_data_inserter.probability_grid_range_data_inserter.miss_probability`

These weight the probability of a hit or miss in the occupancy grid.
- Hit probability: The trust in cells where we get range data
- Miss probability: The trust in cells where we observe free space

If we don't trust the lidar (eg. black tubes), we should reduce the miss probability (free space may mean undetected things).
