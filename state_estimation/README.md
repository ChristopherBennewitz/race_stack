# State Estimation
![State Estimation Architecture](./misc/state_estimation_arch.png)

This directory contains the all required files to launch the state estimation submodule, including Google's *cartographer* localization framework, the Extended Kalan Filter for velocity estimation, and the car state aggregator (*carstate_node*). This includes the relevant launchers, configuration files, and scripts. 

The main launcher for the state estimation components can be found in `state_estimation/launch/state_estimation_launch.xml`. 

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
python3 -m pip install -r requirements-cpu.txt
```

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
