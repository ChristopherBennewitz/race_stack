"""Nominal values of the simulator-model EKF, shared by the node (as ROS parameter defaults) and the offline tools.

Names are the node's ``sim_*`` parameters without the prefix. ROS-free on purpose, so the replay and the tuner
can use exactly what the node runs with.
"""
from .models import VehicleModelParams

# Actuator model, fitted by the simulator's sysid (runs/sysid/fit/20261024-110517): the steering command is
# delayed by steer_delay, then followed through one lag of steer_latency, and the speed command is delayed by
# speed_delay before the speed loop. The rest are the nominal values of
# f1tenth_parallel_sim/configs/f1tenth_car.toml.
MODEL = {
    'steer_gain': 1.43527, 'steer_bias': 4.20711e-05,
    'steer_delay': 0.0458081, 'steer_latency': 0.0441209,
    'speed_delay': 0.0101888,
    'drive_gain': 9.18827, 'brake_gain': 6.51316,
    'front_drive_frac': 0.5, 'constraint_time_constant': 0.03,
    'blend_center': 0.6, 'slip_speed_regularizer': 0.1,
    'mass': 3.718, 'yaw_inertia': 0.0561,
    'front_axle_distance': 0.17334, 'rear_axle_distance': 0.14766,
    'cg_height': 0.074,
    'front_B': 59.9965, 'front_C': 1.00053, 'front_D': 0.52523, 'front_E': 0.0,
    'rear_B': 49.5247, 'rear_C': 1.00001, 'rear_D': 0.624743, 'rear_E': 0.0,
    'imu_x': 0.07,
}

# Process and measurement noise of the LOCAL filter (the one that gives the localization its odometry prior), fitted by
# misc/ekf_tune.py on the six sysid takes of 2026-10-18 (maximum likelihood of the IMU and VESC innovations, held-out
# takes checked; variant "IMU yaw terms fixed"). `R_vesc[vx]`, `R_imu[yaw]` and `R_imu[yaw_rate]` are held at their earlier
# values: left free, the fit made the filter copy the wheel speed and the IMU heading, which raised the likelihood
# and left the velocity error against the SLAM reference no better. Q[x] and Q[y] are not determined by the data (the
# position has no measurement) and keep their earlier values. The covariances went from 2-4x over-confident
# (NIS 8-17 against 4-6 expected) to calibrated; velocity accuracy is unchanged.
#
# VESC pose is not fused (huge R): its x/y/yaw are dead-reckoned in a frame with an arbitrary yaw origin, which would fight
# the IMU heading. The IMU accelerations are fused so the speed is not carried by the command model alone. The accelerometer
# follows the car through a first-order lag of imu_accel_lag (the sensor's low-pass), so it is fused raw
# (no EMA, which would add lag) against the predicted acceleration passed through that lag. The lag is fitted on the
# long drives of data/sysid_20261011 against d(vesc speed)/dt and v * yaw_rate at the IMU (sysid/latency.py);
# imu_accel_bias is the constant offset those fits leave (mounting tilt).
#
# Refit on 2026-10-25 for the actuator parameters of MODEL above (the sim's re-identified car: steer and speed delays, speed
# loop gains, tire curves), same data and same fixed terms, plus Q[x], Q[y] held: three seeds agree. Held-out likelihood
# 1.782 -> 1.753 per sample, velocity error unchanged within the reference's noise (vx 0.203 -> 0.211, wz 0.167 -> 0.165).
NOISE = {
    'R_imu': [0.1, 0.0009, 1.392, 2.136],  # yaw, yaw rate, ax, ay
    'imu_accel_bias': [0.05, 0.56],
    'imu_accel_lag': 0.05,
    'R_vesc': [1000000.0, 1000000.0, 1000000.0, 0.0225, 100.0, 0.8382],  # x, y, yaw, vx, vy, yaw rate
    'Q': [4.9e-5, 3.6e-5, 0.004351, 0.08596, 0.01918, 0.4135, 0.1037],  # per second, on [x, y, yaw, vx, vy, yaw rate, steer]
    'predict_substep': 0.005,
}

# The GLOBAL filter (the one that adds the SLAM pose and feeds the controllers). Its dynamics noise is the local fit
# (the two fits agree: ekf_tune.py --global gave Q[vy], Q[yaw_rate], Q[steer], R_imu[ax], R_imu[ay] and R_vesc[yaw_rate]
# within 15 % of the local ones). The position process noise and the pose variances are NOT noise levels the data can
# identify: scan-matching error is correlated in time, so the one-step-ahead likelihood always prefers a filter that
# follows the previous pose (R_slam to its lower bound, 3x the re-seeds, no smoother). They are a smoothness / tracking
# trade-off, chosen with misc/ekf_global_sweep.py on the six takes and independent (scan-only) SLAM poses:
#     old values (Q[x] = Q[y] = 4.9e-5 / 3.6e-5, R_slam = 0.01):  jitter 272 m/s^2, 16 cm from the pose, 39 re-seeds
#     Q[x] = Q[y] = 0.5, R_slam x30:                              jitter  89 m/s^2,  5 cm from the pose,  0 re-seeds
# (jitter: RMS acceleration implied by the filter's position at 100 Hz; the local filter has 16.) The pose is followed
# to about its own noise, and with these variances a pose more than ~2 m away is still rejected by the gate.
NOISE_GLOBAL = {
    **NOISE,
    'Q': [0.5, 0.5, *NOISE['Q'][2:]],  # the dynamics terms follow the local fit
}

# Global filter only. The pose stamps lag the car by pose_delay; R is generous because the localization is fed the
# local filter's odometry. pose_delay = 10 ms is what the pose innovations favour on independent (scan-only) SLAM poses
# of all six takes, with the car model of MODEL above, for both the generous R below and a sharp one (R = 0.01); the
# likelihood is flat from 0 to 20 ms (0.007 per sample between 10 and 20) and clearly worse from 45 ms. With the car
# parameters before 2026-10-24 the optimum was 20 ms, so it moves with the actuator delays in MODEL. See docs, section 5c.
SLAM = {
    'pose_delay': 0.01,
    'R_slam_pose': [0.3, 0.3, 0.15],
    'pose_gate': 16.0,  # chi-square threshold, 3 dof
    'pose_reseed': 15,  # consecutive gated poses before re-seeding
}


# The map-frame pose as local odometry plus a low-passed map->odom (MapOdomCorrection). tau is set from
# misc/ekf_correction_sweep.py; the gate is geometric because there is no filter covariance to normalise with.
CORRECTION = {
    'tau': 0.15,  # bandwidth of the correction: ~5 cm from the pose, close to the global EKF's tracking
    'smooth': 0.05,  # extra output lag; takes the per-pose step out of the position (jitter 103 -> 32)
    'pose_delay': SLAM['pose_delay'],
    'gate_distance': 2.0,  # m: only a kidnapping; a smaller step (a loop closure) is absorbed at the filter's own pace
    'gate_angle': 1.0,  # rad
    'reseed': 15,  # consecutive rejected poses before the correction is set from the pose
    'std': [0.05, 0.05, 0.03],  # reported pose uncertainty added to the local filter's, [x, y, yaw]
    'std_growth': [0.1, 0.1, 0.05],  # ... and how fast it grows per second without a SLAM pose
}


def vehicle_model_params(get=MODEL.__getitem__):
    """``VehicleModelParams`` from ``get(name)``, which returns the value of one MODEL entry."""
    return VehicleModelParams(
        m=get('mass'), I_z=get('yaw_inertia'), l_f=get('front_axle_distance'), l_r=get('rear_axle_distance'),
        h_cg=get('cg_height'),
        Bf=get('front_B'), Cf=get('front_C'), Df=get('front_D'), Ef=get('front_E'),
        Br=get('rear_B'), Cr=get('rear_C'), Dr=get('rear_D'), Er=get('rear_E'),
        front_drive_frac=get('front_drive_frac'), constraint_time_constant=get('constraint_time_constant'),
        blend_center=get('blend_center'), slip_speed_regularizer=get('slip_speed_regularizer'),
        steer_gain=get('steer_gain'), steer_bias=get('steer_bias'),
        steer_delay=get('steer_delay'), steer_latency=get('steer_latency'), speed_delay=get('speed_delay'),
        drive_gain=get('drive_gain'), brake_gain=get('brake_gain'), imu_x=get('imu_x'))
