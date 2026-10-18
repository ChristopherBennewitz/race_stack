#!/usr/bin/env python3
"""Offline replay of the simulator-model EKF on recorded bags (no ROS needed).

Runs ``SimulatorSingleTrackModel`` with the node's tick structure (fixed-rate predict, then a
sequential update per sensor with a fresh sample, steering command delayed by ``steer_delay``) on the
recorded ``/sensors/imu/raw``, ``/ackermann_cmd_applied`` and the VESC odometry ``/odom``. Bags recorded
before ``/odom`` was added fall back to ``/car_state/odom``'s twist (the previous EKF's output) as the stand-in.

``/car_state/odom``'s SLAM pose is the reference: body-frame velocities are differentiated from it with a
short local linear fit. Pose is not fused (its R is huge); only the twist is checked, because that is what
``carstate_node`` forwards. If the bag has ``/early_fusion/odom`` (the live EKF's output) its error against
the same reference is printed for comparison.

Run with an interpreter that has casadi, numpy and rosbags, e.g.
``f1tenth_parallel_sim/.venv/bin/python misc/ekf_replay.py <bag> [<bag> ...]``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.msg import get_types_from_msg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from state_estimation.ekf_node.models import CommandDelay, SimulatorSingleTrackModel, VehicleModelParams  # noqa: E402
from state_estimation.ekf_node.prediction_history import PredictionHistory  # noqa: E402
from state_estimation.ekf_node.speed_window import TrailingSpeedWindow  # noqa: E402

# Node defaults (ekf_node.py).
PARAMS = VehicleModelParams(
    m=3.718, I_z=0.0561, l_f=0.17334, l_r=0.14766, h_cg=0.074,
    Bf=12.0306, Cf=1.71178, Df=0.402674, Ef=0.0, Br=5.06904, Cr=1.64946, Dr=0.572528, Er=0.0,
    steer_gain=1.6746, steer_bias=0.000807634, steer_delay=0.0871629, steer_latency=0.0338089,
    drive_gain=3.99169, brake_gain=5.51182,
)
Q = np.diag([4.9e-5, 3.6e-5, 3.6e-5, 4.0e-4, 2.6e-3, 0.18, 4.2e-3])
R_IMU = np.diag([0.1, 0.0009, 1.4, 3.0])
R_IMU_NO_ACCEL = np.diag([0.1, 0.0009, 1e6, 1e6])
IMU_ACCEL_BIAS = np.array([0.05, 0.56])
IMU_ACCEL_LAG = 0.05
R_SLAM_SPEED = np.diag([0.04])
SLAM_SPEED_WINDOW = 0.2
SLAM_POSE_DELAY = 0.045
SLAM_SPEED_GATE = 3.0
PREDICT_SUBSTEP = 0.005
R_VESC = np.diag([1e6, 1e6, 1e6, 0.0225, 100.0, 0.1])  # pose not fused in the replay
HZ = 100.0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_of(q):
    return np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def _typestore():
    store = get_typestore(Stores.ROS2_HUMBLE)
    store.register(get_types_from_msg(
        'float32 steering_angle\nfloat32 steering_angle_velocity\nfloat32 speed\nfloat32 acceleration\nfloat32 jerk',
        'ackermann_msgs/msg/AckermannDrive'))
    store.register(get_types_from_msg(
        'std_msgs/Header header\nackermann_msgs/AckermannDrive drive', 'ackermann_msgs/msg/AckermannDriveStamped'))
    return store


def read_bag(path):
    out = {'imu': [], 'cmd': [], 'odom': [], 'vesc': [], 'live': []}
    topics = {'/sensors/imu/raw': 'imu', '/ackermann_cmd_applied': 'cmd', '/car_state/odom': 'odom',
              '/odom': 'vesc', '/early_fusion/odom': 'live'}
    with AnyReader([Path(path)], default_typestore=_typestore()) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        for conn, _, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            kind = topics[conn.topic]
            if kind == 'imu':
                out[kind].append((stamp(m), yaw_of(m.orientation), m.angular_velocity.z,
                                  m.linear_acceleration.x, m.linear_acceleration.y))
            elif kind == 'cmd':
                out[kind].append((stamp(m), m.drive.speed, m.drive.steering_angle))
            else:
                p, t = m.pose.pose, m.twist.twist
                out[kind].append((stamp(m), p.position.x, p.position.y, yaw_of(p.orientation),
                                  t.linear.x, t.linear.y, t.angular.z))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def reference_velocity(odom, half_window=0.1):
    """Body-frame (vx, vy, yaw_rate) from the SLAM pose by a local linear fit."""
    t, x, y, yaw = odom[:, 0], odom[:, 1], odom[:, 2], np.unwrap(odom[:, 3])
    vx, vy, wz = (np.full(len(t), np.nan) for _ in range(3))
    for i in range(len(t)):
        sel = np.abs(t - t[i]) <= half_window
        if sel.sum() < 4:
            continue
        tt = t[sel] - t[i]
        slope = lambda v: np.polyfit(tt, v[sel], 1)[0]
        dx, dy, dyaw = slope(x), slope(y), slope(yaw)
        c, s = np.cos(yaw[i]), np.sin(yaw[i])
        vx[i], vy[i], wz[i] = c * dx + s * dy, -s * dx + c * dy, dyaw
    return vx, vy, wz


def ekf_update(x, P, z, hx, H, R, angle_idx=None):
    y = z - hx
    if angle_idx is not None:
        y[angle_idx] = wrap(y[angle_idx])
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    I_KH = np.eye(len(x)) - K @ H
    return x + K @ y, I_KH @ P @ I_KH.T + K @ R @ K.T, float(y @ np.linalg.solve(S, y))


def replay(bag, speed_bias=0.0, q_scale=1.0, steer_delay=None, accel=True, slam_speed=True):
    model = SimulatorSingleTrackModel(PARAMS, Q * q_scale)
    R_imu = R_IMU if accel else R_IMU_NO_ACCEL
    tracker = TrailingSpeedWindow(SLAM_SPEED_WINDOW)
    history = PredictionHistory(IMU_ACCEL_LAG)
    slam_z, slam_span, slam_rejected = None, None, 0
    delay = CommandDelay(PARAMS.steer_delay if steer_delay is None else steer_delay)
    imu, cmd, odom = bag['imu'], bag['cmd'], bag['odom']
    vesc = bag['vesc'] if len(bag['vesc']) else odom  # raw VESC odometry, else the old EKF's twist
    t0 = max(imu[0, 0], cmd[0, 0], odom[0, 0], vesc[0, 0])
    t1 = min(imu[-1, 0], cmd[-1, 0], odom[-1, 0], vesc[-1, 0])
    ref_vx, ref_vy, ref_wz = reference_velocity(odom)

    yaw0 = imu[np.searchsorted(imu[:, 0], t0), 1]
    x = np.zeros(7)
    x[5] = imu[np.searchsorted(imu[:, 0], t0), 2]
    P = model.P_0.copy()
    P[2, 2], P[5, 5] = R_imu[0, 0], R_imu[1, 1]
    i_imu = i_cmd = i_vesc = 0
    speed_cmd = 0.0
    imu_z = None
    imu_fresh = False
    nis = {'imu': [], 'vesc': []}
    i_pose = 0
    rows = []
    live = bag['live']
    for t in np.arange(t0, t1, 1.0 / HZ):
        while i_cmd < len(cmd) and cmd[i_cmd, 0] <= t:
            speed_cmd = cmd[i_cmd, 1]
            delay.push(cmd[i_cmd, 0], cmd[i_cmd, 2])
            i_cmd += 1
        while i_imu < len(imu) and imu[i_imu, 0] <= t:
            _, yaw, wz, ax, ay = imu[i_imu]
            # imu_transformer: the IMU frame is base_link rotated by 90 degrees about z.
            a = np.array([-ay, ax]) - IMU_ACCEL_BIAS
            imu_z = np.array([wrap(yaw - yaw0), wz, *a])
            i_imu += 1
            imu_fresh = True
        vesc_z = None
        while i_vesc < len(vesc) and vesc[i_vesc, 0] <= t:
            vesc_z = np.array([vesc[i_vesc, 1], vesc[i_vesc, 2], vesc[i_vesc, 3],
                               vesc[i_vesc, 4] + speed_bias, 0.0, vesc[i_vesc, 6]])
            i_vesc += 1
        if t == t0:
            continue
        while i_pose < len(odom) and odom[i_pose, 0] <= t:
            sample = tracker.push(odom[i_pose, 0], odom[i_pose, 1], odom[i_pose, 2], odom[i_pose, 3])
            if sample is not None and slam_speed:
                slam_z = np.array([sample[0]])
                slam_span = (odom[i_pose, 0] - sample[1] - SLAM_POSE_DELAY, odom[i_pose, 0] - SLAM_POSE_DELAY)
            i_pose += 1
        u = np.array([speed_cmd, delay.at(t)])
        substeps = int(np.ceil(1.0 / HZ / PREDICT_SUBSTEP))
        h = 1.0 / HZ / substeps
        for k in range(substeps):
            u_step = np.array([speed_cmd, delay.at(t - 1.0 / HZ + (k + 0.5) * h)])
            xn = np.array(model.f(x, u_step, h)).flatten()
            F = np.array(model.F(x, u_step, h)).reshape(7, 7)
            x, P = xn, F @ P @ F.T + model.Q * h
        if imu_fresh and imu_z is not None:
            hx = model.Hx_imu(x, u)
            if history.accel is not None:
                hx[2:] = history.accel
            x, P, d = ekf_update(x, P, imu_z, hx, model.HJacobian_imu(x, u), R_imu, 0)
            nis['imu'].append(d)
            imu_fresh = False
        if vesc_z is not None:
            x, P, d = ekf_update(x, P, vesc_z, model.Hx_vesc(x), model.HJacobian_vesc(x), R_VESC)
            nis['vesc'].append(d)
        if slam_z is not None:
            predicted = history.mean_vx(*slam_span)
            if predicted is not None:
                H = np.zeros((1, 7))
                H[0, 3] = 1.0
                hx = np.array([predicted])
                S = H @ P @ H.T + R_SLAM_SPEED
                if float((slam_z - hx) @ np.linalg.solve(S, slam_z - hx)) > SLAM_SPEED_GATE ** 2:
                    slam_rejected += 1
                else:
                    x, P, d = ekf_update(x, P, slam_z, hx, H, R_SLAM_SPEED)
                    nis.setdefault('slam', []).append(d)
            slam_z = None
        history.record(t, x[3], model.Hx_imu(x, u)[2:])
        j = np.searchsorted(odom[:, 0], t)
        if j < len(odom) and np.isfinite(ref_vx[j]) and t - t0 > 1.0:
            k = min(np.searchsorted(vesc[:, 0], t), len(vesc) - 1)
            lv = np.full(2, np.nan)
            if len(live):
                m = min(np.searchsorted(live[:, 0], t), len(live) - 1)
                lv = live[m, [4, 6]]
            rows.append((t, x[3], x[4], x[5], ref_vx[j], ref_vy[j], ref_wz[j],
                         vesc[k, 4] + speed_bias, vesc[k, 6], lv[0], lv[1]))
    nis['slam_rejected'] = slam_rejected
    return np.array(rows), nis


def summarize(name, rows, nis):
    e = rows[:, 1:4] - rows[:, 4:7]
    rmse = np.sqrt((e ** 2).mean(0))
    meas = np.sqrt(((rows[:, [7, 8]] - rows[:, [4, 6]]) ** 2).mean(0))
    live = np.sqrt(np.nanmean((rows[:, [9, 10]] - rows[:, [4, 6]]) ** 2, axis=0)) if np.isfinite(rows[:, 9]).any() else None
    live_txt = '' if live is None else f' | live EKF rmse vx {live[0]:.3f} wz {live[1]:.3f}'
    print(f'{name:46s} EKF rmse vx {rmse[0]:.3f} vy {rmse[1]:.3f} wz {rmse[2]:.3f} | '
          f'VESC odom rmse vx {meas[0]:.3f} wz {meas[1]:.3f}{live_txt} | '
          f'NIS imu {np.mean(nis["imu"]):.2f}/2 vesc {np.mean(nis["vesc"]):.2f}/3')
    return rmse, meas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bags', nargs='+')
    ap.add_argument('--speed-bias', type=float, default=0.0, help='offset added to the odometry speed (m/s)')
    ap.add_argument('--q-scale', type=float, default=1.0)
    ap.add_argument('--steer-delay', type=float, default=None, help='override the model steering delay (s)')
    ap.add_argument('--no-accel', action='store_true', help='do not fuse the IMU accelerations (previous behaviour)')
    ap.add_argument('--no-slam-speed', action='store_true', help='do not fuse the SLAM forward speed')
    args = ap.parse_args()
    all_e = []
    for path in args.bags:
        rows, nis = replay(read_bag(path), args.speed_bias, args.q_scale, args.steer_delay,
                           not args.no_accel, not args.no_slam_speed)
        summarize(Path(path).name.replace('sysid_', ''), rows, nis)
        all_e.append(rows[:, 1:4] - rows[:, 4:7])
    e = np.vstack(all_e)
    print('POOLED EKF rmse (vx, vy, wz):', np.sqrt((e ** 2).mean(0)).round(3))


if __name__ == '__main__':
    main()
