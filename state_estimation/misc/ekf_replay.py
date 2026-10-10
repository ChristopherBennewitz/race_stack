#!/usr/bin/env python3
"""Offline replay of the simulator-model EKF on recorded bags (no ROS needed).

Runs ``SimulatorSingleTrackModel`` with the node's tick structure (fixed-rate predict, then a
sequential update per sensor with a fresh sample) on the recorded ``/sensors/imu/raw``,
``/ackermann_cmd_applied`` and ``/car_state/odom``.

The bags contain no raw VESC odometry, so ``/car_state/odom``'s twist (the previous EKF's output,
whose speed error against SLAM is what the sysid's ``v`` sigma describes) stands in for it, and its
SLAM pose is the reference: body-frame velocities are differentiated from it with a short local
linear fit. Pose is not fused (its R is huge); only the twist is checked, because that is what
``carstate_node`` forwards. The proxy is optimistic about odometry errors, so use ``--speed-bias`` to
inject the VESC's rest offset and see its effect.

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
from state_estimation.ekf_node.models import SimulatorSingleTrackModel, VehicleModelParams  # noqa: E402

# Node defaults (ekf_node.py).
PARAMS = VehicleModelParams(
    m=3.718, I_z=0.0561, l_f=0.17334, l_r=0.14766, h_cg=0.074,
    Bf=12.0306, Cf=1.71178, Df=0.402674, Ef=0.0, Br=5.06904, Cr=1.64946, Dr=0.572528, Er=0.0,
    steer_gain=1.50936, steer_bias=0.00332491, steer_latency=0.111489,
    drive_gain=3.59866, brake_gain=5.00361,
)
Q = np.diag([4.9e-5, 3.6e-5, 3.6e-5, 5.8e-4, 1.8e-3, 6.7e-2, 2.3e-5, 6.9e-3])
R_IMU = np.diag([0.1, 0.0009, 1e6, 1e6])
R_VESC = np.diag([1e6, 1e6, 1e6, 0.04, 100.0, 0.1])  # pose not fused in the replay
HZ = 100.0
ACCEL_EMA = 0.2


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
    out = {'imu': [], 'cmd': [], 'odom': []}
    topics = {'/sensors/imu/raw': 'imu', '/ackermann_cmd_applied': 'cmd', '/car_state/odom': 'odom'}
    with AnyReader([Path(path)], default_typestore=_typestore()) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        for conn, _, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            kind = topics[conn.topic]
            if kind == 'imu':
                out['imu'].append((stamp(m), yaw_of(m.orientation), m.angular_velocity.z,
                                   m.linear_acceleration.x, m.linear_acceleration.y))
            elif kind == 'cmd':
                out['cmd'].append((stamp(m), m.drive.speed, m.drive.steering_angle))
            else:
                p, t = m.pose.pose, m.twist.twist
                out['odom'].append((stamp(m), p.position.x, p.position.y, yaw_of(p.orientation),
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


def replay(bag, speed_bias=0.0, q_scale=1.0):
    model = SimulatorSingleTrackModel(PARAMS, Q * q_scale)
    imu, cmd, odom = bag['imu'], bag['cmd'], bag['odom']
    t0 = max(imu[0, 0], cmd[0, 0], odom[0, 0])
    t1 = min(imu[-1, 0], cmd[-1, 0], odom[-1, 0])
    ref_vx, ref_vy, ref_wz = reference_velocity(odom)

    x = np.zeros(8)
    x[2], x[5] = imu[np.searchsorted(imu[:, 0], t0), 1:3]
    P = model.P_0.copy()
    P[2, 2], P[5, 5] = R_IMU[0, 0], R_IMU[1, 1]
    i_imu = i_cmd = i_odom = 0
    ema = None
    u = np.zeros(2)
    imu_z = odom_z = None
    imu_fresh = False
    nis = {'imu': [], 'vesc': []}
    rows = []
    for t in np.arange(t0, t1, 1.0 / HZ):
        while i_cmd < len(cmd) and cmd[i_cmd, 0] <= t:
            u = cmd[i_cmd, 1:3].copy()
            i_cmd += 1
        while i_imu < len(imu) and imu[i_imu, 0] <= t:
            _, yaw, wz, ax, ay = imu[i_imu]
            a = np.array([ax, ay])
            ema = a if ema is None else ACCEL_EMA * a + (1 - ACCEL_EMA) * ema
            imu_z = np.array([wrap(yaw), wz, *ema])
            i_imu += 1
            imu_fresh = True
        odom_fresh = False
        while i_odom < len(odom) and odom[i_odom, 0] <= t:
            odom_z = np.array([odom[i_odom, 1], odom[i_odom, 2], odom[i_odom, 3],
                               odom[i_odom, 4] + speed_bias, 0.0, odom[i_odom, 6]])
            i_odom += 1
            odom_fresh = True
        if t == t0:
            continue
        xn = np.array(model.f(x, u, 1.0 / HZ)).flatten()
        F = np.array(model.F(x, u, 1.0 / HZ)).reshape(8, 8)
        x, P = xn, F @ P @ F.T + model.Q / HZ
        if imu_fresh and imu_z is not None:
            x, P, d = ekf_update(x, P, imu_z, model.Hx_imu(x, u), model.HJacobian_imu(x, u), R_IMU, 0)
            nis['imu'].append(d)
            imu_fresh = False
        if odom_fresh:
            x, P, d = ekf_update(x, P, odom_z, model.Hx_vesc(x), model.HJacobian_vesc(x), R_VESC)
            nis['vesc'].append(d)
        j = np.searchsorted(odom[:, 0], t)
        if j < len(odom) and np.isfinite(ref_vx[j]) and t - t0 > 1.0:
            rows.append((t, x[3], x[4], x[5], ref_vx[j], ref_vy[j], ref_wz[j],
                         odom[j, 4] + speed_bias, odom[j, 6]))
    return np.array(rows), nis


def summarize(name, rows, nis):
    e = rows[:, 1:4] - rows[:, 4:7]
    rmse = np.sqrt((e ** 2).mean(0))
    meas = np.sqrt(((rows[:, [7, 8]] - rows[:, [4, 6]]) ** 2).mean(0))
    print(f'{name:48s} EKF rmse vx {rmse[0]:.3f} vy {rmse[1]:.3f} wz {rmse[2]:.3f} | '
          f'odom-proxy rmse vx {meas[0]:.3f} wz {meas[1]:.3f} | '
          f'NIS imu {np.mean(nis["imu"]):.2f}/2 vesc {np.mean(nis["vesc"]):.2f}/3')
    return rmse, meas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bags', nargs='+')
    ap.add_argument('--speed-bias', type=float, default=0.0, help='offset added to the odometry speed (m/s)')
    ap.add_argument('--q-scale', type=float, default=1.0)
    args = ap.parse_args()
    all_e = []
    for path in args.bags:
        rows, nis = replay(read_bag(path), args.speed_bias, args.q_scale)
        summarize(Path(path).name.replace('sysid_20260904_', ''), rows, nis)
        all_e.append(rows[:, 1:4] - rows[:, 4:7])
    e = np.vstack(all_e)
    print('POOLED EKF rmse (vx, vy, wz):', np.sqrt((e ** 2).mean(0)).round(3))


if __name__ == '__main__':
    main()
