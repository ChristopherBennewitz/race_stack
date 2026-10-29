#!/usr/bin/env python3
"""Offline replay of the local simulator-model EKF on recorded bags (no ROS needed).

Runs ``SimulatorSingleTrackModel`` with the node's tick structure (fixed-rate predict, then a
sequential update per sensor with a fresh sample, steering and speed commands delayed by ``steer_delay`` and ``speed_delay``) on the
recorded ``/sensors/imu/raw``, ``/ackermann_cmd_applied`` and the VESC odometry ``/odom``, with the node's
defaults (``state_estimation.ekf_node.defaults``) unless a ``FilterConfig`` says otherwise.

``/car_state/pose`` (the SLAM pose) is the reference: body-frame velocities are differentiated from it with a
short local linear fit. Pose is not fused (its R is huge); only the twist is checked. Bags that predate
``/car_state/pose`` fall back to ``/car_state/odom``. If the bag has ``/state_estimation/odom`` or
``/early_fusion/odom`` (the live EKF's output) its error against the same reference is printed for comparison.

The sysid bags' SLAM pose was computed from the live estimator's odometry, so the reference is not independent
of it; it is good for judging velocity, not for claiming absolute accuracy.

``replay`` also returns the innovation statistics ``ekf_tune.py`` fits the noise to.

Run with an interpreter that has casadi, numpy and rosbags, e.g.
``f1tenth_parallel_sim/.venv/bin/python misc/ekf_replay.py <bag> [<bag> ...]``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import casadi as ca
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from state_estimation.ekf_node import defaults  # noqa: E402
from state_estimation.ekf_node.models import CommandDelay, SimulatorSingleTrackModel  # noqa: E402
from state_estimation.ekf_node.prediction_history import PredictionHistory  # noqa: E402

HZ = 100.0
WARMUP = 2.0  # seconds; the filter starts from a vague prior, so its first innovations say nothing about the noise
NLL_DIMS = {'imu': [0, 1, 2, 3], 'vesc': [3, 5]}  # the measured components; the rest of the VESC row is a placeholder


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_of(q):
    return np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


@dataclass
class FilterConfig:
    """What the replay varies: the filter's noise and the few sensor constants the tuner may refit."""

    Q: np.ndarray = field(default_factory=lambda: np.diag(defaults.NOISE['Q']))
    R_imu: np.ndarray = field(default_factory=lambda: np.diag(defaults.NOISE['R_imu']))
    R_vesc: np.ndarray = field(default_factory=lambda: np.diag(defaults.NOISE['R_vesc']))
    accel_bias: np.ndarray = field(default_factory=lambda: np.array(defaults.NOISE['imu_accel_bias']))
    accel_lag: float = defaults.NOISE['imu_accel_lag']
    steer_delay: float = defaults.MODEL['steer_delay']
    speed_delay: float = defaults.MODEL['speed_delay']
    speed_bias: float = 0.0  # added to the VESC speed (m/s)


class Engine:
    """The CasADi model and its predict, built once; ``FilterConfig`` only changes the noise."""

    def __init__(self, params=None):
        self.params = params or defaults.vehicle_model_params()
        self.model = SimulatorSingleTrackModel(self.params, np.diag(defaults.NOISE['Q']))
        x, u, dt = ca.MX.sym('x', 7), ca.MX.sym('u', 2), ca.MX.sym('dt')
        # One call returns the next state and its Jacobian; calls dominate the replay's cost.
        self.step = ca.Function('step', [x, u, dt], [self.model.f(x, u, dt), self.model.F(x, u, dt)])


def _typestore():
    from rosbags.typesys import Stores, get_typestore
    from rosbags.typesys.msg import get_types_from_msg

    store = get_typestore(Stores.ROS2_HUMBLE)
    store.register(get_types_from_msg(
        'float32 steering_angle\nfloat32 steering_angle_velocity\nfloat32 speed\nfloat32 acceleration\nfloat32 jerk',
        'ackermann_msgs/msg/AckermannDrive'))
    store.register(get_types_from_msg(
        'std_msgs/Header header\nackermann_msgs/AckermannDrive drive', 'ackermann_msgs/msg/AckermannDriveStamped'))
    return store


def read_bag(path):
    """The replay's inputs from one bag: IMU, applied commands, VESC odometry, reference pose, live EKF output."""
    from rosbags.highlevel import AnyReader

    topics = {'/sensors/imu/raw': 'imu', '/ackermann_cmd_applied': 'cmd', '/car_state/pose': 'odom',
              '/car_state/odom': 'odom_old', '/odom': 'vesc', '/state_estimation/odom': 'live',
              '/early_fusion/odom': 'live_old'}
    rows = {k: [] for k in set(topics.values()) | {'pose'}}
    with AnyReader([Path(path)], default_typestore=_typestore()) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        for conn, received, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            kind = topics[conn.topic]
            if kind == 'imu':
                rows[kind].append((stamp(m), yaw_of(m.orientation), m.angular_velocity.z,
                                   m.linear_acceleration.x, m.linear_acceleration.y))
            elif kind == 'cmd':
                rows[kind].append((stamp(m), m.drive.speed, m.drive.steering_angle))
            elif kind == 'odom':  # PoseStamped
                rows[kind].append((stamp(m), m.pose.position.x, m.pose.position.y, yaw_of(m.pose.orientation),
                                   0.0, 0.0, 0.0))
                rows['pose'].append((received * 1e-9, stamp(m), m.pose.position.x, m.pose.position.y,
                                     yaw_of(m.pose.orientation)))
            else:  # Odometry
                p, t = m.pose.pose, m.twist.twist
                rows[kind].append((stamp(m), p.position.x, p.position.y, yaw_of(p.orientation),
                                   t.linear.x, t.linear.y, t.angular.z))
    out = {'imu': rows['imu'], 'cmd': rows['cmd'], 'vesc': rows['vesc'], 'pose': rows['pose'],
           'odom': rows['odom'] or rows['odom_old'], 'live': rows['live'] or rows['live_old']}
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


def ekf_update(x, P, z, hx, H, R, angle_idx=None, nll_dims=None):
    """Joseph-form EKF update. Returns the state, covariance, NIS and the Gaussian NLL of ``nll_dims`` (or all)."""
    y = z - hx
    if angle_idx is not None:
        y[angle_idx] = wrap(y[angle_idx])
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    I_KH = np.eye(len(x)) - K @ H
    ys, Ss = (y, S) if nll_dims is None else (y[nll_dims], S[np.ix_(nll_dims, nll_dims)])
    nll = 0.5 * (float(ys @ np.linalg.solve(Ss, ys)) + np.linalg.slogdet(Ss)[1] + len(ys) * np.log(2 * np.pi))
    return x + K @ y, I_KH @ P @ I_KH.T + K @ R @ K.T, float(y @ np.linalg.solve(S, y)), nll


def replay(bag, config=None, engine=None, reference_window=0.1):
    """Runs the filter over one bag.

    Returns ``(rows, stats)``. ``rows`` has one line per tick after the first second with a reference:
    (t, vx, vy, yaw rate, reference vx, vy, yaw rate, VESC vx, VESC yaw rate, live vx, live yaw rate).
    plus the filter's base_link pose (x, y, yaw). ``stats`` holds the NIS and the summed NLL with its count per
    sensor, taken after ``WARMUP``.
    """
    config = config or FilterConfig()
    engine = engine or Engine()
    model = engine.model
    step_fn = engine.step
    history = PredictionHistory(config.accel_lag)
    delay = CommandDelay(config.steer_delay)
    speed_delay = CommandDelay(config.speed_delay)
    imu, cmd, odom = bag['imu'], bag['cmd'], bag['odom']
    vesc = bag['vesc'] if len(bag['vesc']) else odom  # raw VESC odometry, else the old EKF's twist
    t0 = max(imu[0, 0], cmd[0, 0], odom[0, 0], vesc[0, 0])
    t1 = min(imu[-1, 0], cmd[-1, 0], odom[-1, 0], vesc[-1, 0])
    if 'reference' not in bag:
        bag['reference'] = reference_velocity(odom, reference_window)  # a polyfit per sample: kept for the next replay
    ref_vx, ref_vy, ref_wz = bag['reference']

    yaw0 = imu[np.searchsorted(imu[:, 0], t0), 1]
    x = np.zeros(7)
    P = model.P_0.copy()
    x[5] = imu[np.searchsorted(imu[:, 0], t0), 2]
    P[2, 2], P[5, 5] = config.R_imu[0, 0], config.R_imu[1, 1]
    i_imu = i_cmd = i_vesc = 0
    speed_cmd = 0.0
    imu_z = None
    imu_fresh = False
    nis = {'imu': [], 'vesc': []}
    nll = {'imu': 0.0, 'vesc': 0.0}
    count = {'imu': 0, 'vesc': 0}
    rows = []
    live = bag['live']
    substeps = int(np.ceil(1.0 / HZ / defaults.NOISE['predict_substep']))
    h = 1.0 / HZ / substeps
    Qh = config.Q * h
    for t in np.arange(t0, t1, 1.0 / HZ):
        while i_cmd < len(cmd) and cmd[i_cmd, 0] <= t:
            speed_cmd = cmd[i_cmd, 1]
            delay.push(cmd[i_cmd, 0], cmd[i_cmd, 2])
            speed_delay.push(cmd[i_cmd, 0], cmd[i_cmd, 1])
            i_cmd += 1
        while i_imu < len(imu) and imu[i_imu, 0] <= t:
            _, yaw, wz, ax, ay = imu[i_imu]
            # imu_transformer: the IMU frame is base_link rotated by 90 degrees about z.
            a = np.array([-ay, ax]) - config.accel_bias
            imu_z = np.array([wrap(yaw - yaw0), wz, *a])
            i_imu += 1
            imu_fresh = True
        vesc_z = None
        while i_vesc < len(vesc) and vesc[i_vesc, 0] <= t:
            vesc_z = np.array([vesc[i_vesc, 1], vesc[i_vesc, 2], vesc[i_vesc, 3],
                               vesc[i_vesc, 4] + config.speed_bias, 0.0, vesc[i_vesc, 6]])
            i_vesc += 1
        if t == t0:
            continue
        u = np.array([speed_delay.at(t), delay.at(t)])
        for k in range(substeps):
            u_step = np.array([speed_delay.at(t - 1.0 / HZ + (k + 0.5) * h), delay.at(t - 1.0 / HZ + (k + 0.5) * h)])
            xn, F = step_fn(x, u_step, h)
            xn, F = np.array(xn).flatten(), np.array(F).reshape(7, 7)
            x, P = xn, F @ P @ F.T + Qh
        scored = t - t0 > WARMUP
        if imu_fresh and imu_z is not None:
            hx = model.Hx_imu(x, u)
            if history.accel is not None:
                hx[2:] = history.accel
            x, P, d, ll = ekf_update(x, P, imu_z, hx, model.HJacobian_imu(x, u), config.R_imu, 0, NLL_DIMS['imu'])
            if scored:
                nis['imu'].append(d)
                nll['imu'] += ll
                count['imu'] += 1
            imu_fresh = False
        if vesc_z is not None:
            x, P, d, ll = ekf_update(x, P, vesc_z, model.Hx_vesc(x), model.HJacobian_vesc(x), config.R_vesc,
                                     None, NLL_DIMS['vesc'])
            if scored:
                nis['vesc'].append(d)
                nll['vesc'] += ll
                count['vesc'] += 1
        history.record(t, model.Hx_pose(x), model.Hx_imu(x, u)[2:])
        j = np.searchsorted(odom[:, 0], t)
        if j < len(odom) and np.isfinite(ref_vx[j]) and t - t0 > 1.0:
            k = min(np.searchsorted(vesc[:, 0], t), len(vesc) - 1)
            lv = np.full(2, np.nan)
            if len(live):
                m = min(np.searchsorted(live[:, 0], t), len(live) - 1)
                lv = live[m, [4, 6]]
            rows.append((t, x[3], x[4], x[5], ref_vx[j], ref_vy[j], ref_wz[j],
                         vesc[k, 4] + config.speed_bias, vesc[k, 6], lv[0], lv[1], *model.Hx_pose(x)))
    stats = {'nis': nis, 'nll': nll, 'count': count}
    return np.array(rows), stats


def summarize(name, rows, stats):
    e = rows[:, 1:4] - rows[:, 4:7]
    rmse = np.sqrt((e ** 2).mean(0))
    meas = np.sqrt(((rows[:, [7, 8]] - rows[:, [4, 6]]) ** 2).mean(0))
    live = np.sqrt(np.nanmean((rows[:, [9, 10]] - rows[:, [4, 6]]) ** 2, axis=0)) if np.isfinite(rows[:, 9]).any() else None
    live_txt = '' if live is None else f' | live EKF rmse vx {live[0]:.3f} wz {live[1]:.3f}'
    nis, count, nll = stats['nis'], stats['count'], stats['nll']
    print(f'{name:46s} EKF rmse vx {rmse[0]:.3f} vy {rmse[1]:.3f} wz {rmse[2]:.3f} | '
          f'VESC odom rmse vx {meas[0]:.3f} wz {meas[1]:.3f}{live_txt} | '
          f'NIS imu {np.mean(nis["imu"]):.2f}/4 vesc {np.mean(nis["vesc"]):.2f}/6 | '
          f'NLL/sample imu {nll["imu"] / max(count["imu"], 1):.2f} vesc {nll["vesc"] / max(count["vesc"], 1):.2f}')
    return rmse, meas


def load_pose_file(path, latency=0.0):
    """Rows of ``t x y yaw`` (csv or npy) as the replay's (receive time, stamp, x, y, yaw)."""
    data = np.loadtxt(path, delimiter=',', ndmin=2) if str(path).endswith('.csv') else np.load(path)
    return np.column_stack([data[:, 0] + latency, data[:, 0], data[:, 1], data[:, 2], data[:, 3]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bags', nargs='+')
    ap.add_argument('--speed-bias', type=float, default=0.0, help='offset added to the odometry speed (m/s)')
    ap.add_argument('--q-scale', type=float, default=1.0)
    ap.add_argument('--steer-delay', type=float, default=None, help='override the model steering delay (s)')
    ap.add_argument('--speed-delay', type=float, default=None, help='override the model speed command delay (s)')
    ap.add_argument('--no-accel', action='store_true', help='do not fuse the IMU accelerations')
    ap.add_argument('--noise-json', help='use the noise of an ekf_tune.py result (its "noise" entry)')
    ap.add_argument('--reference-dir', help='score against the poses in <dir>/<bag name>.csv (t x y yaw), '
                    'e.g. from offline_slam_poses.py, instead of the bag\'s own /car_state/pose')
    ap.add_argument('--reference-window', type=float, default=0.1,
                    help='half-width (s) of the linear fit that turns the reference poses into velocities; a noisier pose '
                    'needs a wider one (0.3 for the offline-SLAM pose)')
    args = ap.parse_args()
    config = FilterConfig()
    config.Q = config.Q * args.q_scale
    config.speed_bias = args.speed_bias
    if args.steer_delay is not None:
        config.steer_delay = args.steer_delay
    if args.speed_delay is not None:
        config.speed_delay = args.speed_delay
    if args.noise_json:
        import json
        noise = json.loads(Path(args.noise_json).read_text())['noise']
        config.Q, config.R_imu, config.R_vesc = np.diag(noise['Q']), np.diag(noise['R_imu']), np.diag(noise['R_vesc'])
        config.accel_bias = np.array(noise['imu_accel_bias'])
    if args.no_accel:
        config.R_imu = np.diag([defaults.NOISE['R_imu'][0], defaults.NOISE['R_imu'][1], 1e6, 1e6])
    engine = Engine()
    all_e = []
    for path in args.bags:
        bag = read_bag(path)
        name = Path(path).name.replace('sysid_', '')
        if args.reference_dir:
            poses = load_pose_file(Path(args.reference_dir) / (Path(path).name + '.csv'))
            bag['odom'] = np.column_stack([poses[:, 1], poses[:, 2], poses[:, 3], poses[:, 4], np.zeros((len(poses), 3))])
        rows, stats = replay(bag, config, engine, reference_window=args.reference_window)
        summarize(name, rows, stats)
        all_e.append(rows[:, 1:4] - rows[:, 4:7])
    e = np.vstack(all_e)
    print('POOLED EKF rmse (vx, vy, wz):', np.sqrt((e ** 2).mean(0)).round(3))


if __name__ == '__main__':
    main()
