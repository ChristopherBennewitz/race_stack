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

With a ``SlamConfig`` it replays the *global* filter instead: the same model and inputs, the IMU heading dropped, and
the SLAM pose fused through ``SlamPoseUpdater`` exactly as ``ekf_global_node`` does (pose delivered at its receive
time, looked up at its stamp minus the pose delay). The pose is the bag's ``/car_state/pose`` unless a file of
``t x y yaw`` rows (one per bag, ``--pose-dir``) replaces it, e.g. a pose from Cartographer re-run offline on the
recorded ``/scan``. The sysid bags' own pose came from the live estimator's odometry, so it is not independent of
the local filter's output; use it to check the plumbing and the delay, and an offline pose to tune the noise.

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
from state_estimation.ekf_node.slam_pose import SlamPoseUpdater  # noqa: E402

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

    @classmethod
    def for_global(cls):
        """The global filter's noise (``defaults.NOISE_GLOBAL``)."""
        n = defaults.NOISE_GLOBAL
        return cls(Q=np.diag(n['Q']), R_imu=np.diag(n['R_imu']), R_vesc=np.diag(n['R_vesc']),
                   accel_bias=np.array(n['imu_accel_bias']), accel_lag=n['imu_accel_lag'])


@dataclass
class SlamConfig:
    """The global filter's SLAM pose fusion; defaults are the node's."""

    R: np.ndarray = field(default_factory=lambda: np.diag(defaults.SLAM['R_slam_pose']))
    delay: float = defaults.SLAM['pose_delay']
    gate: float = defaults.SLAM['pose_gate']
    reseed_after: int = defaults.SLAM['pose_reseed']
    pose: np.ndarray | None = None  # rows of (receive time, stamp, x, y, yaw); the bag's own when None


class ReplayEkf:
    """A state and covariance with filterpy's ``update`` signature, so ``SlamPoseUpdater`` runs unchanged."""

    def __init__(self, x, P):
        self.x, self.P = x, P

    def update(self, z, HJacobian, Hx, R, residual):
        H = HJacobian(self.x)
        y = residual(z, Hx(self.x))
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        I_KH = np.eye(len(self.x)) - K @ H
        self.x = self.x + K @ y
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T


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


def replay(bag, config=None, engine=None, slam=None, reference_window=0.1):
    """Runs the filter over one bag.

    Returns ``(rows, stats)``. ``rows`` has one line per tick after the first second with a reference:
    (t, vx, vy, yaw rate, reference vx, vy, yaw rate, VESC vx, VESC yaw rate, live vx, live yaw rate).
    plus the filter's base_link pose (x, y, yaw). ``stats`` holds the NIS and the summed NLL with its count per
    sensor, taken after ``WARMUP``. With ``slam`` it is the global filter, and ``stats['slam']`` describes the pose
    fusion: outcome counts, the NIS / NLL of the innovations and the innovations themselves.
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
    r_imu = config.R_imu.copy()
    if slam is not None:
        pose = slam.pose if slam.pose is not None else bag['pose']
        t0 = max(t0, pose[0, 0])
        r_imu[0, 0] = 1e6  # the IMU heading is zero-referenced at its first sample, not in the map frame
    t1 = min(imu[-1, 0], cmd[-1, 0], odom[-1, 0], vesc[-1, 0])
    if 'reference' not in bag:
        bag['reference'] = reference_velocity(odom, reference_window)  # a polyfit per sample: kept for the next replay
    ref_vx, ref_vy, ref_wz = bag['reference']

    yaw0 = imu[np.searchsorted(imu[:, 0], t0), 1]
    x = np.zeros(7)
    P = model.P_0.copy()
    if slam is None:
        x[5] = imu[np.searchsorted(imu[:, 0], t0), 2]
        P[2, 2], P[5, 5] = config.R_imu[0, 0], config.R_imu[1, 1]
    else:  # the first pose seeds the state, as in the node
        updater = SlamPoseUpdater(model, slam.R, history, slam.delay, slam.gate, slam.reseed_after)
        seed = ReplayEkf(x, P)
        updater.seed(seed, [pose[0, 2], pose[0, 3], wrap(pose[0, 4])])
        x, P = seed.x, seed.P
        i_pose, last_stamp, pose_z = 1, pose[0, 1], None
        slam_stats = {'outcomes': {}, 'nis': [], 'nll': 0.0, 'count': 0, 'innovations': [], 'nll_clipped': 0.0, 'count_all': 0}
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
            x, P, d, ll = ekf_update(x, P, imu_z, hx, model.HJacobian_imu(x, u), r_imu, 0, NLL_DIMS['imu'])
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
        if slam is not None:
            while i_pose < len(pose) and pose[i_pose, 0] <= t:
                received, st, px, py, pyaw = pose[i_pose]
                i_pose += 1
                if st > last_stamp:  # a republished stamp carries no new scan match
                    last_stamp, pose_z = st, (np.array([px, py, wrap(pyaw)]), st)
            if pose_z is not None:
                ekf = ReplayEkf(x, P)
                outcome = updater.update(ekf, *pose_z)
                x, P = ekf.x, ekf.P
                slam_stats['outcomes'][outcome] = slam_stats['outcomes'].get(outcome, 0) + 1
                if scored and outcome in ('updated', 'gated', 'reseeded') and updater.last_innovation is not None:
                    y, S = updater.last_innovation
                    nis_pose = float(y @ np.linalg.solve(S, y))
                    # What the tuner scores: every pose, a rejected one at the gate. Scoring only the accepted poses would
                    # let a tight R discard the hard ones and look better for it.
                    slam_stats['nll_clipped'] += 0.5 * (min(nis_pose, slam.gate) + np.linalg.slogdet(S)[1] + 3 * np.log(2 * np.pi))
                    slam_stats['count_all'] += 1
                if scored and outcome in ('updated', 'gated') and updater.last_innovation is not None:
                    slam_stats['nis'].append(nis_pose)
                    slam_stats['innovations'].append(y)
                    if outcome == 'updated':  # a gated pose is an outlier by definition; it is counted, not scored
                        slam_stats['nll'] += 0.5 * (nis_pose + np.linalg.slogdet(S)[1] + 3 * np.log(2 * np.pi))
                        slam_stats['count'] += 1
                pose_z = None
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
    if slam is not None:
        slam_stats['innovations'] = np.array(slam_stats['innovations'])
        slam_stats['reseeds'], slam_stats['gated'] = updater.reseeds, updater.rejected
        stats['slam'] = slam_stats
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


def summarize_slam(name, rows, stats, pose):
    """The global filter against the SLAM pose it fuses: innovation consistency and how much it smooths."""
    slam = stats['slam']
    n = max(len(slam['nis']), 1)
    outcomes = ', '.join(f'{k} {v}' for k, v in sorted(slam['outcomes'].items()))
    inn = slam['innovations']
    rms = np.sqrt((inn ** 2).mean(0)) if len(inn) else np.full(3, np.nan)
    # Smoothness: the acceleration implied by the second difference of the position, for the filter at its tick
    # rate and for the raw pose at its own stamps. A jump in the pose shows up here as a spike.
    def accel_rms(t, xy):
        keep = np.diff(t) > 0
        t, xy = np.append(t[:-1][keep], t[-1]), np.append(xy[:-1][keep], xy[-1:], axis=0)
        v = np.diff(xy, axis=0) / np.diff(t)[:, None]
        a = np.diff(v, axis=0) / (0.5 * (t[2:] - t[:-2]))[:, None]
        return float(np.sqrt((np.linalg.norm(a, axis=1) ** 2).mean()))
    inside = (pose[:, 1] >= rows[0, 0]) & (pose[:, 1] <= rows[-1, 0])
    print(f'{name:46s} pose NIS {np.mean(slam["nis"]) if slam["nis"] else float("nan"):.2f}/3 '
          f'innovation rms x {rms[0]:.3f} y {rms[1]:.3f} yaw {rms[2]:.3f} | '
          f'pose accel rms: filter {accel_rms(rows[:, 0], rows[:, 11:13]):.1f}  '
          f'raw SLAM {accel_rms(pose[inside, 1], pose[inside, 2:4]):.1f} m/s^2 | {outcomes}')


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
    ap.add_argument('--noise-json', help='local filter: use the noise of an ekf_tune.py result (its "noise" entry)')
    ap.add_argument('--reference-dir', help='local filter: score against the poses in <dir>/<bag name>.csv (t x y yaw), '
                    'e.g. from offline_slam_poses.py, instead of the bag\'s own /car_state/pose')
    ap.add_argument('--reference-window', type=float, default=0.1,
                    help='half-width (s) of the linear fit that turns the reference poses into velocities; a noisier pose '
                    'needs a wider one (0.3 for the offline-SLAM pose)')
    ap.add_argument('--global', dest='global_filter', action='store_true',
                    help='replay the global filter: the same model plus the SLAM pose')
    ap.add_argument('--pose-dir', help='global: a directory with one <bag name>.csv / .npy of t x y yaw per bag, '
                    'replacing the pose in the bag (e.g. Cartographer re-run offline)')
    ap.add_argument('--pose-latency', type=float, default=0.0, help='global, --pose-dir: delay from stamp to arrival (s)')
    ap.add_argument('--slam-r', type=float, nargs=3, default=defaults.SLAM['R_slam_pose'], metavar=('X', 'Y', 'YAW'),
                    help='global: pose variances')
    ap.add_argument('--slam-delay', type=float, nargs='+', default=[defaults.SLAM['pose_delay']],
                    help='global: pose delay in s; several values replay each in turn')
    ap.add_argument('--slam-gate', type=float, default=defaults.SLAM['pose_gate'])
    ap.add_argument('--estimate-delay', action='store_true',
                    help='global: replay a grid of pose delays (-20..120 ms unless --slam-delay is given) and report '
                    'the one under which the pose innovations are most likely')
    args = ap.parse_args()
    if args.estimate_delay and args.slam_delay == [defaults.SLAM['pose_delay']]:
        args.slam_delay = [round(d, 3) for d in np.arange(-0.02, 0.1201, 0.01)]
    config = FilterConfig.for_global() if args.global_filter else FilterConfig()
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
    by_delay = {}
    for path in args.bags:
        bag = read_bag(path)
        name = Path(path).name.replace('sysid_', '')
        if args.reference_dir and not args.global_filter:
            poses = load_pose_file(Path(args.reference_dir) / (Path(path).name + '.csv'))
            bag['odom'] = np.column_stack([poses[:, 1], poses[:, 2], poses[:, 3], poses[:, 4], np.zeros((len(poses), 3))])
        if not args.global_filter:
            rows, stats = replay(bag, config, engine, reference_window=args.reference_window)
            summarize(name, rows, stats)
            all_e.append(rows[:, 1:4] - rows[:, 4:7])
            continue
        pose = bag['pose']
        if args.pose_dir:
            candidates = [Path(args.pose_dir) / (Path(path).name + ext) for ext in ('.csv', '.npy')]
            found = [c for c in candidates if c.exists()]
            if not found:
                sys.exit(f'no pose file for {Path(path).name} in {args.pose_dir}')
            pose = load_pose_file(found[0], args.pose_latency)
        for delay in args.slam_delay:
            slam = SlamConfig(R=np.diag(args.slam_r), delay=delay, gate=args.slam_gate, pose=pose)
            rows, stats = replay(bag, config, engine, slam)
            summarize(f'{name} d={delay * 1e3:.0f}ms', rows, stats)
            summarize_slam(f'{name} d={delay * 1e3:.0f}ms', rows, stats, pose)
            all_e.append(rows[:, 1:4] - rows[:, 4:7])
            acc = by_delay.setdefault(delay, {'nll': 0.0, 'count': 0, 'nis': [], 'gated': 0})
            acc['nll'] += stats['slam']['nll']
            acc['count'] += stats['slam']['count']
            acc['nis'] += stats['slam']['nis']
            acc['gated'] += stats['slam']['gated']
    e = np.vstack(all_e)
    print('POOLED EKF rmse (vx, vy, wz):', np.sqrt((e ** 2).mean(0)).round(3))
    if len(by_delay) > 1:
        print('\npose delay    pose NLL/sample   NIS /3   gated')
        for delay, acc in by_delay.items():
            print(f'{delay * 1e3:7.0f} ms   {acc["nll"] / max(acc["count"], 1):14.3f}   {np.mean(acc["nis"]):6.2f}   {acc["gated"]:5d}')
        best = min(by_delay, key=lambda d: by_delay[d]['nll'] / max(by_delay[d]['count'], 1))
        print(f'most likely pose delay: {best * 1e3:.0f} ms'
              + ('' if args.pose_dir else '  (the bags\' own pose came from the local filter\'s odometry: this is biased '
                 'towards small delays; use --pose-dir with an offline-SLAM pose)'))


if __name__ == '__main__':
    main()
