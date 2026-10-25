"""The EKF replay as one compiled CasADi call per bag, for fitting the filters' noise (see ``ekf_tune.py``).

``ekf_replay.replay`` loops over the bag's 10 ms ticks in Python, around ten CasADi calls per tick. Here one tick
(predict, IMU update, VESC update, the accelerometer-lag bookkeeping and, for the global filter, the SLAM pose update
with its delay compensation, gating and re-seeding) is a single symbolic function, and ``fold`` runs it over the
whole bag in C++. The ticks' inputs (which command is delayed to which substep, whether a sensor is fresh, its value,
which two past estimates a pose is compared with) do not depend on the noise, so they are prepared once per bag in
numpy.

It returns exactly what ``replay`` scores: the Gaussian NLL of the IMU innovations (all four components), of the VESC
speed and yaw rate and, for the global filter, of the SLAM pose, after ``WARMUP``. ``replay`` stays the reference
implementation; the tests compare the two.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ekf_replay import HZ, WARMUP, FilterConfig, SlamConfig, wrap
from state_estimation.ekf_node import defaults
from state_estimation.ekf_node.models import CommandDelay

SUBSTEPS = int(np.ceil(1.0 / HZ / defaults.NOISE['predict_substep']))
N_STATE = 7 + 49 + 2 + 1  # x, P, the lagged predicted acceleration, whether that has a value yet
N_TICK = 2 * SUBSTEPS + 2 + 1 + 4 + 1 + 6 + 1  # delayed commands, command now, IMU fresh + value, VESC fresh + value, scored
N_PARAM = 7 + 4 + 6 + 2 + 1
K_HIST = 16  # past estimates kept for the pose lookup: 160 ms, far more than a pose's age plus its delay
N_STATE_G = N_STATE + 3 * K_HIST + 1  # + the pose history (newest first) and the count of consecutive gated poses
N_TICK_G = N_TICK + 1 + 3 + 2 * K_HIST + 1  # + pose fresh, pose, the two history slots it falls between, the fraction
N_PARAM_G = N_PARAM + 3 + 2  # + the pose variances, the gate, the re-seed streak
N_ACC = 4
N_ACC_G = 12
_IMU_PARAM = slice(7, 11)


def _wrap(a):
    shifted = a + np.pi
    return shifted - 2 * np.pi * ca.floor(shifted / (2 * np.pi)) - np.pi


class _Functions:
    """The model's CasADi functions as SX, which evaluate several times faster than MX."""

    def __init__(self, engine):
        model = engine.model
        self.f = model.f.expand()
        self.jac = model.F.expand()
        self.h_imu = model.casadi_functions['h_imu'].expand()
        self.H_imu = model.casadi_functions['H_imu'].expand()
        self.l_r = model.params.l_r
        self.h = 1.0 / HZ / SUBSTEPS


def _core(fn, x, P, lag, valid, d, p):
    """Predict, IMU update, VESC update and the accelerometer-lag bookkeeping of one tick.

    Returns the new state ``(x, P, lag)`` and the scored terms ``(nll_imu, nll_vesc, nis_imu, nis_vesc)``.
    """
    u_sub = [d[2 * k:2 * k + 2] for k in range(SUBSTEPS)]
    o = 2 * SUBSTEPS
    u_now, f_imu, imu_raw, f_vesc, z_vesc, scored = d[o:o + 2], d[o + 2], d[o + 3:o + 7], d[o + 7], d[o + 8:o + 14], d[o + 14]
    q, r_imu, r_vesc, bias, weight = p[:7], p[7:11], p[11:17], p[17:19], p[19]

    for u in u_sub:  # predict
        F = fn.jac(x, u, fn.h)
        x = fn.f(x, u, fn.h)
        P = F @ P @ F.T + ca.diag(q) * fn.h

    def update(x, P, y, H, R, fresh):
        """Joseph-form update scaled by ``fresh`` (0 leaves the state alone). Returns the state, covariance, S."""
        S = H @ P @ H.T + R
        K = fresh * ca.solve(S, H @ P).T
        IKH = ca.SX.eye(7) - K @ H
        return x + K @ y, IKH @ P @ IKH.T + K @ R @ K.T, S

    # IMU: yaw, yaw rate, ax, ay against the prediction, the accelerations as the lagged accelerometer sees them
    hx = fn.h_imu(x, u_now)
    predicted = ca.vertcat(hx[0], hx[1], ca.if_else(valid > 0.5, lag, hx[2:4]))
    z = ca.vertcat(imu_raw[0], imu_raw[1], imu_raw[2] - bias[0], imu_raw[3] - bias[1])
    y = z - predicted
    y = ca.vertcat(_wrap(y[0]), y[1:])
    x, P, S = update(x, P, y, fn.H_imu(x, u_now), ca.diag(r_imu), f_imu)
    quad = ca.dot(y, ca.solve(S, y))
    keep = f_imu * scored
    nll_imu = keep * 0.5 * (quad + ca.log(ca.det(S)) + 4 * np.log(2 * np.pi))
    nis_imu = keep * quad

    # VESC: [x, y, yaw, vx, vy, yaw rate] = the first six states; only vx and the yaw rate are scored
    y = z_vesc - x[:6]
    H = ca.horzcat(ca.SX.eye(6), ca.SX.zeros(6, 1))
    x, P, S = update(x, P, y, H, ca.diag(r_vesc), f_vesc)
    quad = ca.dot(y, ca.solve(S, y))
    sub = [3, 5]
    S2, y2 = S[sub, sub], y[sub]
    keep = f_vesc * scored
    nll_vesc = keep * 0.5 * (ca.dot(y2, ca.solve(S2, y2)) + ca.log(S2[0, 0] * S2[1, 1] - S2[0, 1] * S2[1, 0])
                             + 2 * np.log(2 * np.pi))
    nis_vesc = keep * quad
    return x, P, lag, (nll_imu, nll_vesc, nis_imu, nis_vesc)


def _lag_update(fn, x, d, lag, valid, p):
    """The accelerometer lag follows the prediction from the updated state."""
    u_now = d[2 * SUBSTEPS:2 * SUBSTEPS + 2]
    inst = fn.h_imu(x, u_now)[2:4]
    return ca.if_else(valid > 0.5, lag + p[19] * (inst - lag), inst)


_JIT_DIRS = []  # temporary compile directories, kept until the process exits (the compiled code is loaded from them)


def _jit(name, inputs, outputs):
    """Compiled to machine code (gcc, once per process); the symbolic evaluation is several times slower.

    CasADi writes its C source and the library into the working directory, so the compile runs in a temporary one.
    """
    import atexit
    import os
    import shutil
    import tempfile

    workdir = tempfile.mkdtemp(prefix='ekf_jit_')
    _JIT_DIRS.append(workdir)
    if len(_JIT_DIRS) == 1:
        atexit.register(lambda: [shutil.rmtree(d, ignore_errors=True) for d in _JIT_DIRS])
    previous = os.getcwd()
    try:
        os.chdir(workdir)
        fn = ca.Function(name, inputs, outputs, {'jit': True, 'compiler': 'shell', 'jit_options': {'flags': ['-O2'], 'compiler': 'gcc'}})
        fn(*[np.zeros(i.numel()) for i in inputs])  # compile now, or fall back
        return fn
    except Exception:
        return ca.Function(name, inputs, outputs)
    finally:
        os.chdir(previous)


class FastFilter:
    """The local filter's tick function, built once from a replay ``Engine``'s model."""

    def __init__(self, engine):
        fn = _Functions(engine)
        s, d, p = ca.SX.sym('s', N_STATE), ca.SX.sym('d', N_TICK), ca.SX.sym('p', N_PARAM)
        x, P, lag, valid = s[:7], ca.reshape(s[7:56], 7, 7), s[56:58], s[58]
        x, P, _, terms = _core(fn, x, P, lag, valid, d, p)
        lag_next = _lag_update(fn, x, d, lag, valid, p)
        s_next = ca.vertcat(x, ca.reshape(P, 49, 1), lag_next, 1)
        out = ca.vertcat(*terms, x[3], x[4], x[5])
        self.tick = ca.Function('tick', [s, d, p], [s_next, out])  # symbolic: per-tick outputs, for diagnostics

        # Scoring only needs the sums, so they ride along in the carried state, and so do the (constant) noise
        # parameters: ``fold`` then returns just the final state and the scan reads one input per tick.
        sa = ca.SX.sym('sa', N_STATE + N_ACC + N_PARAM)
        s_in, acc, p_in = sa[:N_STATE], sa[N_STATE:N_STATE + N_ACC], sa[N_STATE + N_ACC:]
        s_out, tick_out = ca.substitute([s_next, out], [s, p], [s_in, p_in])
        self.tick_sum = _jit('tick_sum', [sa, d], [ca.vertcat(s_out, acc + tick_out[:N_ACC], p_in)])
        self._scans, self._folds = {}, {}

    def scan(self, n):
        if n not in self._scans:
            self._scans[n] = self.tick.mapaccum('scan', n)
        return self._scans[n]

    def fold(self, n):
        if n not in self._folds:
            self._folds[n] = self.tick_sum.fold(n)
        return self._folds[n]


class FastGlobalFilter:
    """The global filter's tick: the local one plus the SLAM pose update (``SlamPoseUpdater``), carrying the pose history."""

    def __init__(self, engine):
        fn = _Functions(engine)
        l_r = fn.l_r
        n_acc = N_ACC_G
        sa = ca.SX.sym('sa', N_STATE_G + n_acc + N_PARAM_G)
        d = ca.SX.sym('d', N_TICK_G)
        s, acc, p = sa[:N_STATE_G], sa[N_STATE_G:N_STATE_G + n_acc], sa[N_STATE_G + n_acc:]
        x, P, lag, valid = s[:7], ca.reshape(s[7:56], 7, 7), s[56:58], s[58]
        buf = ca.reshape(s[N_STATE:N_STATE + 3 * K_HIST], 3, K_HIST)  # rows x, y, yaw; column 0 is the newest estimate
        streak = s[N_STATE + 3 * K_HIST]
        scored = d[N_TICK - 1]
        o = N_TICK
        f_p, z_p = d[o], d[o + 1:o + 4]
        a_slot, b_slot, frac = d[o + 4:o + 4 + K_HIST], d[o + 4 + K_HIST:o + 4 + 2 * K_HIST], d[o + 4 + 2 * K_HIST]
        r_slam, gate, reseed_after = ca.diag(p[N_PARAM:N_PARAM + 3]), p[N_PARAM + 3], p[N_PARAM + 4]

        x, P, _, terms = _core(fn, x, P, lag, valid, d[:N_TICK], p[:N_PARAM])
        lag_next = _lag_update(fn, x, d, lag, valid, p)

        # the pose the filter estimated when the SLAM pose was taken: between two history slots
        row = lambda r: buf[r, :]
        xa, xb = ca.dot(a_slot, row(0).T), ca.dot(b_slot, row(0).T)
        ya, yb = ca.dot(a_slot, row(1).T), ca.dot(b_slot, row(1).T)
        yawa, yawb = ca.dot(a_slot, row(2).T), ca.dot(b_slot, row(2).T)
        predicted = ca.vertcat((1 - frac) * xa + frac * xb, (1 - frac) * ya + frac * yb, yawa + frac * _wrap(yawb - yawa))
        innovation = ca.vertcat(z_p[0] - predicted[0], z_p[1] - predicted[1], _wrap(z_p[2] - predicted[2]))
        yaw = x[2]
        H = ca.SX.zeros(3, 7)
        H[0, 0], H[0, 2] = 1.0, l_r * ca.sin(yaw)
        H[1, 1], H[1, 2] = 1.0, -l_r * ca.cos(yaw)
        H[2, 2] = 1.0
        S = H @ P @ H.T + r_slam
        chi2 = ca.dot(innovation, ca.solve(S, innovation))
        gated = ca.if_else(chi2 > gate, 1.0, 0.0)
        streak_next = streak + 1
        reseed = f_p * gated * ca.if_else(streak_next >= reseed_after, 1.0, 0.0)
        update = f_p * (1.0 - gated)
        K = update * ca.solve(S, H @ P).T
        IKH = ca.SX.eye(7) - K @ H
        x_upd, P_upd = x + K @ innovation, IKH @ P @ IKH.T + K @ r_slam @ K.T

        # re-seeding: the state takes the pose, the pose block of P its variance, and the history follows
        x_seed = ca.vertcat(z_p[0] + l_r * ca.cos(z_p[2]), z_p[1] + l_r * ca.sin(z_p[2]), z_p[2], x[3:7])
        keep_mask = np.ones((7, 7))
        keep_mask[:3, :] = 0
        keep_mask[:, :3] = 0
        pad = ca.SX.zeros(7, 7)
        pad[:3, :3] = r_slam
        P_seed = P * ca.DM(keep_mask) + pad
        old = ca.vertcat(x[0] - l_r * ca.cos(yaw), x[1] - l_r * ca.sin(yaw), yaw)
        c_o, s_o, c_n, s_n = ca.cos(old[2]), ca.sin(old[2]), ca.cos(z_p[2]), ca.sin(z_p[2])
        dx, dy = row(0) - old[0], row(1) - old[1]
        rx, ry = c_o * dx + s_o * dy, -s_o * dx + c_o * dy
        moved = ca.vertcat(z_p[0] + c_n * rx - s_n * ry, z_p[1] + s_n * rx + c_n * ry, z_p[2] + _wrap(row(2) - old[2]))
        buf = ca.if_else(reseed > 0.5, moved, buf)
        x = ca.if_else(reseed > 0.5, x_seed, x_upd)
        P = ca.if_else(reseed > 0.5, P_seed, P_upd)
        streak_out = ca.if_else(f_p > 0.5, ca.if_else(gated > 0.5, ca.if_else(streak_next >= reseed_after, 0.0, streak_next), 0.0), streak)

        nll_pose = update * scored * 0.5 * (chi2 + ca.log(ca.det(S)) + 3 * np.log(2 * np.pi))
        counted = f_p * scored * (1.0 - reseed)  # replay scores the NIS of updated and gated poses, not re-seeded ones
        nll_clipped = f_p * scored * 0.5 * (ca.fmin(chi2, gate) + ca.log(ca.det(S)) + 3 * np.log(2 * np.pi))
        terms_g = ca.vertcat(*terms, nll_pose, counted * chi2, update * scored, counted, f_p * gated, reseed, nll_clipped, f_p * scored)

        pose_now = ca.vertcat(x[0] - l_r * ca.cos(x[2]), x[1] - l_r * ca.sin(x[2]), x[2])
        buf = ca.horzcat(pose_now, buf[:, :K_HIST - 1])
        s_next = ca.vertcat(x, ca.reshape(P, 49, 1), lag_next, 1, ca.reshape(buf, 3 * K_HIST, 1), streak_out)
        self.tick_sum = _jit('tick_sum_g', [sa, d], [ca.vertcat(s_next, acc + terms_g, p)])
        self._folds = {}

    def fold(self, n):
        if n not in self._folds:
            self._folds[n] = self.tick_sum.fold(n)
        return self._folds[n]


def parameter_vector(config: FilterConfig):
    weight = 1.0 - np.exp(-(1.0 / HZ) / config.accel_lag) if config.accel_lag > 0 else 1.0
    return np.concatenate([np.diag(config.Q), np.diag(config.R_imu), np.diag(config.R_vesc), config.accel_bias, [weight]])


def global_parameter_vector(config: FilterConfig, slam: SlamConfig):
    r_imu = np.diag(config.R_imu).copy()
    r_imu[0] = 1e6  # the IMU heading is not fused by the global filter
    base = parameter_vector(config)
    base[_IMU_PARAM] = r_imu
    return np.concatenate([base, np.diag(slam.R), [slam.gate, slam.reseed_after]])


def prepare(bag, config: FilterConfig, slam: SlamConfig | None = None):
    """The noise-independent inputs of every tick of ``bag``: ``(ticks, t0, imu)``.

    Mirrors the loop in ``ekf_replay.replay`` (commands and sensors are sample-and-hold, the first tick only
    collects, a sensor sample is fresh until the tick that consumes it). With ``slam`` it also resolves, for every
    pose, which two past estimates it is compared with.
    """
    imu, cmd, odom = bag['imu'], bag['cmd'], bag['odom']
    vesc = bag['vesc'] if len(bag['vesc']) else odom
    t0 = max(imu[0, 0], cmd[0, 0], odom[0, 0], vesc[0, 0])
    t1 = min(imu[-1, 0], cmd[-1, 0], odom[-1, 0], vesc[-1, 0])
    if slam is not None:
        pose = slam.pose if slam.pose is not None else bag['pose']
        t0 = max(t0, pose[0, 0])
        i_pose, last_stamp, pose_z = 1, pose[0, 1], None
    yaw0 = imu[np.searchsorted(imu[:, 0], t0), 1]
    delay = CommandDelay(config.steer_delay)
    speed_delay = CommandDelay(config.speed_delay)
    h = 1.0 / HZ / SUBSTEPS
    ticks, times = [], []
    i_imu = i_cmd = i_vesc = 0
    speed_cmd = 0.0
    imu_z, imu_fresh = None, False
    for t in np.arange(t0, t1, 1.0 / HZ):
        while i_cmd < len(cmd) and cmd[i_cmd, 0] <= t:
            speed_cmd = cmd[i_cmd, 1]
            delay.push(cmd[i_cmd, 0], cmd[i_cmd, 2])
            speed_delay.push(cmd[i_cmd, 0], cmd[i_cmd, 1])
            i_cmd += 1
        while i_imu < len(imu) and imu[i_imu, 0] <= t:
            _, yaw, wz, ax, ay = imu[i_imu]
            imu_z = np.array([wrap(yaw - yaw0), wz, -ay, ax])  # the bias is subtracted inside the filter
            i_imu += 1
            imu_fresh = True
        vesc_z = None
        while i_vesc < len(vesc) and vesc[i_vesc, 0] <= t:
            vesc_z = np.array([vesc[i_vesc, 1], vesc[i_vesc, 2], vesc[i_vesc, 3],
                               vesc[i_vesc, 4] + config.speed_bias, 0.0, vesc[i_vesc, 6]])
            i_vesc += 1
        if t == t0:
            continue
        column = []
        for k in range(SUBSTEPS):
            column += [speed_delay.at(t - 1.0 / HZ + (k + 0.5) * h), delay.at(t - 1.0 / HZ + (k + 0.5) * h)]
        column += [speed_delay.at(t), delay.at(t)]
        use_imu = imu_fresh and imu_z is not None
        column += [1.0 if use_imu else 0.0, *(imu_z if imu_z is not None else np.zeros(4))]
        column += [1.0 if vesc_z is not None else 0.0, *(vesc_z if vesc_z is not None else np.zeros(6))]
        column += [1.0 if t - t0 > WARMUP else 0.0]
        imu_fresh = False if use_imu else imu_fresh

        if slam is not None:
            while i_pose < len(pose) and pose[i_pose, 0] <= t:
                _, stamp, px, py, pyaw = pose[i_pose]
                i_pose += 1
                if stamp > last_stamp:  # a republished stamp carries no new scan match
                    last_stamp, pose_z = stamp, (np.array([px, py, wrap(pyaw)]), stamp)
            extra = np.zeros(1 + 3 + 2 * K_HIST + 1)
            if pose_z is not None:
                z, stamp = pose_z
                pose_z = None
                moment = stamp - slam.delay
                n_hist = len(times)
                if n_hist >= 2 and moment >= times[0]:  # else the history does not reach back: skipped
                    j = int(np.searchsorted(times, moment))
                    if j >= n_hist:
                        a = b = n_hist - 1
                        frac = 0.0
                    elif j == 0:
                        a = b = 0
                        frac = 0.0
                    else:
                        a, b = j - 1, j
                        frac = (moment - times[a]) / (times[b] - times[a])
                    slot_a, slot_b = n_hist - 1 - a, n_hist - 1 - b  # column 0 of the history is the newest record
                    if max(slot_a, slot_b) < K_HIST:
                        extra[0] = 1.0
                        extra[1:4] = z
                        extra[4 + slot_a] = 1.0
                        extra[4 + K_HIST + slot_b] = 1.0
                        extra[4 + 2 * K_HIST] = frac
            column += list(extra)
        times.append(t)
        ticks.append(column)

    return np.array(ticks).T, t0, imu


def initial_state(model, config: FilterConfig, imu, t0):
    x = np.zeros(7)
    x[5] = imu[np.searchsorted(imu[:, 0], t0), 2]
    P = model.P_0.copy()
    P[2, 2], P[5, 5] = config.R_imu[0, 0], config.R_imu[1, 1]
    return np.concatenate([x, P.reshape(-1, order='F'), np.zeros(2), [0.0]])


def initial_state_global(model, slam: SlamConfig, first_pose):
    """The global filter starts at the first SLAM pose (``SlamPoseUpdater.seed``)."""
    l_r = model.params.l_r
    x, y, yaw = first_pose
    xs = np.zeros(7)
    xs[:3] = [x + l_r * np.cos(yaw), y + l_r * np.sin(yaw), yaw]
    P = model.P_0.copy()
    P[:3, :] = 0.0
    P[:, :3] = 0.0
    P[:3, :3] = slam.R
    return np.concatenate([xs, P.reshape(-1, order='F'), np.zeros(2), [0.0], np.zeros(3 * K_HIST + 1)])


class FastReplay:
    """Noise-independent inputs of one bag, ready to be scored under any ``FilterConfig``."""

    def __init__(self, bag, config, engine, fast: FastFilter):
        self.fast = fast
        self.model = engine.model
        self.ticks, self.t0, imu = prepare(bag, config)
        self.imu = imu
        self.scored = self.ticks[-1] > 0.5
        self.counts = {'imu': int((self.ticks[2 * SUBSTEPS + 2] * self.scored).sum()),
                       'vesc': int((self.ticks[2 * SUBSTEPS + 7] * self.scored).sum())}

    def run(self, config: FilterConfig):
        """``{'nll': {...}, 'nis_sum': {...}, 'count': {...}}`` for ``config``: what the search scores."""
        n = self.ticks.shape[1]
        s0 = np.concatenate([initial_state(self.model, config, self.imu, self.t0), np.zeros(N_ACC), parameter_vector(config)])
        final = np.array(self.fast.fold(n)(s0, self.ticks)).ravel()
        acc = final[N_STATE:N_STATE + N_ACC]
        return {'nll': {'imu': float(acc[0]), 'vesc': float(acc[1])}, 'nis_sum': {'imu': float(acc[2]), 'vesc': float(acc[3])},
                'count': dict(self.counts)}

    def trace(self, config: FilterConfig):
        """The same run with per-tick outputs: ``(nll_imu, nll_vesc, nis_imu, nis_vesc, vx, vy, yaw rate)`` by tick."""
        n = self.ticks.shape[1]
        s0 = initial_state(self.model, config, self.imu, self.t0)
        _, out = self.fast.scan(n)(s0, self.ticks, np.tile(parameter_vector(config)[:, None], (1, n)))
        return np.array(out)


class FastGlobalReplay:
    """The global filter on one bag and one SLAM pose track: the pose delay and the pose's arrival times are fixed here."""

    def __init__(self, bag, config, slam: SlamConfig, engine, fast: FastGlobalFilter):
        self.fast = fast
        self.model = engine.model
        self.ticks, self.t0, self.imu = prepare(bag, config, slam)
        pose = slam.pose if slam.pose is not None else bag['pose']
        self.first_pose = [pose[0, 2], pose[0, 3], wrap(pose[0, 4])]
        scored = self.ticks[N_TICK - 1] > 0.5
        self.counts = {'imu': int((self.ticks[2 * SUBSTEPS + 2] * scored).sum()),
                       'vesc': int((self.ticks[2 * SUBSTEPS + 7] * scored).sum())}

    def run(self, config: FilterConfig, slam: SlamConfig):
        """What the search scores, and the pose-fusion diagnostics, for ``config`` and ``slam``'s noise, gate and streak."""
        n = self.ticks.shape[1]
        s0 = np.concatenate([initial_state_global(self.model, slam, self.first_pose), np.zeros(N_ACC_G),
                             global_parameter_vector(config, slam)])
        final = np.array(self.fast.fold(n)(s0, self.ticks)).ravel()
        acc = final[N_STATE_G:N_STATE_G + N_ACC_G]
        # 'nll' / 'count' are what the search scores: every pose, one the gate rejects at the gate (see replay).
        return {'nll': {'imu': float(acc[0]), 'vesc': float(acc[1]), 'pose': float(acc[10])},
                'nis_sum': {'imu': float(acc[2]), 'vesc': float(acc[3]), 'pose': float(acc[5])},
                'count': {**self.counts, 'pose': int(round(acc[11]))},
                'pose_accepted': {'nll': float(acc[4]), 'count': int(round(acc[6]))}, 'pose_nis_count': int(round(acc[7])),
                'gated': int(round(acc[8])), 'reseeds': int(round(acc[9]))}
