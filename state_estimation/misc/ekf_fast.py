"""The EKF replay as one compiled CasADi call per bag, for fitting the filters' noise (see ``ekf_tune.py``).

``ekf_replay.replay`` loops over the bag's 10 ms ticks in Python, around ten CasADi calls per tick. Here one tick
(predict, IMU update, VESC update and the accelerometer-lag bookkeeping) is a single symbolic function, and ``fold``
runs it over the whole bag in C++. The ticks' inputs (which command is delayed to which substep, whether a sensor is
fresh, its value) do not depend on the noise, so they are prepared once per bag in numpy.

It returns exactly what ``replay`` scores: the Gaussian NLL of the IMU innovations (all four components) and of the
VESC speed and yaw rate, after ``WARMUP``. ``replay`` stays the reference implementation; the tests compare the two.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ekf_replay import HZ, WARMUP, FilterConfig, wrap
from state_estimation.ekf_node import defaults
from state_estimation.ekf_node.models import CommandDelay

SUBSTEPS = int(np.ceil(1.0 / HZ / defaults.NOISE['predict_substep']))
N_STATE = 7 + 49 + 2 + 1  # x, P, the lagged predicted acceleration, whether that has a value yet
N_TICK = 2 * SUBSTEPS + 2 + 1 + 4 + 1 + 6 + 1  # delayed commands, command now, IMU fresh + value, VESC fresh + value, scored
N_PARAM = 7 + 4 + 6 + 2 + 1
N_ACC = 4


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


def parameter_vector(config: FilterConfig):
    weight = 1.0 - np.exp(-(1.0 / HZ) / config.accel_lag) if config.accel_lag > 0 else 1.0
    return np.concatenate([np.diag(config.Q), np.diag(config.R_imu), np.diag(config.R_vesc), config.accel_bias, [weight]])


def prepare(bag, config: FilterConfig):
    """The noise-independent inputs of every tick of ``bag``: ``(ticks, t0, imu)``.

    Mirrors the loop in ``ekf_replay.replay`` (commands and sensors are sample-and-hold, the first tick only
    collects, a sensor sample is fresh until the tick that consumes it).
    """
    imu, cmd, odom = bag['imu'], bag['cmd'], bag['odom']
    vesc = bag['vesc'] if len(bag['vesc']) else odom
    t0 = max(imu[0, 0], cmd[0, 0], odom[0, 0], vesc[0, 0])
    t1 = min(imu[-1, 0], cmd[-1, 0], odom[-1, 0], vesc[-1, 0])
    yaw0 = imu[np.searchsorted(imu[:, 0], t0), 1]
    delay = CommandDelay(config.steer_delay)
    speed_delay = CommandDelay(config.speed_delay)
    h = 1.0 / HZ / SUBSTEPS
    ticks = []
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

        ticks.append(column)

    return np.array(ticks).T, t0, imu


def initial_state(model, config: FilterConfig, imu, t0):
    x = np.zeros(7)
    x[5] = imu[np.searchsorted(imu[:, 0], t0), 2]
    P = model.P_0.copy()
    P[2, 2], P[5, 5] = config.R_imu[0, 0], config.R_imu[1, 1]
    return np.concatenate([x, P.reshape(-1, order='F'), np.zeros(2), [0.0]])


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
