"""CasADi symbolic mirror of the simulator's blended single-track model.

The state and force equations intentionally follow
``f1tenth_parallel_sim.f1tenth_sim.dynamics.VehicleDynamics`` (previously
mirrored in NumPy with finite-difference Jacobians). The dynamics are now
built once as a CasADi graph, and both the forward step (``f``) and its
Jacobian (``F``, ``HJacobian_*``) are derived from that single graph via
``ca.jacobian`` -- so there is exactly one definition of the model, and the
Jacobians can never drift out of sync with what is actually stepped forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np


GRAVITY = 9.81
STATE_DIM = 7


@dataclass(frozen=True)
class VehicleModelParams:
    """Nominal vehicle and tire parameters used by the simulator model."""

    m: float
    I_z: float
    l_f: float
    l_r: float
    h_cg: float
    Bf: float
    Cf: float
    Df: float
    Ef: float
    Br: float
    Cr: float
    Dr: float
    Er: float
    front_drive_frac: float = 0.5
    constraint_time_constant: float = 0.03
    blend_center: float = 0.6
    slip_speed_regularizer: float = 0.1
    steer_gain: float = 1.0
    steer_bias: float = 0.0
    steer_delay: float = 0.0
    steer_latency: float = 0.1
    drive_gain: float = 3.0
    brake_gain: float = 5.0

    @property
    def wheelbase(self) -> float:
        return self.l_f + self.l_r


class CommandDelay:
    """Time-stamped command history that returns the command as it was ``delay`` seconds ago.

    The simulator delays the steering command by ``steer_delay`` before it reaches the first-order
    steering lag (``BatchedSim._delayed_steer``). The command is a known input, so the EKF does not need
    delay states: it looks the delayed command up here and feeds it to the model as ``u[1]``. Values are
    linearly interpolated, held at the newest command, and before the first sample the first value is used.
    """

    def __init__(self, delay: float, horizon: float = 1.0) -> None:
        self.delay = float(delay)
        self.horizon = max(float(horizon), self.delay + 0.1)
        self._t: list[float] = []
        self._v: list[float] = []

    def push(self, t: float, value: float) -> None:
        if self._t and t <= self._t[-1]:
            return
        self._t.append(float(t))
        self._v.append(float(value))
        cut = t - self.horizon
        while len(self._t) > 2 and self._t[1] < cut:
            del self._t[0], self._v[0]

    def at(self, t: float) -> float:
        """The command in effect at the plant at time ``t``, i.e. the one issued at ``t - delay``."""
        if not self._t:
            return 0.0
        return float(np.interp(t - self.delay, self._t, self._v))


def _magic_formula(alpha, B, C, D, E, fz):
    b_alpha = B * alpha
    return D * fz * ca.sin(C * ca.atan(b_alpha - E * (b_alpha - ca.atan(b_alpha))))


def _wrap_angle(angle):
    """Symbolic angle wrap to (-pi, pi], matching the old ``normalize_angle``."""
    return ca.atan2(ca.sin(angle), ca.cos(angle))


class SimulatorSingleTrackModel:
    """CasADi-symbolic, simulator-compatible seven-state single-track model.

    State x = [x, y, yaw, vx, vy, yaw_rate, steer]
    Control u = [speed_command, steer_command_delayed]

    The dynamic/kinematic blending, tire forces, load transfer and
    front-drive force split match the parallel simulator, as does the
    actuator model (``f1tenth_sim.actuators.ActuatorModel``): the steering
    command, already delayed by ``steer_delay``, maps linearly (gain, bias)
    to a road-wheel target that the wheel follows through a first-order lag
    ``steer_latency``, and the drive effort per unit mass is a proportional
    speed loop ``gain * (speed_command - vx)`` with separate drive/brake
    gains. The delay is not in the graph: ``u[1]`` is the steering command
    *as delayed by the caller* (see :class:`CommandDelay`), which is exact
    because the command is a known input. The command rate/range limits
    live in the actuation limiter, not here.
    """

    dim_x = STATE_DIM
    dim_u = 2

    def __init__(self, params: VehicleModelParams, process_noise: np.ndarray) -> None:
        self.params = params
        self.Q = np.asarray(process_noise, dtype=float)
        if self.Q.shape != (STATE_DIM, STATE_DIM):
            raise ValueError(f'process_noise must be a {STATE_DIM}x{STATE_DIM} matrix')
        self.x_0 = np.zeros(STATE_DIM)
        self.P_0 = np.eye(STATE_DIM) * 10.0

        p = params
        inv_tau = 1.0 / max(p.steer_latency, 1e-4)

        # CasADi symbols
        x_sym = ca.MX.sym('x', self.dim_x)
        u_sym = ca.MX.sym('u', self.dim_u)
        dt = ca.MX.sym('dt')

        # for readability
        x = x_sym[0]
        y = x_sym[1]
        yaw = x_sym[2]
        vx = x_sym[3]
        vy = x_sym[4]
        yaw_rate = x_sym[5]
        steer = x_sym[6]
        vel_cmd = u_sym[0]
        steer_cmd = u_sym[1]

        # Actuators: first-order steering lag toward the (already delayed) command, proportional speed loop.
        # `accel` is drive effort per unit mass (a force request, F = m * accel), not physical acceleration.
        steer_target = p.steer_gain * steer_cmd + p.steer_bias
        steer_rate = (steer_target - steer) * inv_tau
        accel = ca.if_else(vel_cmd**2 > vx**2, p.drive_gain, p.brake_gain) * (vel_cmd - vx)

        # Lateral tire forces (simulator sign conventions).
        sin_steer, cos_steer = ca.sin(steer), ca.cos(steer)
        vy_front_body = vy + p.l_f * yaw_rate
        vx_front = vx * cos_steer + vy_front_body * sin_steer
        vy_front = -vx * sin_steer + vy_front_body * cos_steer
        alpha_f = -ca.atan2(vy_front, ca.sqrt(vx_front**2 + p.slip_speed_regularizer**2))
        alpha_r = -ca.atan2(vy - p.l_r * yaw_rate, ca.sqrt(vx**2 + p.slip_speed_regularizer**2))
        fzf = p.m * (-accel * p.h_cg + GRAVITY * p.l_r) / p.wheelbase
        fzr = p.m * (accel * p.h_cg + GRAVITY * p.l_f) / p.wheelbase
        fyf = _magic_formula(alpha_f, p.Bf, p.Cf, p.Df, p.Ef, fzf)
        fyr = _magic_formula(alpha_r, p.Br, p.Cr, p.Dr, p.Er, fzr)

        # Kinematic and dynamic single-track continuous-time derivatives.
        yaw_rate_star = vx * ca.tan(steer) / p.wheelbase
        vy_star = p.l_r * yaw_rate_star
        yaw_rate_star_dot = (accel * ca.tan(steer) + vx * steer_rate / cos_steer**2) / p.wheelbase
        vy_star_dot = p.l_r * yaw_rate_star_dot
        kin = ca.vertcat(
            vx * ca.cos(yaw) - vy * ca.sin(yaw),
            vx * ca.sin(yaw) + vy * ca.cos(yaw),
            yaw_rate,
            accel,
            vy_star_dot + (vy_star - vy) / p.constraint_time_constant,
            yaw_rate_star_dot + (yaw_rate_star - yaw_rate) / p.constraint_time_constant,
            steer_rate,
        )

        drive_force = p.m * accel
        fxf = p.front_drive_frac * drive_force
        fxr = (1.0 - p.front_drive_frac) * drive_force
        sum_fx = fxr + fxf * cos_steer - fyf * sin_steer
        sum_fy = fyr + fxf * sin_steer + fyf * cos_steer
        sum_mz = p.l_f * (fxf * sin_steer + fyf * cos_steer) - p.l_r * fyr
        dyn = ca.vertcat(
            kin[0], kin[1], yaw_rate,
            sum_fx / p.m + yaw_rate * vy,
            sum_fy / p.m - yaw_rate * vx,
            sum_mz / p.I_z,
            steer_rate,
        )

        weight_dynamic = vx**2 / (vx**2 + p.blend_center**2)
        derivatives = (1.0 - weight_dynamic) * kin + weight_dynamic * dyn

        ################
        # System Model #
        ################
        x_next_raw = x_sym + dt * derivatives
        x_next = ca.vertcat(
            x_next_raw[0], x_next_raw[1], _wrap_angle(x_next_raw[2]),
            x_next_raw[3], x_next_raw[4], x_next_raw[5], x_next_raw[6],
        )
        self.f = ca.Function('f', [x_sym, u_sym, dt], [x_next])
        self.F = ca.Function('F', [x_sym, u_sym, dt], [ca.jacobian(x_next, x_sym)])

        #####################
        # Measurement Model #
        #####################
        ax = derivatives[3] - yaw_rate * vy
        ay = derivatives[4] + yaw_rate * vx
        h_imu_sym = ca.vertcat(yaw, yaw_rate, ax, ay)
        _h_imu = ca.Function('h_imu', [x_sym, u_sym], [h_imu_sym])
        _H_imu = ca.Function('H_imu', [x_sym, u_sym], [ca.jacobian(h_imu_sym, x_sym)])
        self.Hx_imu = lambda state, u: np.array(_h_imu(state, u)).flatten()
        self.HJacobian_imu = lambda state, u: np.array(_H_imu(state, u)).reshape(4, self.dim_x)

        H_vesc = np.eye(6, self.dim_x)
        self.Hx_vesc = lambda state: H_vesc @ np.asarray(state, dtype=float).flatten()
        self.HJacobian_vesc = lambda state: H_vesc

    def predict(self, ekf, dt: float, u: np.ndarray) -> None:
        """Predict a filterpy-style EKF object exposing ``x`` and ``P``."""
        x_flat = np.asarray(ekf.x, dtype=float).flatten()
        x_new = np.array(self.f(x_flat, u, dt)).flatten()
        F = np.array(self.F(x_flat, u, dt)).reshape(self.dim_x, self.dim_x)
        ekf.x = x_new
        ekf.P = F @ ekf.P @ F.T + self.Q * dt

    @staticmethod
    def get_odom_state(x: np.ndarray) -> np.ndarray:
        # x = [x, y, yaw, vx, vy, yaw_rate, steer]
        # state = [x, y, yaw, vx, vy, yaw_rate]
        return np.asarray(x, dtype=float)[:6]
