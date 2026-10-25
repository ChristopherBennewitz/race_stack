from bisect import bisect_right
from dataclasses import dataclass

import casadi as ca
import numpy as np


class point_mass_model:
    """
    omnidirectional, three-dimensional point-mass motion model

    State x = [x, y, theta, vx, vy, omega]
    Control inputs u = [] (not used)
    """

    def __init__(self, Q):
        ################
        # System Model #
        ################
        self.dim_x = 6
        self.dim_u = 2  # not used here but keep for generic structure
        self.Q = Q

        # Initial Values
        self.x_0 = np.zeros(self.dim_x)
        self.P_0 = np.eye(self.dim_x) * 10
        self.P_0[4][4] = 0.0001

        # CasADi Symbols
        x_sym = ca.MX.sym('x', self.dim_x)
        u_sym = ca.MX.sym('u', self.dim_u)
        dt = ca.MX.sym('dt')

        x = x_sym[0]
        y = x_sym[1]
        psi = x_sym[2]
        vx = x_sym[3]
        vy = x_sym[4]
        vpsi = x_sym[5]
        accel = u_sym[0]
        steer_agnle = u_sym[1]

        # nonlinear transition functions
        fxu = ca.vertcat(
            x + (vx * ca.cos(psi) - vy * ca.sin(psi)) * dt,
            y + (vx * ca.sin(psi) + vy * ca.cos(psi)) * dt,
            psi + vpsi * dt,
            vx,
            vy,
            vpsi
        )
        self.f = ca.Function('f', [x_sym, u_sym, dt], [fxu])
        self.F = ca.Function('F', [x_sym, u_sym, dt], [ca.jacobian(fxu, x_sym)])

        #####################
        # Measurement Model #
        #####################
        H_imu = np.array([[0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]])
        self.Hx_imu = lambda x, u: H_imu @ np.array(x).flatten()
        self.HJacobian_imu = lambda x, u: H_imu

        H_vesc = np.eye(6)
        self.Hx_vesc = lambda x: H_vesc @ np.array(x).flatten()
        self.HJacobian_vesc = lambda x: H_vesc

    def predict(self, ekf, dt, u):
        x_flat = np.array(ekf.x).flatten()

        x_new = np.array(self.f(x_flat, u, dt)).flatten()

        F = np.array(self.F(x_flat, u, dt)).reshape(self.dim_x, self.dim_x)

        ekf.x = x_new
        ekf.P = F @ ekf.P @ F.T + self.Q

    def get_odom_state(self, x):
        return x


class kinematic_bicycle_model:
    """
    Kinematic Bicycle Model Class. Uses model defined in the vehicle_dynamics_update_k function of f1tenth_pysim.dynamic_models_ros

    State vector x = [x, y, steer_angle, velocity, theta, angular_velocity, slip_angle]
    Control input u = [accel, steer_angle]

    """

    def __init__(self, config, Q):
        self.p = config

        ################
        # System Model #
        ################
        self.dim_x = 7
        self.dim_u = 2

        self.Q = Q

        # Initial Values
        self.x_0 = np.zeros(self.dim_x)
        self.P_0 = np.eye(self.dim_x) * 10

        x_sym = ca.MX.sym('x', self.dim_x)
        u_sym = ca.MX.sym('u', self.dim_u)
        dt = ca.MX.sym('dt')

        # for readability
        x = x_sym[0]
        y = x_sym[1]
        steer = x_sym[2]
        v = x_sym[3]
        theta = x_sym[4]
        omega = x_sym[5]
        slip = x_sym[6]

        accel = u_sym[0]
        steer_cmd = u_sym[1]

        fxu = ca.vertcat(
            x + v * ca.cos(theta) * dt,
            y + v * ca.sin(theta) * dt,
            steer_cmd,
            v + accel * dt,
            theta + v * ca.tan(steer) / self.p['wheelbase'] * dt,
            v * ca.tan(steer) / self.p['wheelbase'],
            ca.MX(0.0)
        )

        self.f = ca.Function('f', [x_sym, u_sym, dt], [fxu])
        self.F = ca.Function('F', [x_sym, u_sym, dt], [ca.jacobian(fxu, x_sym)])

        #####################
        # Measurement Model #
        #####################
        h_imu_sym = ca.vertcat(
            theta,
            omega,
            accel,
            0,
        )
        _h_imu = ca.Function
        _h_imu = ca.Function('h_imu', [x_sym, u_sym], [h_imu_sym])
        _H_imu = ca.Function('H_imu', [x_sym, u_sym], [ca.jacobian(h_imu_sym, x_sym)])
        self.Hx_imu = lambda x, u: np.array(_h_imu(x, u)).flatten()
        self.HJacobian_imu = lambda x, u: np.array(_H_imu(x, u)).reshape(4, self.dim_x)

        h_vesc_sym = ca.vertcat(
            x,
            y,
            theta,
            v,
            ca.MX(0.0),
            omega,
        )
        _h_vesc = ca.Function('h_vesc', [x_sym], [h_vesc_sym])
        _H_vesc = ca.Function('H_vesc', [x_sym], [ca.jacobian(h_vesc_sym, x_sym)])
        self.Hx_vesc = lambda x: np.array(_h_vesc(x)).flatten()
        self.HJacobian_vesc = lambda x: np.array(_H_vesc(x)).reshape(6, self.dim_x)

    def predict(self, ekf, dt, u):
        x_flat = np.array(ekf.x).flatten()
        x_new = np.array(self.f(x_flat, u, dt)).flatten()

        F = np.array(self.F(x_flat, u, dt)).reshape(self.dim_x, self.dim_x)

        ekf.x = x_new
        ekf.P = F @ ekf.P @ F.T + self.Q

    def get_odom_state(self, x):
        # x_kin = [x, y, steer_angle, velocity, theta, angular_velocity, slip_angle]
        # state = [x, y, psi, vx, vy, vpsi]
        state = np.zeros(6)
        state[0] = x[0]
        state[1] = x[1]
        state[2] = x[4]

        state[3] = x[3]
        state[4] = 0.0

        state[5] = x[5]

        return state


class single_track_model:
    """
    EKF Model for the single-track model. Uses model defined in the vehicle_dynamics_st_update_pacejka function of f1tenth_pysim.dynamic_models_ros

    State vector x = [x, y, theta, vx, vy, omega]
    Control input u = [accel, steer_angle]

    Threshold value for mixing logic v_b is tuned empirically to 2.05

    """

    def __init__(self, config, Q):
        self.p = config

        ################
        # System Model #
        ################
        self.dim_x = 6
        self.dim_u = 2

        self.Q = Q

        # Initial Values
        self.x_0 = np.zeros(self.dim_x)
        self.P_0 = np.eye(self.dim_x) * 10

        # CasADi symbols
        x_sym = ca.MX.sym('x', self.dim_x)
        u = ca.MX.sym('u', self.dim_u)
        dt = ca.MX.sym('dt')

        # for readability
        x = x_sym[0]
        y = x_sym[1]
        theta = x_sym[2]
        vx = x_sym[3]
        vy = x_sym[4]
        omega = x_sym[5]
        accel = u[0]
        steer = u[1]

        # Compute Forces used in Model (math coppied from original model definition)
        F_zf = self.p['m'] * (-accel * self.p['h_cg'] + 9.81 * self.p['l_r']) / (self.p['l_f'] + self.p['l_r'])
        F_zr = self.p['m'] * (accel * self.p['h_cg'] + 9.81 * self.p['l_f']) / (self.p['l_f'] + self.p['l_r'])

        eps = 1e-4
        v_safe = vx + ca.if_else(vx >= 0, eps, -eps)  # prevent division by 0 later

        alpha_f = ca.atan2((-vy - self.p['l_f'] * omega), v_safe) + steer
        alpha_r = ca.atan2((-vy + self.p['l_r'] * omega), v_safe)

        Fyf = self.p['mu'] * self.p['Df'] * F_zf * ca.sin(
            self.p['Cf'] * ca.atan(
                self.p['Bf'] * alpha_f - self.p['Ef'] * (self.p['Bf'] * alpha_f - ca.atan(
                    self.p['Bf'] * alpha_f))))
        Fyr = self.p['mu'] * self.p['Dr'] * F_zr * ca.sin(
            self.p['Cr'] * ca.atan(
                self.p['Br'] * alpha_r - self.p['Er'] * (self.p['Br'] * alpha_r - ca.atan(
                    self.p['Br'] * alpha_r))))

        Fxr = accel * self.p['m']
        Fxf = ca.MX(0.0)

        fxu_dyn = ca.vertcat(
            x + dt * (vx * ca.cos(theta) - vy * ca.sin(theta)),
            y + dt * (vx * ca.sin(theta) + vy * ca.cos(theta)),
            theta + dt * omega,
            vx + dt * ((Fxr + Fxf * ca.cos(steer) - Fyf * ca.sin(steer) + self.p['m'] * vy * omega) / self.p['m']),
            vy + dt * ((Fyr + Fxf * ca.sin(steer) + Fyf * ca.cos(steer) - self.p['m'] * vx * omega) / self.p['m']),
            omega + dt * ((Fyf * self.p['l_f'] * ca.cos(steer) + Fxf * self.p['l_f']
                          * ca.sin(steer) - Fyr * self.p['l_r']) / self.p['I_z'])
        )

        # kinematic bicycle model uses overall velocity not vx and vy (1e-6 to prevent division by 0 later)
        speed_mag = ca.sqrt(vx**2 + vy**2 + 1e-6)
        speed = ca.sign(vx) * speed_mag
        fxu_kin = ca.vertcat(
            x + vx * ca.cos(theta) * dt,
            y + vx * ca.sin(theta) * dt,
            theta + (speed * ca.tan(steer) / self.p['wheelbase']) * dt,
            vx + accel * dt,
            ca.MX(0.0),
            speed * ca.tan(steer) / self.p['wheelbase']
        )

        # mixing weights
        v_b = 2.05  # empirically tuned variable
        v_s = 1
        w_std = 0.5 * (1 + ca.tanh((speed_mag - v_b) / v_s))
        w_std = ca.if_else(speed_mag < (v_b - 2 * v_s), 0.0, w_std)

        f_mixed = w_std * fxu_dyn + (1 - w_std) * fxu_kin

        self.f = ca.Function('f', [x_sym, u, dt], [f_mixed])
        self.F = ca.Function('F', [x_sym, u, dt], [ca.jacobian(f_mixed, x_sym)])

        #####################
        # Measurement Model #
        #####################
        ax_dyn = (Fxr + Fxf * ca.cos(steer) - Fyf * ca.sin(steer)) / self.p['m']
        ay_dyn = (Fyr + Fxf * ca.sin(steer) + Fyf * ca.cos(steer)) / self.p['m']
        ax_kin = accel
        ay_kin = 0.0

        ax_mixed = w_std * ax_dyn + (1 - w_std) * ax_kin
        ay_mixed = w_std * ay_dyn + (1 - w_std) * ay_kin

        h_imu_sym = ca.vertcat(
            theta,
            omega,
            ax_mixed,
            ay_mixed,
        )

        # Note: h_imu now depends on x_sym AND u_sym
        self._h_imu = ca.Function('h_imu', [x_sym, u], [h_imu_sym])
        self._H_imu = ca.Function('H_imu', [x_sym, u], [ca.jacobian(h_imu_sym, x_sym)])
        self.Hx_imu = lambda x, u: np.array(self._h_imu(x, u)).flatten()
        self.HJacobian_imu = lambda x, u: np.array(self._H_imu(x, u)).reshape(4, self.dim_x)

        H_vesc = np.eye(6)
        self.Hx_vesc = lambda x: H_vesc @ np.array(x).flatten()
        self.HJacobian_vesc = lambda x: H_vesc

    def predict(self, ekf, dt, u):
        x_flat = np.array(ekf.x).flatten()
        x_new = np.array(self.f(x_flat, u, dt)).flatten()

        F = np.array(self.F(x_flat, u, dt)).reshape(self.dim_x, self.dim_x)

        ekf.x = x_new
        ekf.P = F @ ekf.P @ F.T + self.Q

    def get_odom_state(self, x):
        # x_model = [x, y, psi, vx, vy, vpsi]
        # state = [x, y, psi, vx, vy, vpsi]
        return x


# ---------------------------------------------------------------------------
# Simulator-compatible CasADi single-track model (model_type: simulator_single_track_model)
#
# Mirrors f1tenth_parallel_sim's blended single-track dynamics and actuator model.
# Its parameters are deliberately kept apart from the pbl_config car/tire config.
# State x = [x, y, yaw, vx, vy, yaw_rate, steer], control u = [speed_command, delayed_steer_command].
# ---------------------------------------------------------------------------



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
    speed_delay: float = 0.0
    drive_gain: float = 3.0
    brake_gain: float = 5.0
    imu_x: float = 0.07  # IMU ahead of base_link (the rear axle), the base_link -> imu static transform

    @property
    def wheelbase(self) -> float:
        return self.l_f + self.l_r


class CommandDelay:
    """Time-stamped command history that returns the command as it was ``delay`` seconds ago.

    The simulator delays the steering command by ``steer_delay`` before it reaches the first-order
    steering lag and the speed command by ``speed_delay`` before it reaches the speed loop
    (``BatchedSim._delayed``). The command is a known input, so the EKF does not need
    delay states: it looks the delayed command up here and feeds it to the model as ``u[0]`` / ``u[1]``. Commands are
    piecewise constant (the actuation manager republishes each one on a timer and the sim holds it over
    the physics substeps), so the lookup holds the last command issued at or before ``t - delay`` rather
    than interpolating, which would add about half a command period of lag. Before the first sample the
    first value is used.
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
        index = bisect_right(self._t, t - self.delay) - 1
        return self._v[max(index, 0)]


def _magic_formula(alpha, B, C, D, E, fz):
    b_alpha = B * alpha
    return D * fz * ca.sin(C * ca.atan(b_alpha - E * (b_alpha - ca.atan(b_alpha))))


def _wrap_angle(angle):
    """Symbolic angle wrap to (-pi, pi], matching the old ``normalize_angle``."""
    return ca.atan2(ca.sin(angle), ca.cos(angle))


class SimulatorSingleTrackModel:
    """CasADi-symbolic, simulator-compatible seven-state single-track model.

    State x = [x, y, yaw, vx, vy, yaw_rate, steer]
    Control u = [speed_command_delayed, steer_command_delayed]

    ``predict`` integrates with RK4 over the step, holding ``u``; call it once per substep for steps
    over which the delayed steering command changes.

    The dynamic/kinematic blending, tire forces, load transfer and
    front-drive force split match the parallel simulator, as does the
    actuator model (``f1tenth_sim.actuators.ActuatorModel``): the steering
    command, already delayed by ``steer_delay``, maps linearly (gain, bias)
    to a road-wheel target that the wheel follows through a first-order lag
    ``steer_latency``, and the drive effort per unit mass is a proportional
    speed loop ``gain * (speed_command - vx)`` with separate drive/brake
    gains, the speed command delayed by ``speed_delay``. The delays are not in the graph: ``u`` holds the
    commands *as delayed by the caller* (see :class:`CommandDelay`), which is exact
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
        # Classical RK4 over the predict step with the command held, like the simulator's integrator.
        xdot = ca.Function('xdot', [x_sym, u_sym], [derivatives])
        k1 = xdot(x_sym, u_sym)
        k2 = xdot(x_sym + 0.5 * dt * k1, u_sym)
        k3 = xdot(x_sym + 0.5 * dt * k2, u_sym)
        k4 = xdot(x_sym + dt * k3, u_sym)
        x_next_raw = x_sym + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x_next = ca.vertcat(
            x_next_raw[0], x_next_raw[1], _wrap_angle(x_next_raw[2]),
            x_next_raw[3], x_next_raw[4], x_next_raw[5], x_next_raw[6],
        )
        self.f = ca.Function('f', [x_sym, u_sym, dt], [x_next])
        self.F = ca.Function('F', [x_sym, u_sym, dt], [ca.jacobian(x_next, x_sym)])

        #####################
        # Measurement Model #
        #####################
        # Accelerations at the IMU, imu_x - l_r ahead of the CoG along the body x axis: imu_transformer only
        # rotates them into base_link, it does not move them to another point.
        r_imu = p.imu_x - p.l_r
        ax = derivatives[3] - yaw_rate * vy - yaw_rate**2 * r_imu
        ay = derivatives[4] + yaw_rate * vx + derivatives[5] * r_imu
        h_imu_sym = ca.vertcat(yaw, yaw_rate, ax, ay)
        _h_imu = ca.Function('h_imu', [x_sym, u_sym], [h_imu_sym])
        _H_imu = ca.Function('H_imu', [x_sym, u_sym], [ca.jacobian(h_imu_sym, x_sym)])
        self.casadi_functions = {'h_imu': _h_imu, 'H_imu': _H_imu}  # for offline tools that rebuild the filter symbolically
        self.Hx_imu = lambda state, u: np.array(_h_imu(state, u)).flatten()
        self.HJacobian_imu = lambda state, u: np.array(_H_imu(state, u)).reshape(4, self.dim_x)

        # Pose of base_link (the rear axle, l_r behind the CoG) in the state's frame, which is what SLAM reports.
        h_pose_sym = ca.vertcat(x - p.l_r * ca.cos(yaw), y - p.l_r * ca.sin(yaw), yaw)
        _h_pose = ca.Function('h_pose', [x_sym], [h_pose_sym])
        _H_pose = ca.Function('H_pose', [x_sym], [ca.jacobian(h_pose_sym, x_sym)])
        self.Hx_pose = lambda state: np.array(_h_pose(state)).flatten()
        self.HJacobian_pose = lambda state: np.array(_H_pose(state)).reshape(3, self.dim_x)

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
