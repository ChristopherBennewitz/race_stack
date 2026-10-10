# EKF Formulation

Formal specification of `state_estimation/ekf_node` as currently implemented:
a `filterpy.kalman.ExtendedKalmanFilter` driven by a CasADi-symbolic,
simulator-matched single-track vehicle model, fused sequentially with IMU and
VESC odometry. VIO is not integrated.

## 1. State and control

$$
\mathbf{x} = \begin{bmatrix} x & y & \psi & v_x & v_y & \omega & \delta & \delta_{\ell} \end{bmatrix}^\top \in \mathbb{R}^8
$$

| symbol | meaning |
|---|---|
| $x,y$ | position in the `odom` frame |
| $\psi$ | yaw |
| $v_x, v_y$ | body-frame longitudinal / lateral velocity |
| $\omega$ | yaw rate |
| $\delta$ | steering angle (filter **state**, not a direct input) |
| $\delta_\ell$ | intermediate steering-lag state (second stage of the actuator) |

$$
\mathbf{u} = \begin{bmatrix} v_{\mathrm{cmd}} & \delta_{\mathrm{cmd}} \end{bmatrix}^\top \in \mathbb{R}^2
$$

$v_{\mathrm{cmd}}$ is the applied speed command (Ackermann `speed`, used as is) and
$\delta_{\mathrm{cmd}}$ is the commanded steering angle (Ackermann
`steering_angle`, positive = left, as in the simulator). Both come from
`/ackermann_cmd_applied`, the actuation manager's rate/range-limited output
(`cmd_topic`) -- the plant input the model was identified against, not the raw
`/drive` request.
The drive effort per unit mass $a$ used below is *derived* from the state and
$v_{\mathrm{cmd}}$ by the speed loop of §3; it is a force request
($F = m a$), not the physical acceleration.

## 2. Vehicle and tire parameters (ROS parameters)

$m, I_z, l_f, l_r, h_{cg}$ (mass, yaw inertia, axle distances, CoG height);
front/rear Pacejka coefficients $B_i, C_i, D_i, E_i$ for $i \in \{f, r\}$;
$\lambda$ = `front_drive_frac`; $T_c$ = `constraint_time_constant`; $v_b$ =
`blend_center`; $\varepsilon$ = `slip_speed_regularizer`; $\tau$ =
`steer_latency` (total of the two steering lags); $k_\delta$, $b_\delta$ =
`steer_gain`, `steer_bias`; $k_d$, $k_b$ = `drive_gain`, `brake_gain`.
Wheelbase $L = l_f + l_r$. Defaults are the simulator's fitted
`f1tenth_car.toml` values. Command range/rate limits belong to the actuation
limiter and are not modelled here.

## 3. Continuous-time dynamics

Everything below is built once as a single CasADi graph; the discrete
transition **and** every Jacobian used by the filter are derived from that one
graph (`ca.jacobian`), so they cannot drift apart from each other.

**Actuators** (mirrors `f1tenth_sim.actuators.ActuatorModel`). The steering
command maps linearly to a road-wheel target that the wheel follows through two
equal first-order lags whose time constants add up to $T_s$ (`steer_latency`),
i.e. a stage time constant $T_s/2$:

$$
\delta^\star = k_\delta\,\delta_{\mathrm{cmd}} + b_\delta, \qquad
\dot\delta = \frac{\delta_\ell - \delta}{T_s/2}, \qquad
\dot\delta_\ell = \frac{\delta^\star - \delta_\ell}{T_s/2}
$$

Drive effort per unit mass is a proportional speed loop, with the gain
switching where the error is zero (braking gain when $|v_{\mathrm{cmd}}| \le |v_x|$):

$$
a = k\,(v_{\mathrm{cmd}} - v_x), \qquad
k = \begin{cases} k_d & v_{\mathrm{cmd}}^2 > v_x^2 \\ k_b & \text{otherwise}\end{cases}
$$

**Tire slip angles**, regularized against the low-speed singularity:

$$
v_{y,f} = -v_x\sin\delta + (v_y + l_f\omega)\cos\delta, \qquad
v_{x,f} = v_x\cos\delta + (v_y + l_f\omega)\sin\delta
$$

$$
\alpha_f = -\operatorname{atan2}\!\left(v_{y,f},\ \sqrt{v_{x,f}^2+\varepsilon^2}\right), \qquad
\alpha_r = -\operatorname{atan2}\!\left(v_y - l_r\omega,\ \sqrt{v_x^2+\varepsilon^2}\right)
$$

**Quasi-static load transfer** (constant $g = 9.81\,\mathrm{m/s^2}$):

$$
F_{z,f} = \frac{m(-a\, h_{cg} + g\, l_r)}{L}, \qquad
F_{z,r} = \frac{m(a\, h_{cg} + g\, l_f)}{L}
$$

**Magic-formula lateral force**, for $i \in \{f,r\}$:

$$
F_{y,i} = D_i F_{z,i} \sin\!\Big(C_i \arctan\big(B_i\alpha_i - E_i(B_i\alpha_i - \arctan(B_i\alpha_i))\big)\Big)
$$

**Longitudinal force split** (open-loop, no traction/slip model):

$$
F_{x,f} = \lambda\, m a, \qquad F_{x,r} = (1-\lambda)\, m a
$$

**Dynamic (single-track) derivatives:**

$$
\dot{\mathbf{x}}_{\mathrm{dyn}} =
\begin{bmatrix}
v_x\cos\psi - v_y\sin\psi \\
v_x\sin\psi + v_y\cos\psi \\
\omega \\
\dfrac{F_{x,r}+F_{x,f}\cos\delta-F_{y,f}\sin\delta}{m} + \omega v_y \\
\dfrac{F_{y,r}+F_{x,f}\sin\delta+F_{y,f}\cos\delta}{m} - \omega v_x \\
\dfrac{l_f(F_{x,f}\sin\delta+F_{y,f}\cos\delta) - l_r F_{y,r}}{I_z} \\
\dot\delta \\
\dot\delta_\ell
\end{bmatrix}
$$

**Kinematic (no-slip) derivatives**, relaxed towards the constraint with time
constant $T_c$ instead of enforced algebraically (keeps everything smooth):

$$
\omega^\* = \frac{v_x\tan\delta}{L}, \qquad
v_y^\* = l_r\omega^\*, \qquad
\dot\omega^\* = \frac{a\tan\delta + v_x\dot\delta/\cos^2\delta}{L}, \qquad
\dot v_y^\* = l_r\dot\omega^\*
$$

$$
\dot{\mathbf{x}}_{\mathrm{kin}} =
\begin{bmatrix}
v_x\cos\psi - v_y\sin\psi \\
v_x\sin\psi + v_y\cos\psi \\
\omega \\
a \\
\dot v_y^\* + (v_y^\*-v_y)/T_c \\
\dot\omega^\* + (\omega^\*-\omega)/T_c \\
\dot\delta \\
\dot\delta_\ell
\end{bmatrix}
$$

**Speed-based blend** ($w\to0$ at low speed / pure kinematic, $w\to1$ at high
speed / pure dynamic):

$$
w(v_x) = \frac{v_x^2}{v_x^2+v_b^2}, \qquad
\dot{\mathbf{x}} = f_c(\mathbf{x},\mathbf{u}) = (1-w)\,\dot{\mathbf{x}}_{\mathrm{kin}} + w\,\dot{\mathbf{x}}_{\mathrm{dyn}}
$$

## 4. Discretization and prediction

Forward Euler with the measured wall-clock step $\mathrm{d}t$ (nominally
$1/100\,\mathrm{s}$; ticks with $\mathrm{d}t \le 0$ or $> 0.25\,\mathrm{s}$ are
skipped):

$$
f(\mathbf{x},\mathbf{u},\mathrm{d}t) = \mathbf{x} + \mathrm{d}t\, f_c(\mathbf{x},\mathbf{u}),
\quad\text{with } \psi \leftarrow \operatorname{atan2}(\sin\psi,\cos\psi)
$$

$$
F(\mathbf{x},\mathbf{u},\mathrm{d}t) = \frac{\partial f}{\partial \mathbf{x}} \quad \text{(exact, via } \texttt{ca.jacobian}\text{)}
$$

Prediction step:

$$
\hat{\mathbf{x}}_{k+1|k} = f(\hat{\mathbf{x}}_k, \mathbf{u}_k, \mathrm{d}t), \qquad
P_{k+1|k} = F_k P_k F_k^\top + Q\,\mathrm{d}t
$$

$Q = \operatorname{diag}(q_1,\dots,q_8)$ is a fixed ROS-parameter tuning matrix
(default $\operatorname{diag}(4.9\mathrm{e}{-5},\,3.6\mathrm{e}{-5},\,3.6\mathrm{e}{-5},\,5.8\mathrm{e}{-4},\,1.8\mathrm{e}{-3},\,6.7\mathrm{e}{-2},\,2.3\mathrm{e}{-5},\,6.9\mathrm{e}{-3})$, see below), scaled by $\mathrm{d}t$ so covariance growth is
time-proportional.

The default $Q$ comes from the simulator's system identification
(`runs/sysid/fit_clean/20261004-233031`): the fitted parameter covariance
(steering, drive/brake and Pacejka parameters, with their correlations) was
sampled and propagated through $f_c$ over a grid of speeds, steering angles and
commands; the variance of the resulting derivative error per state, times the
nominal $\mathrm{d}t = 0.01\,\mathrm{s}$, gives $q_i$ for
$v_x, v_y, \omega, \delta, \delta_\ell$. $x, y, \psi$ depend on no identified
parameter directly, so they use the sysid pose noise variances. The yaw-rate
entry is large because the rear tire stiffness/shape parameters are poorly
determined. The fit's residuals were 9.4x the assumed noise, so these values
are, if anything, optimistic. The default $R$ takes the sysid gyro yaw-rate
variance ($0.03^2$) for the IMU and the odometry speed variance ($0.2^2$) for
$v_x$; the other entries are unchanged (the sysid pose sigmas describe SLAM,
not VESC odometry).

## 5. Measurement models

**IMU** (`/sensors/imu/raw`, 4-dimensional), after an EMA on the accelerometer
channel with weight $\alpha$ = `imu_accel_alpha` ($\tilde a_k = \alpha a_k + (1-\alpha)\tilde a_{k-1}$):

$$
\mathbf{z}_{\mathrm{imu}} = \begin{bmatrix}\psi_m & \omega_m & \tilde a_{x,m} & \tilde a_{y,m}\end{bmatrix}^\top,
\qquad
h_{\mathrm{imu}}(\mathbf{x},\mathbf{u}) = \begin{bmatrix}\psi \\ \omega \\ \dot v_x - \omega v_y \\ \dot v_y + \omega v_x\end{bmatrix}
$$

where $\dot v_x, \dot v_y$ are the **blended** velocity derivatives from
$f_c$, so the expected acceleration is consistent with whatever
kinematic/dynamic mix is active. $H_{\mathrm{imu}} = \partial h_{\mathrm{imu}}/\partial\mathbf{x} \in \mathbb{R}^{4\times8}$,
CasADi-exact.

$$
R_{\mathrm{imu}} = \operatorname{diag}(\sigma_\psi^2,\sigma_\omega^2,\sigma_{a_x}^2,\sigma_{a_y}^2), \quad \text{default } \operatorname{diag}(0.1,\,9\mathrm{e}{-4},\,10^6,\,10^6)
$$

The acceleration variance is deliberately enormous: $a_x,a_y$ contribute
negligible correction weight and exist as a held-out consistency channel
(see `ekf_compare_launch.py`/`ekf_benchmark_node.py`), not a correction
source. The first IMU message also seeds the filter:
$\hat x_2 \leftarrow \psi_m,\ \hat x_5 \leftarrow \omega_m,\ P_{22}\leftarrow R_{\mathrm{imu},00},\ P_{55}\leftarrow R_{\mathrm{imu},11}$;
the node does not run `predict`/`update` at all before this happens.

**VESC odometry** (`/odom`, 6-dimensional):

$$
\mathbf{z}_{\mathrm{vesc}} = \begin{bmatrix}x_m & y_m & \psi_m & v_{x,m} & v_{y,m} & \omega_m\end{bmatrix}^\top,
\qquad
h_{\mathrm{vesc}}(\mathbf{x}) = H_{\mathrm{vesc}}\,\mathbf{x}, \quad H_{\mathrm{vesc}} = \begin{bmatrix}I_6 & \mathbf 0_{6\times2}\end{bmatrix}
$$

i.e. a fixed linear selection of the first six states (the steering states are
unobserved by this sensor); $H_{\mathrm{vesc}}$ is thus its own Jacobian.

$$
R_{\mathrm{vesc}} = \operatorname{diag}(\sigma_x^2,\sigma_y^2,\sigma_\psi^2,\sigma_{v_x}^2,\sigma_{v_y}^2,\sigma_\omega^2), \quad \text{default } \operatorname{diag}(1,\,1,\,1,\,0.04,\,100,\,0.1)
$$

## 6. Sequential update

At every $100\,\mathrm{Hz}$ tick, `predict` runs once, then each sensor with a
*fresh* (not yet consumed) sample applies its own update **in sequence** on
the same filter state — no stacked joint-innovation update. For a sensor with
measurement function $h$, Jacobian $H$, noise $R$ and measurement $\mathbf z$
(filterpy's `ExtendedKalmanFilter.update`, Joseph-form covariance):

$$
\mathbf y = \mathbf z - h(\hat{\mathbf x}_{k+1|k}) \quad\text{(yaw component wrapped to } (-\pi,\pi])
$$

$$
S = H P_{k+1|k} H^\top + R, \qquad K = P_{k+1|k} H^\top S^{-1}
$$

$$
\hat{\mathbf x} \leftarrow \hat{\mathbf x}_{k+1|k} + K\mathbf y, \qquad
P \leftarrow (I-KH)\,P_{k+1|k}\,(I-KH)^\top + K R K^\top
$$

Only the **residual** $\mathbf y$ wraps the yaw component; $\hat x_2$ itself is
not re-wrapped after an update (it is re-wrapped again inside the next
`predict`, which is harmless since $\sin,\cos$ are periodic — but note the
published/raw state can transiently sit outside $(-\pi,\pi]$ between an
update and the next predict).

## 7. Initialization

$$
\hat{\mathbf x}_0 = \mathbf 0_8, \qquad P_0 = 10\,I_8
$$

refined immediately by the first IMU message as in §5. The filter does not
predict or update until `is_initialized` is set.

## 8. Output

Published state is $[\hat x_0,\hat x_1,\hat x_2,\hat x_3,\hat x_4,\hat x_5]$
(steering states $\hat x_6,\hat x_7$ are internal only) as `nav_msgs/Odometry` on
`/early_fusion/odom` at the filter's own rate (i.e. only on ticks that actually ran a
predict). The `odom → base_link` TF is off by default (`publish_tf`): Cartographer
publishes `map → base_link` and a static `map → odom` exists, so a second parent for
`base_link` would make the SLAM pose jump.
