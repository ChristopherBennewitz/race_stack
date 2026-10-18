# EKF Formulation

Formal specification of `state_estimation/ekf_node` as currently implemented:
a `filterpy.kalman.ExtendedKalmanFilter` driven by a CasADi-symbolic,
simulator-matched single-track vehicle model, fused sequentially with IMU and
VESC odometry. VIO is not integrated.

## 1. State and control

$$
\mathbf{x} = \begin{bmatrix} x & y & \psi & v_x & v_y & \omega & \delta \end{bmatrix}^\top \in \mathbb{R}^7
$$

| symbol | meaning |
|---|---|
| $x,y$ | position in the `odom` frame |
| $\psi$ | yaw |
| $v_x, v_y$ | body-frame longitudinal / lateral velocity |
| $\omega$ | yaw rate |
| $\delta$ | steering angle (filter **state**, not a direct input) |

$$
\mathbf{u} = \begin{bmatrix} v_{\mathrm{cmd}} & \delta_{\mathrm{cmd}}(t-T_d) \end{bmatrix}^\top \in \mathbb{R}^2
$$

$v_{\mathrm{cmd}}$ is the applied speed command (Ackermann `speed`, used as is) and
$\delta_{\mathrm{cmd}}$ is the commanded steering angle (Ackermann
`steering_angle`, positive = left, as in the simulator). Both come from
`/ackermann_cmd_applied`, the actuation manager's rate/range-limited output
(`cmd_topic`) -- the plant input the model was identified against, not the raw
`/drive` request. The steering entry of $\mathbf u$ is the command **as delayed by**
$T_d$ (`steer_delay`): the node keeps a time-stamped history of the commands
(`CommandDelay`) and looks up the one in effect $T_d$ earlier: the last command issued at
or before $t-T_d$, held (the commands are piecewise constant, so interpolating would add
about half a command period of lag). The prediction looks it up at the middle of each
integration substep (§4), the measurement updates at the current time. The command is a known input, so this is exact
and needs no delay states; it mirrors the simulator's `BatchedSim._delayed_steer`.
The drive effort per unit mass $a$ used below is *derived* from the state and
$v_{\mathrm{cmd}}$ by the speed loop of §3; it is a force request
($F = m a$), not the physical acceleration.

## 2. Vehicle and tire parameters (ROS parameters)

$m, I_z, l_f, l_r, h_{cg}$ (mass, yaw inertia, axle distances, CoG height);
front/rear Pacejka coefficients $B_i, C_i, D_i, E_i$ for $i \in \{f, r\}$;
$\lambda$ = `front_drive_frac`; $T_c$ = `constraint_time_constant`; $v_b$ =
`blend_center`; $\varepsilon$ = `slip_speed_regularizer`; $\tau$ =
`steer_latency` (the single steering lag); $T_d$ = `steer_delay`; $k_\delta$, $b_\delta$ =
`steer_gain`, `steer_bias`; $k_d$, $k_b$ = `drive_gain`, `brake_gain`.
Wheelbase $L = l_f + l_r$. Defaults are the simulator's fitted
`f1tenth_car.toml` values (actuator parameters from `runs/sysid/fit/20261011-151030`, tires from
`fit_clean/20261004-233031`). Command range/rate limits belong to the actuation
limiter and are not modelled here.

## 3. Continuous-time dynamics

Everything below is built once as a single CasADi graph; the discrete
transition **and** every Jacobian used by the filter are derived from that one
graph (`ca.jacobian`), so they cannot drift apart from each other.

**Actuators** (mirrors `f1tenth_sim.actuators.ActuatorModel`). The steering
command, already delayed by $T_d$, maps linearly to a road-wheel target that the wheel
follows through a first-order lag of time constant $\tau$ (`steer_latency`):

$$
\delta^\star = k_\delta\,\delta_{\mathrm{cmd}}(t-T_d) + b_\delta, \qquad
\dot\delta = \frac{\delta^\star - \delta}{\tau}
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
\dot\delta
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
\dot\delta
\end{bmatrix}
$$

**Speed-based blend** ($w\to0$ at low speed / pure kinematic, $w\to1$ at high
speed / pure dynamic):

$$
w(v_x) = \frac{v_x^2}{v_x^2+v_b^2}, \qquad
\dot{\mathbf{x}} = f_c(\mathbf{x},\mathbf{u}) = (1-w)\,\dot{\mathbf{x}}_{\mathrm{kin}} + w\,\dot{\mathbf{x}}_{\mathrm{dyn}}
$$

## 4. Discretization and prediction

Classical RK4 with the measured wall-clock step $\mathrm{d}t$ (nominally
$1/100\,\mathrm{s}$; ticks with $\mathrm{d}t \le 0$ or $> 0.25\,\mathrm{s}$ are
skipped), holding $\mathbf u$ over the step like the simulator's integrator:

$$
f(\mathbf{x},\mathbf{u},\mathrm{d}t) = \mathbf{x} + \tfrac{\mathrm{d}t}{6}\,(k_1 + 2k_2 + 2k_3 + k_4),
\quad\text{with } \psi \leftarrow \operatorname{atan2}(\sin\psi,\cos\psi)
$$

$$
k_1 = f_c(\mathbf{x},\mathbf{u}),\;
k_2 = f_c(\mathbf{x}+\tfrac{\mathrm{d}t}{2}k_1,\mathbf{u}),\;
k_3 = f_c(\mathbf{x}+\tfrac{\mathrm{d}t}{2}k_2,\mathbf{u}),\;
k_4 = f_c(\mathbf{x}+\mathrm{d}t\,k_3,\mathbf{u})
$$

$$
F(\mathbf{x},\mathbf{u},\mathrm{d}t) = \frac{\partial f}{\partial \mathbf{x}} \quad \text{(exact, via } \texttt{ca.jacobian}\text{)}
$$

The node splits each tick into substeps of at most `sim_predict_substep` (5 ms) and runs the
prediction below once per substep, each with its own delayed steering command (the command
can change within a tick), covariance growth $Q\,h$ per substep of length $h$.

Prediction step:

$$
\hat{\mathbf{x}}_{k+1|k} = f(\hat{\mathbf{x}}_k, \mathbf{u}_k, \mathrm{d}t), \qquad
P_{k+1|k} = F_k P_k F_k^\top + Q\,\mathrm{d}t
$$

$Q = \operatorname{diag}(q_1,\dots,q_7)$ is a fixed ROS-parameter tuning matrix
(default $\operatorname{diag}(4.9\mathrm{e}{-5},\,3.6\mathrm{e}{-5},\,3.6\mathrm{e}{-5},\,4.0\mathrm{e}{-4},\,2.6\mathrm{e}{-3},\,0.18,\,4.2\mathrm{e}{-3})$, see below), scaled by $\mathrm{d}t$ so covariance growth is
time-proportional.

The default $Q$ comes from the simulator's system identification
(actuator block from `runs/sysid/fit/20261011-151030`, tire block from
`fit_clean/20261004-233031`, the tire values the car config still uses): the fitted
parameter covariance (steering, drive/brake and Pacejka parameters, with their
correlations within each block) was
sampled and propagated through $f_c$ over a grid of speeds, steering angles and
commands (the steering-delay uncertainty enters as an equivalent steering-command error,
delay error times a steering rate of $0,\pm1$ rad/s, because the delay is not in the graph);
the variance of the resulting derivative error per state, times the
nominal $\mathrm{d}t = 0.01\,\mathrm{s}$, gives $q_i$ for
$v_x, v_y, \omega, \delta$. $x, y, \psi$ depend on no identified
parameter directly, so they use the sysid pose noise variances. The yaw-rate
entry is large because the rear tire parameters and the steering delay are poorly
determined. The fit's residuals were about 10x the assumed noise, so these values
are, if anything, optimistic: on the Oct 11 bags a threefold larger $Q$ improved the
yaw-rate error slightly. The default $R$ takes the sysid gyro yaw-rate
variance ($0.03^2$) for the IMU and the odometry speed variance ($0.15^2$) for
$v_x$. The VESC pose entries are set huge (see §5) and the remaining entries are
unchanged (the sysid pose sigmas describe SLAM, not VESC odometry).

## 5. Measurement models

**IMU** (`/sensors/imu/raw`, 4-dimensional), after an EMA on the accelerometer
channel with weight $\alpha$ = `imu_accel_alpha` ($\tilde a_k = \alpha a_k + (1-\alpha)\tilde a_{k-1}$; $\alpha = 1$, i.e. no
EMA, for the simulator model, see below):

$$
\mathbf{z}_{\mathrm{imu}} = \begin{bmatrix}\psi_m & \omega_m & \tilde a_{x,m} & \tilde a_{y,m}\end{bmatrix}^\top,
\qquad
h_{\mathrm{imu}}(\mathbf{x},\mathbf{u}) = \begin{bmatrix}\psi \\ \omega \\ \dot v_x - \omega v_y \\ \dot v_y + \omega v_x\end{bmatrix}
$$

where $\dot v_x, \dot v_y$ are the **blended** velocity derivatives from
$f_c$, so the expected acceleration is consistent with whatever
kinematic/dynamic mix is active. $H_{\mathrm{imu}} = \partial h_{\mathrm{imu}}/\partial\mathbf{x} \in \mathbb{R}^{4\times7}$,
CasADi-exact.

$$
R_{\mathrm{imu}} = \operatorname{diag}(\sigma_\psi^2,\sigma_\omega^2,\sigma_{a_x}^2,\sigma_{a_y}^2), \quad \text{default } \operatorname{diag}(0.1,\,9\mathrm{e}{-4},\,1.4,\,3.0)
$$

The predicted accelerations are taken **at the IMU**, $r = $ `sim_imu_x` $- l_r$ ahead of the CoG
(`sim_imu_x` = 0.07 m, the `base_link -> imu` static transform; `base_link` is the rear axle):
$a_x \leftarrow a_x - \omega^2 r$, $a_y \leftarrow a_y + \dot\omega\, r$. `imu_transformer` only rotates the
sensor's 90°-yawed frame into `base_link`; it does not move the measurement to another point.

The accelerometer reports the acceleration through a first-order lag of about
`sim_imu_accel_lag` = 50 ms (the sensor's own low-pass; fitted with `sysid/latency.py` against
$\mathrm{d}v/\mathrm{d}t$ of the VESC speed and $v\omega$ from the gyro: 30–50 ms on $a_x$, 65–70 ms on $a_y$).
So for the simulator model it is fused **raw** (no EMA, which would add another ~75 ms) and compared with
the predicted acceleration passed through the same lag, kept by `PredictionHistory`; the innovation is
applied to the present state through the present Jacobian. The variances $\sigma_{a_x}^2 = 1.4$,
$\sigma_{a_y}^2 = 3.0$ are the residuals of those fits (the raw accelerometer carries 0.6–1.5 m/s² of
vibration), and `sim_imu_accel_bias` = $(0.05, 0.56)$ m/s² is the constant mounting offset they leave.
The yaw measurement $\psi_m$ is
the IMU orientation yaw **relative to its first
sample** (the IMU's absolute yaw is in its own, mounted-rotated frame), so the odom frame
starts with its $x$ axis along the car's initial heading and matches the VESC odometry
frame. The first IMU message also seeds the filter:
$\hat x_2 \leftarrow \psi_m,\ \hat x_5 \leftarrow \omega_m,\ P_{22}\leftarrow R_{\mathrm{imu},00},\ P_{55}\leftarrow R_{\mathrm{imu},11}$;
the node does not run `predict`/`update` at all before this happens.

**VESC odometry** (`/odom`, 6-dimensional):

$$
\mathbf{z}_{\mathrm{vesc}} = \begin{bmatrix}x_m & y_m & \psi_m & v_{x,m} & v_{y,m} & \omega_m\end{bmatrix}^\top,
\qquad
h_{\mathrm{vesc}}(\mathbf{x}) = H_{\mathrm{vesc}}\,\mathbf{x}, \quad H_{\mathrm{vesc}} = \begin{bmatrix}I_6 & \mathbf 0_{6\times1}\end{bmatrix}
$$

i.e. a fixed linear selection of the first six states (the steering angle is
unobserved by this sensor); $H_{\mathrm{vesc}}$ is thus its own Jacobian.

$$
R_{\mathrm{vesc}} = \operatorname{diag}(\sigma_x^2,\sigma_y^2,\sigma_\psi^2,\sigma_{v_x}^2,\sigma_{v_y}^2,\sigma_\omega^2), \quad \text{default } \operatorname{diag}(10^6,\,10^6,\,10^6,\,0.0225,\,100,\,0.1)
$$

The VESC pose entries are deliberately huge, so the pose is **not** fused: the VESC odometry
integrates $x, y, \psi$ from zero in its own frame, which would fight the IMU heading and the
filter's own integration of $v_x$ along it. Only $v_x, v_y, \omega$ are used from this sensor.
The pose comes from integrating velocity along the IMU-derived heading; Cartographer
corrects it.

**SLAM forward speed** (`/car_state/pose`, 1-dimensional). The map-frame pose from TF is differenced over
consecutive, non-overlapping windows of `sim_slam_speed_window` (0.2 s; `TrailingSpeedWindow`): the
displacement along the mean heading divided by the window, i.e. the mean speed over
$[t_0, t_1]$ in pose-stamp time. The pose stamps lag the car by `sim_slam_pose_delay` = 45 ms (the window
speed at its mid-stamp still trails the VESC speed by that much, `sysid/latency.py`), so the sample is
compared with the filter's own mean speed estimate over $[t_0 - 45\,\mathrm{ms},\, t_1 - 45\,\mathrm{ms}]$
from `PredictionHistory`, not with the present speed. Its Jacobian is the $v_x$ selector and
$R_{\mathrm{slam}} = 0.04$. The sample is applied once per window (5 Hz), which re-grounds $v_x$
against wheel-speed scale error and slip without letting SLAM dominate the 50 Hz VESC speed. A sample whose
normalized innovation exceeds `sim_slam_speed_gate` (3 standard deviations) is dropped, which keeps
scan-matching jumps and loop closures out. The localization is fed this filter's own odometry, so on short
windows its speed largely echoes the filter; the long window and the large $R$ keep that feedback weak.

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
\hat{\mathbf x}_0 = \mathbf 0_7, \qquad P_0 = 10\,I_7
$$

refined immediately by the first IMU message as in §5. The filter does not
predict or update until `is_initialized` is set.

## 8. Output

Published state is $[\hat x_0,\hat x_1,\hat x_2,\hat x_3,\hat x_4,\hat x_5]$
(the steering angle $\hat x_6$ is internal only) as `nav_msgs/Odometry` on
`/early_fusion/odom` at the filter's own rate (i.e. only on ticks that actually ran a
predict). The `odom → base_link` TF is off by default (`publish_tf`): Cartographer
publishes `map → base_link` and a static `map → odom` exists, so a second parent for
`base_link` would make the SLAM pose jump.
