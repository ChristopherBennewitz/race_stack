#include "progress_mpc/progress_mpc.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "acados_solver_progress_mpc.h"
#include "mpc_config.h"
#include "mpc_step.h"

namespace progress_mpc {
namespace {

using namespace generated;

constexpr int kRefDim = 7;  // centre x, y; tangent x, y; arc length at centre; left, right width
constexpr int kStateDim = kVehicleDim + kHistory;  // vehicle state, then the steering and the speed history, newest first
constexpr int kSpeedHistory = kHistory - kSteerHistory;
constexpr int kAugDim = kStateDim + 1;             // the OCP state adds the last speed command
constexpr int kVehicleSearchLo = -15, kVehicleSearchHi = 45;  // centerline points around the last match
constexpr int kStageSearch = 10;  // points a later stage's reference may advance past the previous one

static_assert(kVehicleDim == kVehicleStateDim, "the generated model's vehicle state changed");
static_assert(kAugDim == PROGRESS_MPC_NX, "the generated OCP does not match the exported step function");
static_assert(kHorizon == PROGRESS_MPC_N, "horizon mismatch between mpc_config.h and the acados code");

enum StateIdx { kX = 0, kY = 1, kYaw = 2, kVx = 3, kVy = 4, kYawRate = 5, kSteer = 6 };

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

double clamp(double v, double lo, double hi) { return std::min(std::max(v, lo), hi); }

int wrap(int i, int n) { return ((i % n) + n) % n; }

// A closed centerline with per-point tangent, arc length and widths, at uniform spacing.
struct Track {
  int count = 0;
  double length = 0.0;
  std::vector<std::array<double, 2>> center, tangent;
  std::vector<double> cum_s, w_left, w_right;
};

Track build_track(std::vector<std::array<double, 2>> pts, std::vector<double> wl, std::vector<double> wr,
                  double spacing) {
  if (pts.size() < 4 || wl.size() != pts.size() || wr.size() != pts.size())
    throw std::invalid_argument("track needs >= 4 points and one left/right width per point");
  // A repeated closing point would make a zero-length segment.
  if (std::hypot(pts.front()[0] - pts.back()[0], pts.front()[1] - pts.back()[1]) < 1e-6) {
    pts.pop_back();
    wl.pop_back();
    wr.pop_back();
  }
  const size_t n = pts.size();
  std::vector<double> seg_start(n + 1, 0.0);
  for (size_t i = 0; i < n; ++i) {
    const auto& a = pts[i];
    const auto& b = pts[(i + 1) % n];
    seg_start[i + 1] = seg_start[i] + std::hypot(b[0] - a[0], b[1] - a[1]);
  }
  const double total = seg_start[n];
  if (!(total > 0.4)) throw std::invalid_argument("track is too short");

  Track t;
  t.length = total;
  if (spacing > 0.0) {
    t.count = std::max(4, static_cast<int>(std::lround(total / spacing)));
    const double step = total / t.count;
    t.center.resize(t.count);
    t.cum_s.resize(t.count);
    t.w_left.resize(t.count);
    t.w_right.resize(t.count);
    size_t seg = 0;
    for (int k = 0; k < t.count; ++k) {
      const double s = k * step;
      while (seg + 1 < n && seg_start[seg + 1] <= s) ++seg;
      const double len = seg_start[seg + 1] - seg_start[seg];
      const double f = len > 0.0 ? (s - seg_start[seg]) / len : 0.0;
      const size_t j = (seg + 1) % n;
      t.center[k] = {pts[seg][0] + f * (pts[j][0] - pts[seg][0]), pts[seg][1] + f * (pts[j][1] - pts[seg][1])};
      t.w_left[k] = wl[seg] + f * (wl[j] - wl[seg]);
      t.w_right[k] = wr[seg] + f * (wr[j] - wr[seg]);
      t.cum_s[k] = s;
    }
  } else {
    t.count = static_cast<int>(n);
    t.center = std::move(pts);
    t.w_left = std::move(wl);
    t.w_right = std::move(wr);
    t.cum_s.assign(seg_start.begin(), seg_start.end() - 1);
  }
  t.tangent.resize(t.count);
  for (int k = 0; k < t.count; ++k) {
    const auto& a = t.center[wrap(k - 1, t.count)];
    const auto& b = t.center[wrap(k + 1, t.count)];
    const double dx = b[0] - a[0], dy = b[1] - a[1];
    const double norm = std::hypot(dx, dy);
    if (norm < 1e-9) throw std::invalid_argument("track has coincident points");
    t.tangent[k] = {dx / norm, dy / norm};
  }
  return t;
}

}  // namespace

struct ProgressMpc::Impl {
  using Vec = std::vector<double>;
  // Time-major: states (horizon + 1, kStateDim), controls (horizon, 2), refs (horizon + 1, kRefDim).
  struct Plan {
    Vec states, controls;
  };

  explicit Impl(double spacing) : track_spacing(spacing) {
    capsule = progress_mpc_acados_create_capsule();
    if (progress_mpc_acados_create(capsule) != 0) {
      progress_mpc_acados_free_capsule(capsule);
      throw std::runtime_error("acados failed to create the progress MPC solver");
    }
    config = progress_mpc_acados_get_nlp_config(capsule);
    dims = progress_mpc_acados_get_nlp_dims(capsule);
    in = progress_mpc_acados_get_nlp_in(capsule);
    out = progress_mpc_acados_get_nlp_out(capsule);
    solver = progress_mpc_acados_get_nlp_solver(capsule);
    plan_world.assign(kHorizon + 1, {0.0, 0.0});
    plan_frenet.assign(kHorizon + 1, {0.0, 0.0});
  }

  ~Impl() {
    progress_mpc_acados_free(capsule);
    progress_mpc_acados_free_capsule(capsule);
  }

  progress_mpc_solver_capsule* capsule = nullptr;
  ocp_nlp_config* config = nullptr;
  ocp_nlp_dims* dims = nullptr;
  ocp_nlp_in* in = nullptr;
  ocp_nlp_out* out = nullptr;
  ocp_nlp_solver* solver = nullptr;

  double track_spacing;
  Track track;
  bool have_track = false;

  // Per-period command change the actuation manager is guaranteed to apply (steer, speed).
  const std::array<double, 2> step_lo{-kSteerStep, -kSpeedStepDown};
  const std::array<double, 2> step_hi{kSteerStep, kSpeedStepUp};

  bool have_warm = false;
  Plan warm;
  Vec applied;  // steering history, then speed history (both newest first); empty before the first report
  int prev_idx = -1;
  double steer_estimate = 0.0;
  Stats stats;
  std::vector<std::array<double, 2>> plan_world, plan_frenet;

  // The model over one control period: z' = F(z, u).
  void model(const double* z, const double* u, double* z_next) const {
    const double* arg[2] = {z, u};
    double* res[1] = {z_next};
    casadi_int iw[F_step_SZ_IW + 1];
    double w[F_step_SZ_W + 1];
    if (F_step(arg, res, iw, w, 0) != 0) throw std::runtime_error("model step evaluation failed");
  }

  Vec simulate(const double* x0, const Vec& controls) const {
    const int n = static_cast<int>(controls.size() / 2);
    Vec states(static_cast<size_t>(n + 1) * kStateDim);
    std::copy(x0, x0 + kStateDim, states.begin());
    for (int k = 0; k < n; ++k)
      model(&states[k * kStateDim], &controls[2 * k], &states[(k + 1) * kStateDim]);
    return states;
  }

  static std::array<double, 2> command_of(const Vec& a) { return {a[0], a[kSteerHistory]}; }

  Vec applied_or_rest(double vx) const {
    if (!applied.empty()) return applied;
    Vec rest(kHistory, 0.0);
    rest[kSteerHistory] = vx;
    return rest;
  }

  // The straight launch used before there is a plan: accelerate towards 2 m/s within the rate limits.
  Plan rollout_guess(const Vec& x0, const Vec& ap) const {
    Plan p;
    p.states.assign(static_cast<size_t>(kHorizon + 1) * kStateDim, 0.0);
    p.controls.assign(static_cast<size_t>(kHorizon) * 2, 0.0);
    std::copy(x0.begin(), x0.end(), p.states.begin());
    auto cmd = command_of(ap);
    const double target[2] = {0.0, 2.0};
    for (int k = 0; k < kHorizon; ++k) {
      for (int i = 0; i < 2; ++i) cmd[i] += clamp(target[i] - cmd[i], step_lo[i], step_hi[i]);
      p.controls[2 * k] = cmd[0];
      p.controls[2 * k + 1] = cmd[1];
      model(&p.states[k * kStateDim], cmd.data(), &p.states[(k + 1) * kStateDim]);
    }
    return p;
  }

  // The previous controls shifted by one step and re-simulated from x0. Re-simulating keeps the warm start
  // dynamically consistent: an RTI solution's states are only the QP's linearization.
  Plan shifted_guess(const Vec& x0, const Vec& ap) const {
    if (!have_warm) return rollout_guess(x0, ap);
    Plan p;
    p.controls = warm.controls;
    std::copy(warm.controls.begin() + 2, warm.controls.end(), p.controls.begin());  // last command is repeated
    p.states = simulate(x0.data(), p.controls);
    return p;
  }

  // Projects planned positions onto the centerline; later stages may only advance, by a bounded step.
  Vec references(const Vec& states) {
    Vec refs(static_cast<size_t>(kHorizon + 1) * kRefDim);
    const int count = track.count;
    int idx = prev_idx;
    bool have_s = false;
    double s_prev = 0.0;
    for (int k = 0; k <= kHorizon; ++k) {
      const double px = states[k * kStateDim + kX], py = states[k * kStateDim + kY];
      // Candidates are offsets from the last match; before the first match, every point.
      int lo = 0, hi = count;
      if (idx >= 0) {
        lo = (k == 0) ? kVehicleSearchLo : 0;
        hi = (k == 0) ? kVehicleSearchHi : kStageSearch + 1;
      }
      double best = std::numeric_limits<double>::infinity();
      int best_idx = 0;
      for (int o = lo; o < hi; ++o) {
        const int c = (idx >= 0) ? wrap(idx + o, count) : o;
        const double dx = track.center[c][0] - px, dy = track.center[c][1] - py;
        const double d = dx * dx + dy * dy;
        if (d < best) {
          best = d;
          best_idx = c;
        }
      }
      idx = best_idx;
      if (k == 0) prev_idx = idx;
      double s = track.cum_s[idx];
      if (have_s) s += track.length * std::nearbyint((s_prev - s) / track.length);
      s_prev = s;
      have_s = true;
      double* r = &refs[k * kRefDim];
      r[0] = track.center[idx][0];
      r[1] = track.center[idx][1];
      r[2] = track.tangent[idx][0];
      r[3] = track.tangent[idx][1];
      r[4] = s;
      r[5] = track.w_left[idx];
      r[6] = track.w_right[idx];
    }
    return refs;
  }

  // One SQP-RTI iteration from the guess; true and the plan on success.
  bool solve(const Vec& x0, const Vec& ap, const Plan& guess, const Vec& refs, Plan& solution) {
    const int n = kHorizon;
    Vec x0_aug(x0);
    x0_aug.push_back(ap[kSteerHistory]);
    ocp_nlp_constraints_model_set(config, dims, in, 0, "lbx", x0_aug.data());
    ocp_nlp_constraints_model_set(config, dims, in, 0, "ubx", x0_aug.data());
    Vec x_k(kAugDim);
    for (int k = 0; k <= n; ++k) {
      progress_mpc_acados_update_params(capsule, k, const_cast<double*>(&refs[k * kRefDim]), kRefDim);
      std::copy(&guess.states[k * kStateDim], &guess.states[(k + 1) * kStateDim], x_k.begin());
      // The last speed command at stage k: the applied one, then the plan's own.
      x_k[kStateDim] = (k == 0) ? ap[kSteerHistory] : guess.controls[2 * (k - 1) + 1];
      ocp_nlp_out_set(config, dims, out, k, "x", x_k.data());
      if (k < n) ocp_nlp_out_set(config, dims, out, k, "u", const_cast<double*>(&guess.controls[2 * k]));
    }

    const auto start = Clock::now();
    const int status = progress_mpc_acados_solve(capsule);
    stats.solver_only_seconds = seconds_since(start);
    int iterations = 0;
    ocp_nlp_get(solver, "sqp_iter", &iterations);
    stats.iterations = iterations;
    if (status != 0) {
      stats.status = "acados_status_" + std::to_string(status);
      progress_mpc_acados_reset(capsule, 1);  // acados keeps its iterates; a NaN would poison every later step
      return false;
    }
    stats.status = "Solve_Succeeded";

    double su[PROGRESS_MPC_NS > PROGRESS_MPC_NSN ? PROGRESS_MPC_NS : PROGRESS_MPC_NSN];
    double slack = 0.0;
    for (int k = 1; k <= n; ++k) {
      ocp_nlp_out_get(config, dims, out, k, "su", su);
      const int ns = (k < n) ? PROGRESS_MPC_NS : PROGRESS_MPC_NSN;
      for (int i = 0; i < ns; ++i) slack = std::max(slack, su[i]);
    }
    stats.slack_max = slack;

    solution.states.resize(static_cast<size_t>(n + 1) * kStateDim);
    solution.controls.resize(static_cast<size_t>(n) * 2);
    for (int k = 0; k <= n; ++k) {
      ocp_nlp_out_get(config, dims, out, k, "x", x_k.data());
      std::copy(x_k.begin(), x_k.begin() + kStateDim, solution.states.begin() + k * kStateDim);
      if (k < n) ocp_nlp_out_get(config, dims, out, k, "u", &solution.controls[2 * k]);
    }
    return true;
  }

  // Keeps a plan as the next warm start; the stored states are the planned commands re-simulated from the
  // measured state, so the recorded plan is a trajectory the model can drive.
  Command accept(const Plan& solution, const Vec& refs) {
    warm.controls = solution.controls;
    warm.states = simulate(solution.states.data(), warm.controls);
    have_warm = true;
    for (int k = 0; k <= kHorizon; ++k) {
      const double* r = &refs[k * kRefDim];
      const double px = warm.states[k * kStateDim + kX], py = warm.states[k * kStateDim + kY];
      const double ox = px - r[0], oy = py - r[1];
      plan_world[k] = {px, py};
      plan_frenet[k] = {r[4] + r[2] * ox + r[3] * oy, -r[3] * ox + r[2] * oy};
    }
    return {solution.controls[0], solution.controls[1]};
  }
};

ProgressMpc::ProgressMpc(double track_spacing) : impl_(std::make_unique<Impl>(track_spacing)) {}
ProgressMpc::~ProgressMpc() = default;

void ProgressMpc::set_track(const std::vector<std::array<double, 2>>& centerline,
                            const std::vector<double>& width_left, const std::vector<double>& width_right) {
  impl_->track = build_track(centerline, width_left, width_right, impl_->track_spacing);
  impl_->have_track = true;
  impl_->prev_idx = -1;  // the old index means nothing on a new path
  impl_->have_warm = false;
}

bool ProgressMpc::has_track() const { return impl_->have_track; }
double ProgressMpc::track_length() const { return impl_->track.length; }

void ProgressMpc::reset(int centerline_index) {
  impl_->prev_idx = centerline_index;
  impl_->have_warm = false;
  impl_->applied.clear();
  impl_->steer_estimate = 0.0;
  progress_mpc_acados_reset(impl_->capsule, 1);
}

void ProgressMpc::report_applied(double steer, double speed) {
  Impl& m = *impl_;
  Impl::Vec applied(kHistory, 0.0);
  if (!m.applied.empty()) applied = m.applied;
  const auto steer_history = applied.begin();
  const auto speed_history = applied.begin() + kSteerHistory;

  // Advance the steering estimate over the period this command starts, as the model's build_step does: the
  // plant sees the command issued `whole + 1` periods ago for the first `fraction` of the period, then the one
  // issued `whole` periods ago (this one when whole = 0), and follows it through a first-order lag.
  const double periods = kSteerDelay / kControlDt;
  const int whole = static_cast<int>(std::floor(periods + 1e-9));
  double fraction = periods - whole;
  if (fraction < 1e-9) fraction = 0.0;
  auto issued = [&](int periods_ago) { return periods_ago == 0 ? steer : steer_history[periods_ago - 1]; };
  auto follow = [&](double command, double duration) {
    const double target = kSteerGain * command + kSteerCubic * command * command * command + kSteerBias;
    m.steer_estimate += (target - m.steer_estimate) * (1.0 - std::exp(-duration / kSteerLatency));
  };
  if (fraction > 0.0) {
    follow(issued(whole + 1), fraction * kControlDt);
    follow(issued(whole), (1.0 - fraction) * kControlDt);
  } else {
    follow(issued(whole), kControlDt);
  }

  Impl::Vec next(kHistory);
  next[0] = steer;
  std::copy(steer_history, steer_history + kSteerHistory - 1, next.begin() + 1);
  next[kSteerHistory] = speed;
  std::copy(speed_history, speed_history + kSpeedHistory - 1, next.begin() + kSteerHistory + 1);
  m.applied = std::move(next);
}

Command ProgressMpc::step(const std::array<double, 6>& state, const double* steer) {
  Impl& m = *impl_;
  if (!m.have_track) throw std::runtime_error("set_track must be called before step");
  const auto start = Clock::now();

  Impl::Vec x0(kStateDim, 0.0);
  std::copy(state.begin(), state.end(), x0.begin());
  x0[kSteer] = steer ? *steer : m.steer_estimate;
  if (!m.applied.empty()) std::copy(m.applied.begin(), m.applied.begin() + kHistory, x0.begin() + kVehicleDim);
  const Impl::Vec applied = m.applied_or_rest(x0[kVx]);

  const Impl::Plan guess = m.shifted_guess(x0, applied);
  const Impl::Vec refs = m.references(guess.states);

  Impl::Plan solution;
  Command command;
  if (!m.solve(x0, applied, guess, refs, solution)) {
    ++m.stats.failures;
    const auto fallback = m.have_warm ? std::array<double, 2>{guess.controls[0], guess.controls[1]}
                                      : Impl::command_of(applied);
    m.have_warm = false;
    command = {fallback[0], fallback[1]};
  } else {
    command = m.accept(solution, refs);
  }
  m.stats.solve_seconds = seconds_since(start);
  return command;
}

std::vector<double> ProgressMpc::model_step(const std::vector<double>& z, const Command& command) const {
  if (z.size() != static_cast<size_t>(kStateDim)) throw std::invalid_argument("z must hold the state and history");
  const double u[2] = {command.steer, command.speed};
  std::vector<double> next(kStateDim);
  impl_->model(z.data(), u, next.data());
  return next;
}

double ProgressMpc::steer_estimate() const { return impl_->steer_estimate; }
const Stats& ProgressMpc::stats() const { return impl_->stats; }
const std::vector<std::array<double, 2>>& ProgressMpc::plan_world() const { return impl_->plan_world; }
const std::vector<std::array<double, 2>>& ProgressMpc::plan_frenet() const { return impl_->plan_frenet; }
int ProgressMpc::horizon() const { return kHorizon; }
double ProgressMpc::control_dt() const { return kControlDt; }
int ProgressMpc::history_length() const { return kHistory; }
double ProgressMpc::speed_max() const { return kSpeedMax; }
double ProgressMpc::steer_max() const { return kSteerMax; }
double ProgressMpc::cog_offset() const { return kLr; }

}  // namespace progress_mpc
