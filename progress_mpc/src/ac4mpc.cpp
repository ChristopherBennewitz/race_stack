#include "progress_mpc/ac4mpc.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

#include <ATen/Parallel.h>

extern "C" {
#include "ac4mpc_casadi.h"
}
#include "acados_solver_ac4mpc_critic.h"
#include "ac4mpc_config.h"
#include "policy.hpp"
#include "track.hpp"

namespace progress_mpc {
namespace {

using namespace ac4mpc::generated;
using Vec = std::vector<double>;
using Clock = std::chrono::steady_clock;

constexpr int kStateDim = kVehicleDim + kHistory;  // vehicle state, then the steering and the speed history, newest first
constexpr int kSpeedHistory = kHistory - kSteerHistory;
constexpr int kParamDim = kRefDim + 2 * kAugDim;  // the stage reference, then the critic's point and gradient
constexpr int kPrevSteer = kVehicleDim;           // the newest steering command in the state
constexpr double kInf = std::numeric_limits<double>::infinity();

static_assert(kAugDim == kStateDim + 1, "the OCP state is the model state and the last speed command");
static_assert(kAugDim == AC4MPC_CRITIC_NX, "the generated OCP does not match the exported step function");
static_assert(kParamDim == AC4MPC_CRITIC_NP, "the generated OCP's parameters are not the reference and the critic");
static_assert(kHorizon == AC4MPC_CRITIC_N, "horizon mismatch between ac4mpc_config.h and the acados code");

enum StateIdx { kX = 0, kY = 1, kYaw = 2, kVx = 3, kVy = 4, kYawRate = 5, kSteer = 6 };

double seconds_since(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

// ------------------------------------------------------------------------------------------ CasADi functions

void step_model(const double* z, const double* u, double* z_next) {
  const double* arg[ac4mpc_step_SZ_ARG + 1] = {z, u};
  double* res[ac4mpc_step_SZ_RES + 1] = {z_next};
  casadi_int iw[ac4mpc_step_SZ_IW + 1];
  double w[ac4mpc_step_SZ_W + 1];
  if (ac4mpc_step(arg, res, iw, w, 0) != 0) throw std::runtime_error("model step evaluation failed");
}

double rate_cost(const double* aug, const double* u) {
  const double* arg[ac4mpc_rate_SZ_ARG + 1] = {aug, u};
  double cost = 0.0;
  double* res[ac4mpc_rate_SZ_RES + 1] = {&cost};
  casadi_int iw[ac4mpc_rate_SZ_IW + 1];
  double w[ac4mpc_rate_SZ_W + 1];
  ac4mpc_rate(arg, res, iw, w, 0);
  return cost;
}

double stage_cost(const double* aug, const double* u, const double* ref) {
  const double* arg[ac4mpc_stage_SZ_ARG + 1] = {aug, u, ref};
  double cost = 0.0;
  double* res[ac4mpc_stage_SZ_RES + 1] = {&cost};
  casadi_int iw[ac4mpc_stage_SZ_IW + 1];
  double w[ac4mpc_stage_SZ_W + 1];
  ac4mpc_stage(arg, res, iw, w, 0);
  return cost;
}

double terminal_cost(const double* aug, const double* ref) {
  const double* arg[ac4mpc_terminal_SZ_ARG + 1] = {aug, ref};
  double cost = 0.0;
  double* res[ac4mpc_terminal_SZ_RES + 1] = {&cost};
  casadi_int iw[ac4mpc_terminal_SZ_IW + 1];
  double w[ac4mpc_terminal_SZ_W + 1];
  ac4mpc_terminal(arg, res, iw, w, 0);
  return cost;
}

// ------------------------------------------------------------------------------------------ acados

// One acados instance; each branch that solves owns one, so they can run at the same time.
class Solver {
 public:
  struct Result {
    bool ok = false;
    Vec states, controls;  // (horizon + 1, kStateDim), (horizon, 2)
    double seconds = 0.0, slack = 0.0;
    int iterations = 0, status = 0;
  };

  Solver() {
    capsule_ = ac4mpc_critic_acados_create_capsule();
    if (ac4mpc_critic_acados_create(capsule_) != 0) {
      ac4mpc_critic_acados_free_capsule(capsule_);
      throw std::runtime_error("acados failed to create the AC4MPC solver");
    }
    config_ = ac4mpc_critic_acados_get_nlp_config(capsule_);
    dims_ = ac4mpc_critic_acados_get_nlp_dims(capsule_);
    in_ = ac4mpc_critic_acados_get_nlp_in(capsule_);
    out_ = ac4mpc_critic_acados_get_nlp_out(capsule_);
    solver_ = ac4mpc_critic_acados_get_nlp_solver(capsule_);
  }
  ~Solver() {
    ac4mpc_critic_acados_free(capsule_);
    ac4mpc_critic_acados_free_capsule(capsule_);
  }
  Solver(const Solver&) = delete;
  Solver& operator=(const Solver&) = delete;

  void reset() { ac4mpc_critic_acados_reset(capsule_, 1); }

  // One SQP-RTI iteration from the guess; `params` is (horizon + 1, kParamDim).
  Result run(const Vec& x0, double last_speed, const Vec& states, const Vec& controls, const Vec& params) {
    const int n = kHorizon;
    Vec x0_aug(x0);
    x0_aug.push_back(last_speed);
    ocp_nlp_constraints_model_set(config_, dims_, in_, 0, "lbx", x0_aug.data());
    ocp_nlp_constraints_model_set(config_, dims_, in_, 0, "ubx", x0_aug.data());
    Vec x_k(kAugDim);
    for (int k = 0; k <= n; ++k) {
      ac4mpc_critic_acados_update_params(capsule_, k, const_cast<double*>(&params[static_cast<size_t>(k) * kParamDim]),
                                         kParamDim);
      std::copy(&states[static_cast<size_t>(k) * kStateDim], &states[static_cast<size_t>(k + 1) * kStateDim],
                x_k.begin());
      // The last speed command at stage k: the applied one, then the plan's own.
      x_k[kStateDim] = (k == 0) ? last_speed : controls[2 * (k - 1) + 1];
      ocp_nlp_out_set(config_, dims_, out_, k, "x", x_k.data());
      if (k < n) ocp_nlp_out_set(config_, dims_, out_, k, "u", const_cast<double*>(&controls[2 * k]));
    }

    Result result;
    const auto start = Clock::now();
    result.status = ac4mpc_critic_acados_solve(capsule_);
    result.seconds = seconds_since(start);
    ocp_nlp_get(solver_, "sqp_iter", &result.iterations);
    if (result.status != 0) {
      reset();  // acados keeps its iterates; a NaN would poison every later step
      return result;
    }

    double su[AC4MPC_CRITIC_NS > AC4MPC_CRITIC_NSN ? AC4MPC_CRITIC_NS : AC4MPC_CRITIC_NSN];
    for (int k = 1; k <= n; ++k) {
      ocp_nlp_out_get(config_, dims_, out_, k, "su", su);
      const int ns = (k < n) ? AC4MPC_CRITIC_NS : AC4MPC_CRITIC_NSN;
      for (int i = 0; i < ns; ++i) result.slack = std::max(result.slack, su[i]);
    }
    result.states.resize(static_cast<size_t>(n + 1) * kStateDim);
    result.controls.resize(static_cast<size_t>(n) * 2);
    for (int k = 0; k <= n; ++k) {
      ocp_nlp_out_get(config_, dims_, out_, k, "x", x_k.data());
      std::copy(x_k.begin(), x_k.begin() + kStateDim, result.states.begin() + static_cast<size_t>(k) * kStateDim);
      if (k < n) ocp_nlp_out_get(config_, dims_, out_, k, "u", &result.controls[2 * k]);
    }
    result.ok = true;
    return result;
  }

 private:
  ac4mpc_critic_solver_capsule* capsule_ = nullptr;
  ocp_nlp_config* config_ = nullptr;
  ocp_nlp_dims* dims_ = nullptr;
  ocp_nlp_in* in_ = nullptr;
  ocp_nlp_out* out_ = nullptr;
  ocp_nlp_solver* solver_ = nullptr;
};

// ------------------------------------------------------------------------------------------ plans and branches

struct Guess {
  Vec states, controls;  // (horizon + 1, kStateDim), (horizon, 2)
};

// A chosen plan kept for the next step's warm start: the plan, the centerline index of its last stage, and the policy
// history of the step it was made in.
struct Memory {
  bool valid = false;
  Vec states, controls, history;
  int last_idx = -1;
};

struct Branch {
  bool has_plan = false;
  Vec states, controls, refs;  // refs (horizon + 1, kRefDim)
  std::vector<int> indices;
  double cost = kInf, solver_seconds = 0.0, slack = 0.0, wall_seconds = 0.0;
  Vec first_guess;  // the warm start's first command, the fallback if every solve fails
  int failures = 0, iterations = 0;
};

// What a step's branches share: the measurement and the policy history; read-only while they run.
struct StepInput {
  Vec x0, applied, history;  // history: (policy history_len, kHistoryRowDim)
  int prev = -1;             // the centerline index of the vehicle
};

struct Settings {
  Ac4Config config;
  bool uses_actor = false, uses_critic = false;
  double value_weight = 0.0, eval_value_weight = 0.0;
  std::array<double, 2> step_lo{}, step_hi{}, lower{}, upper{};  // (steer, speed)
};

// A branch's own machinery: its policy runner and (for the solving branches) its solver. A worker is used by one
// thread at a time.
class Worker {
 public:
  Worker(const Settings* settings, const Track* track, PolicyRunner policy, bool with_solver)
      : s_(settings), track_(track), policy_(std::move(policy)) {
    if (with_solver) solver_ = std::make_unique<Solver>();
  }

  PolicyRunner& policy() { return policy_; }
  Solver* solver() { return solver_.get(); }
  Memory& memory() { return memory_; }

  // ---- the model

  Vec simulate(const double* x0, const Vec& controls) const {
    const int n = static_cast<int>(controls.size() / 2);
    Vec states(static_cast<size_t>(n + 1) * kStateDim);
    std::copy(x0, x0 + kStateDim, states.begin());
    for (int k = 0; k < n; ++k)
      step_model(&states[static_cast<size_t>(k) * kStateDim], &controls[2 * k],
                 &states[static_cast<size_t>(k + 1) * kStateDim]);
    return states;
  }

  std::array<double, 2> limit(const double* command, const double* previous) const {
    std::array<double, 2> out;
    for (int j = 0; j < 2; ++j) {
      const double rated = std::clamp(command[j], previous[j] + s_->step_lo[j], previous[j] + s_->step_hi[j]);
      out[j] = std::clamp(rated, s_->lower[j], s_->upper[j]);
    }
    return out;
  }

  // The straight launch used before there is a plan: accelerate towards 2 m/s within the rate limits.
  Guess launch(const StepInput& in) const {
    Guess g;
    g.states.assign(static_cast<size_t>(kHorizon + 1) * kStateDim, 0.0);
    g.controls.assign(static_cast<size_t>(kHorizon) * 2, 0.0);
    std::copy(in.x0.begin(), in.x0.end(), g.states.begin());
    std::array<double, 2> cmd{in.applied[0], in.applied[kSteerHistory]};
    const double target[2] = {0.0, 2.0};
    for (int k = 0; k < kHorizon; ++k) {
      for (int i = 0; i < 2; ++i)
        cmd[i] += std::clamp(target[i] - cmd[i], s_->step_lo[i], s_->step_hi[i]);
      g.controls[2 * k] = cmd[0];
      g.controls[2 * k + 1] = cmd[1];
      step_model(&g.states[static_cast<size_t>(k) * kStateDim], cmd.data(),
                 &g.states[static_cast<size_t>(k + 1) * kStateDim]);
    }
    return g;
  }

  // ---- the policy's observation history

  // `history` followed by the rows the env pushes while driving `states`/`controls` for `n` steps.
  Vec history_rows(const Vec& history, const Vec& states, const Vec& controls, int n) const {
    const int rows = policy_.meta().history_len;
    Vec out(static_cast<size_t>(rows + n) * kHistoryRowDim);
    std::memcpy(out.data(), history.data(), sizeof(double) * kHistoryRowDim * rows);
    for (int k = 0; k < n; ++k) {
      double* r = &out[static_cast<size_t>(rows + k) * kHistoryRowDim];
      const double* st = &states[static_cast<size_t>(k + 1) * kStateDim];
      r[0] = st[kVx], r[1] = st[kYawRate], r[2] = controls[2 * k + 1], r[3] = controls[2 * k];
    }
    return out;
  }

  void augment(const double* state, double prev_speed, double* xa) const {
    std::memcpy(xa, state, sizeof(double) * kStateDim);
    xa[kStateDim] = prev_speed;
  }

  // ---- the branches' pieces (mpc.ac4mpc)

  // `_closed_loop`: the model under `feedforward + gain * pi` (the actor alone past the feedforward).
  void closed_loop(const StepInput& in, int steps, const double* feedforward, int n_ff, double gain, Vec& states,
                   Vec& controls, std::vector<int>& indices) {
    const int hist = policy_.meta().history_len;
    states.assign(static_cast<size_t>(steps + 1) * kStateDim, 0.0);
    controls.assign(static_cast<size_t>(steps) * 2, 0.0);
    indices.assign(static_cast<size_t>(steps) + 1, 0);
    Vec rows(static_cast<size_t>(hist + steps) * kHistoryRowDim, 0.0);
    std::memcpy(rows.data(), in.history.data(), sizeof(double) * kHistoryRowDim * hist);
    std::copy(in.x0.begin(), in.x0.end(), states.begin());
    std::array<double, 2> command{in.applied[0], in.applied[kSteerHistory]};
    int idx = in.prev, first = 0;
    if (feedforward && gain == 0.0) {
      first = n_ff;
      std::copy(feedforward, feedforward + 2 * n_ff, controls.begin());
      const Vec prefix = simulate(in.x0.data(), Vec(feedforward, feedforward + 2 * n_ff));
      std::copy(prefix.begin(), prefix.end(), states.begin());
      const Vec prefix_rows = history_rows(in.history, states, controls, n_ff);
      std::copy(prefix_rows.begin(), prefix_rows.end(), rows.begin());
      project(*track_, states.data(), kStateDim, n_ff + 1, in.prev, nullptr, indices.data());
      command = {feedforward[2 * (n_ff - 1)], feedforward[2 * (n_ff - 1) + 1]};
      idx = indices[n_ff];
    }
    Vec xa(kAugDim);
    for (int k = first; k < steps; ++k) {
      const double* s = &states[static_cast<size_t>(k) * kStateDim];
      idx = policy_.nearest(s[kX], s[kY], idx);
      indices[k] = idx;
      augment(s, command[1], xa.data());
      std::array<double, 2> action = policy_.act(xa.data(), idx, &rows[static_cast<size_t>(k) * kHistoryRowDim]);
      if (feedforward && k < n_ff) {
        action[0] = feedforward[2 * k] + gain * action[0];
        action[1] = feedforward[2 * k + 1] + gain * action[1];
      }
      command = limit(action.data(), command.data());
      controls[2 * k] = command[0], controls[2 * k + 1] = command[1];
      step_model(s, command.data(), &states[static_cast<size_t>(k + 1) * kStateDim]);
      double* r = &rows[static_cast<size_t>(hist + k) * kHistoryRowDim];
      const double* s1 = &states[static_cast<size_t>(k + 1) * kStateDim];
      r[0] = s1[kVx], r[1] = s1[kYawRate], r[2] = command[1], r[3] = command[0];
    }
    const double* last = &states[static_cast<size_t>(steps) * kStateDim];
    indices[steps] = policy_.nearest(last[kX], last[kY], idx);
  }

  // `_ac4eval` (Alg. 3): the cost of a (possibly infeasible) plan, simulated with actor feedback, plus the critic.
  double ac4eval(const StepInput& in, const Vec& states, const Vec& controls) {
    const Ac4Config& cfg = s_->config;
    const int n = kHorizon, steps = kHorizon + cfg.eval_rollout;
    Vec feedforward(controls.begin(), controls.begin() + 2 * n);
    if (cfg.eval_correction > 0.0) {  // `_plan_actions`
      const Vec rows = history_rows(in.history, states, controls, n);
      int idx = in.prev;
      Vec xa(kAugDim);
      for (int k = 0; k < n; ++k) {
        const double* s = &states[static_cast<size_t>(k) * kStateDim];
        idx = policy_.nearest(s[kX], s[kY], idx);
        augment(s, k == 0 ? in.applied[kSteerHistory] : controls[2 * (k - 1) + 1], xa.data());
        const auto action = policy_.act(xa.data(), idx, &rows[static_cast<size_t>(k) * kHistoryRowDim]);
        feedforward[2 * k] -= cfg.eval_correction * action[0];
        feedforward[2 * k + 1] -= cfg.eval_correction * action[1];
      }
    }
    Vec sim_states, sim_controls;
    std::vector<int> indices;
    closed_loop(in, steps, feedforward.data(), n, cfg.eval_correction, sim_states, sim_controls, indices);
    Vec refs(static_cast<size_t>(steps + 1) * kRefDim);
    project(*track_, sim_states.data(), kStateDim, steps + 1, in.prev, refs.data(), nullptr);
    Vec aug(static_cast<size_t>(steps + 1) * kAugDim);
    for (int k = 0; k <= steps; ++k)
      augment(&sim_states[static_cast<size_t>(k) * kStateDim], k == 0 ? in.applied[kSteerHistory] : sim_controls[2 * (k - 1) + 1],
              &aug[static_cast<size_t>(k) * kAugDim]);

    double cost = rate_cost(aug.data(), sim_controls.data());
    for (int k = 1; k < steps; ++k)
      cost += stage_cost(&aug[static_cast<size_t>(k) * kAugDim], &sim_controls[2 * k], &refs[static_cast<size_t>(k) * kRefDim]);
    cost += terminal_cost(&aug[static_cast<size_t>(steps) * kAugDim], &refs[static_cast<size_t>(steps) * kRefDim]);
    if (s_->uses_critic) {
      const Vec rows = history_rows(in.history, sim_states, sim_controls, steps);
      cost -= s_->eval_value_weight * policy_.value(&aug[static_cast<size_t>(steps) * kAugDim], indices[steps],
                                                    &rows[static_cast<size_t>(steps) * kHistoryRowDim], nullptr);
    }
    return cost;
  }

  // `_shift`: the plan's controls without the first, then one more: an actor step when the actor runs, else a repeat.
  // The plan is re-simulated first, so the actor acts on a terminal state the commands actually reach. Returns the
  // last stage's centerline index.
  int shift(const Memory& plan, Vec& out_controls) {
    const int n = kHorizon;
    out_controls.assign(plan.controls.begin() + 2, plan.controls.end());
    const double* last_command = &plan.controls[2 * (n - 1)];
    if (!s_->uses_actor) {
      out_controls.insert(out_controls.end(), last_command, last_command + 2);
      return plan.last_idx;
    }
    const Vec sim = simulate(plan.states.data(), plan.controls);
    const double* last = &sim[static_cast<size_t>(n) * kStateDim];
    const int idx = policy_.nearest(last[kX], last[kY], plan.last_idx);
    const Vec rows = history_rows(plan.history, sim, plan.controls, n);
    Vec xa(kAugDim);
    augment(last, last_command[1], xa.data());
    const auto action = policy_.act(xa.data(), idx, &rows[static_cast<size_t>(n) * kHistoryRowDim]);
    const auto command = limit(action.data(), last_command);
    out_controls.push_back(command[0]);
    out_controls.push_back(command[1]);
    return idx;
  }

  // `_active_guess`: the last chosen plan shifted and re-simulated, or the actor's rollout (or a launch) without one.
  Guess active_guess(const StepInput& in, const Memory& previous, const Guess* rollout) {
    Guess g;
    if (previous.valid) {
      shift(previous, g.controls);
      g.states = simulate(in.x0.data(), g.controls);
      return g;
    }
    if (rollout) return *rollout;
    if (s_->uses_actor) {
      std::vector<int> indices;
      closed_loop(in, kHorizon, nullptr, 0, 1.0, g.states, g.controls, indices);
      return g;
    }
    return launch(in);
  }

  // One solver iteration from `guess`: references, the critic's linearization, the solve, and (with `evaluate`)
  // ac4eval of the result.
  Branch solver_branch(const StepInput& in, const Guess& guess, bool evaluate) {
    const int n = kHorizon;
    Branch b;
    b.first_guess.assign(guess.controls.begin(), guess.controls.begin() + 2);
    b.refs.resize(static_cast<size_t>(n + 1) * kRefDim);
    b.indices.resize(n + 1);
    project(*track_, guess.states.data(), kStateDim, n + 1, in.prev, b.refs.data(), b.indices.data());

    Vec params(static_cast<size_t>(n + 1) * kParamDim, 0.0);
    for (int k = 0; k <= n; ++k)
      std::copy(&b.refs[static_cast<size_t>(k) * kRefDim], &b.refs[static_cast<size_t>(k + 1) * kRefDim],
                &params[static_cast<size_t>(k) * kParamDim]);
    if (s_->uses_critic) {
      Vec xa(kAugDim), grad(kAugDim);
      augment(&guess.states[static_cast<size_t>(n) * kStateDim], guess.controls[2 * (n - 1) + 1], xa.data());
      const Vec rows = history_rows(in.history, guess.states, guess.controls, n);
      policy_.value(xa.data(), b.indices[n], &rows[static_cast<size_t>(n) * kHistoryRowDim], grad.data());
      double* p = &params[static_cast<size_t>(n) * kParamDim + kRefDim];
      std::copy(xa.begin(), xa.end(), p);
      for (int i = 0; i < kAugDim; ++i) p[kAugDim + i] = s_->value_weight * grad[i];
    }

    Solver::Result r = solver_->run(in.x0, in.applied[kSteerHistory], guess.states, guess.controls, params);
    b.solver_seconds = r.seconds;
    b.iterations = r.iterations;
    if (!r.ok) {
      b.failures = 1;
      return b;
    }
    b.has_plan = true;
    b.states = std::move(r.states);
    b.controls = std::move(r.controls);
    b.slack = r.slack;
    if (evaluate) b.cost = ac4eval(in, b.states, b.controls);
    return b;
  }

  // The parallel solver: from the actor's rollout every `reinit_period` steps, else from its own last plan.
  Branch parallel_branch(const StepInput& in, bool reseed, const Guess* rollout) {
    Guess guess;
    if (reseed || !memory_.valid) {
      if (rollout) {
        guess = *rollout;
      } else {
        std::vector<int> indices;
        closed_loop(in, kHorizon, nullptr, 0, 1.0, guess.states, guess.controls, indices);
      }
    } else {
      shift(memory_, guess.controls);
      guess.states = simulate(in.x0.data(), guess.controls);
    }
    Branch b = solver_branch(in, guess, true);
    memory_.valid = b.has_plan;
    if (b.has_plan) {
      memory_.states = b.states;
      memory_.controls = b.controls;
      memory_.last_idx = b.indices[kHorizon];
      memory_.history = in.history;
    }
    return b;
  }

  // The actor's rollout as a candidate.
  Branch actor_branch(const StepInput& in) {
    Branch b;
    closed_loop(in, kHorizon, nullptr, 0, 1.0, b.states, b.controls, b.indices);
    b.refs.resize(static_cast<size_t>(kHorizon + 1) * kRefDim);
    b.indices.resize(kHorizon + 1);
    project(*track_, b.states.data(), kStateDim, kHorizon + 1, in.prev, b.refs.data(), b.indices.data());
    b.has_plan = true;
    b.cost = ac4eval(in, b.states, b.controls);
    return b;
  }

 private:
  const Settings* s_;
  const Track* track_;
  PolicyRunner policy_;
  std::unique_ptr<Solver> solver_;
  Memory memory_;  // the parallel solver's last plan
};

// A thread that runs one task at a time on request.
class Thread {
 public:
  Thread() : thread_([this] { loop(); }) {}
  ~Thread() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      quit_ = true;
    }
    cv_.notify_all();
    thread_.join();
  }
  Thread(const Thread&) = delete;
  Thread& operator=(const Thread&) = delete;

  void submit(std::function<void()> task) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      task_ = std::move(task);
      error_ = nullptr;
      busy_ = true;
    }
    cv_.notify_all();
  }

  // Waits for the submitted task; rethrows what it threw.
  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return !busy_; });
    if (error_) std::rethrow_exception(error_);
  }

 private:
  void loop() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
      cv_.wait(lock, [this] { return quit_ || busy_; });
      if (quit_) return;
      std::function<void()> task = std::move(task_);
      lock.unlock();
      try {
        task();
      } catch (...) {
        lock.lock();
        error_ = std::current_exception();
        lock.unlock();
      }
      lock.lock();
      busy_ = false;
      done_.notify_all();
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_, done_;
  std::function<void()> task_;
  std::exception_ptr error_;
  bool busy_ = false, quit_ = false;
  std::thread thread_;
};

}  // namespace

// ------------------------------------------------------------------------------------------ the controller

struct Ac4Mpc::Impl {
  Impl(const std::string& policy_path, const Ac4Config& cfg) {
    settings.config = cfg;
    settings.uses_actor = cfg.mode != Ac4Mode::kCritic;
    settings.uses_critic = cfg.mode != Ac4Mode::kActor;
    at::set_num_threads(cfg.torch_threads);
    try {
      at::set_num_interop_threads(1);
    } catch (const std::exception&) {  // can only be set once, before any parallel work
    }

    module = load_policy_module(policy_path, meta);
    if (std::abs(meta.control_dt - kControlDt) > 1e-9)
      throw std::runtime_error("the policy acts every " + std::to_string(meta.control_dt) +
                               " s, the exported MPC every " + std::to_string(kControlDt) + " s");
    const double scale = kQProgress / meta.progress_weight;
    settings.value_weight = cfg.value_weight * scale * std::pow(meta.discount, kHorizon);
    settings.eval_value_weight = scale * std::pow(meta.discount, kHorizon + cfg.eval_rollout);
    settings.step_lo = {-kSteerStep, -kSpeedStepDown};
    settings.step_hi = {kSteerStep, kSpeedStepUp};
    settings.lower = {-kSteerMax, 0.0};
    settings.upper = {kSteerMax, kSpeedMax};
    plan_world.assign(kHorizon + 1, {0.0, 0.0});
    plan_frenet.assign(kHorizon + 1, {0.0, 0.0});
  }

  Settings settings;
  PolicyMeta meta;
  torch::jit::Module module;
  Track track;
  PolicyTrack policy_track;
  bool have_track = false;
  double track_spacing = 0.1;

  // The branches. `active` solves from the last chosen plan; `parallel` from its own, re-seeded by the actor; the
  // actor only rolls out. In critic mode there is `active` alone.
  std::unique_ptr<Worker> active, parallel, actor;
  std::unique_ptr<Thread> active_thread, parallel_thread;

  Vec applied;  // steering history, then speed history (both newest first); empty before the first report
  Vec history;  // the policy's observation history, (history_len, kHistoryRowDim)
  bool have_issued = false;
  std::array<double, 2> issued{};  // (speed, steer), the last command returned
  Memory chosen;                    // the plan the active solver continues from
  int prev_idx = -1;
  long iteration = 0;
  double steer_estimate = 0.0;
  Ac4Stats stats;
  std::map<std::string, long> wins;
  std::vector<std::array<double, 2>> plan_world, plan_frenet;

  void build_workers() {
    auto runner = [&] { return PolicyRunner(module.clone(), meta, &policy_track, kStateDim, kPrevSteer); };
    active = std::make_unique<Worker>(&settings, &track, runner(), true);
    if (settings.uses_actor) {
      parallel = std::make_unique<Worker>(&settings, &track, runner(), true);
      actor = std::make_unique<Worker>(&settings, &track, runner(), false);
      if (settings.config.threads) {
        active_thread = std::make_unique<Thread>();
        parallel_thread = std::make_unique<Thread>();
      }
    }
    history.assign(static_cast<size_t>(meta.history_len) * kHistoryRowDim, 0.0);
  }

  // Runs every network and the solvers once, so the first control step does not pay for lazy initialization.
  void warm_up() {
    Vec xa(kAugDim, 0.0), grad(kAugDim), past(static_cast<size_t>(meta.history_len - 1) * kHistoryRowDim, 0.0);
    xa[kX] = track.center[0][0], xa[kY] = track.center[0][1];
    xa[kYaw] = std::atan2(track.tangent[0][1], track.tangent[0][0]);
    xa[kVx] = 1.0;
    for (Worker* w : {active.get(), parallel.get(), actor.get()}) {
      if (!w) continue;
      for (int i = 0; i < 4; ++i) {
        w->policy().act(xa.data(), 0, past.data());
        w->policy().value(xa.data(), 0, past.data(), nullptr);
        w->policy().value(xa.data(), 0, past.data(), grad.data());
      }
    }
  }

  Vec applied_or_rest(double vx) const {
    if (!applied.empty()) return applied;
    Vec rest(kHistory, 0.0);
    rest[kSteerHistory] = vx;
    return rest;
  }

  void forget_plans() {
    chosen.valid = false;
    if (parallel) parallel->memory().valid = false;
  }

  // The branches of one step, by name, in the tie-break order.
  struct Branches {
    Branch active, parallel, actor;
    bool have_parallel = false, have_actor = false;
  };

  Branches run_branches(const StepInput& in) {
    Branches out;
    const Ac4Config& cfg = settings.config;
    if (!settings.uses_actor) {
      const auto start = Clock::now();
      Guess guess = active->active_guess(in, chosen, nullptr);
      out.active = active->solver_branch(in, guess, false);
      out.active.wall_seconds = seconds_since(start);
      return out;
    }
    out.have_parallel = out.have_actor = true;
    const bool reseed = iteration % cfg.reinit_period == 0;
    if (active_thread) {
      // Every branch on its own thread; the longest, the actor's, on this one.
      active_thread->submit([&] {
        const auto start = Clock::now();
        Guess guess = active->active_guess(in, chosen, nullptr);
        out.active = active->solver_branch(in, guess, true);
        out.active.wall_seconds = seconds_since(start);
      });
      parallel_thread->submit([&] {
        const auto start = Clock::now();
        out.parallel = parallel->parallel_branch(in, reseed, nullptr);
        out.parallel.wall_seconds = seconds_since(start);
      });
      std::exception_ptr error;
      try {
        const auto start = Clock::now();
        out.actor = actor->actor_branch(in);
        out.actor.wall_seconds = seconds_since(start);
      } catch (...) {
        error = std::current_exception();
      }
      active_thread->wait();
      parallel_thread->wait();
      if (error) std::rethrow_exception(error);
      return out;
    }
    // One thread: the actor's rollout seeds both solvers.
    auto start = Clock::now();
    out.actor = actor->actor_branch(in);
    out.actor.wall_seconds = seconds_since(start);
    const Guess rollout{out.actor.states, out.actor.controls};
    start = Clock::now();
    Guess guess = active->active_guess(in, chosen, &rollout);
    out.active = active->solver_branch(in, guess, true);
    out.active.wall_seconds = seconds_since(start);
    start = Clock::now();
    out.parallel = parallel->parallel_branch(in, reseed, &rollout);
    out.parallel.wall_seconds = seconds_since(start);
    return out;
  }
};

Ac4Mpc::Ac4Mpc(const std::string& policy_path, const Ac4Config& config)
    : impl_(std::make_unique<Impl>(policy_path, config)) {
  impl_->track_spacing = config.track_spacing;
}

Ac4Mpc::~Ac4Mpc() = default;

void Ac4Mpc::set_track(const std::vector<std::array<double, 2>>& centerline, const std::vector<double>& width_left,
                       const std::vector<double>& width_right) {
  Impl& m = *impl_;
  m.track = build_track(centerline, width_left, width_right, m.track_spacing);
  m.policy_track.build(m.track);
  m.have_track = true;
  m.prev_idx = -1;
  if (!m.active) {
    m.build_workers();
    m.warm_up();
  }
  m.forget_plans();
}

bool Ac4Mpc::has_track() const { return impl_->have_track; }
double Ac4Mpc::track_length() const { return impl_->track.length; }

void Ac4Mpc::reset(int centerline_index) {
  Impl& m = *impl_;
  m.prev_idx = centerline_index;
  m.applied.clear();
  m.steer_estimate = 0.0;
  std::fill(m.history.begin(), m.history.end(), 0.0);
  m.have_issued = false;
  m.iteration = 0;
  m.forget_plans();
  for (Worker* w : {m.active.get(), m.parallel.get()})
    if (w) w->solver()->reset();
}

void Ac4Mpc::report_applied(double steer, double speed) {
  Impl& m = *impl_;
  Vec applied(kHistory, 0.0);
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

  Vec next(kHistory);
  next[0] = steer;
  std::copy(steer_history, steer_history + kSteerHistory - 1, next.begin() + 1);
  next[kSteerHistory] = speed;
  std::copy(speed_history, speed_history + kSpeedHistory - 1, next.begin() + kSteerHistory + 1);
  m.applied = std::move(next);
}

Command Ac4Mpc::step(const std::array<double, 6>& state, const double* steer) {
  Impl& m = *impl_;
  if (!m.have_track) throw std::runtime_error("set_track must be called before step");
  const auto start = Clock::now();

  StepInput in;
  in.x0.assign(kStateDim, 0.0);
  std::copy(state.begin(), state.end(), in.x0.begin());
  in.x0[kSteer] = steer ? *steer : m.steer_estimate;
  if (!m.applied.empty()) std::copy(m.applied.begin(), m.applied.begin() + kHistory, in.x0.begin() + kVehicleDim);
  in.applied = m.applied_or_rest(in.x0[kVx]);
  if (m.have_issued) {  // the row the env pushes after each step: the measurement with the command just issued
    std::copy(m.history.begin() + kHistoryRowDim, m.history.end(), m.history.begin());
    double* newest = &m.history[m.history.size() - kHistoryRowDim];
    newest[0] = in.x0[kVx], newest[1] = in.x0[kYawRate], newest[2] = m.issued[0], newest[3] = m.issued[1];
  }
  in.history = m.history;
  in.prev = m.prev_idx = project(m.track, in.x0.data(), kStateDim, 1, m.prev_idx, nullptr, nullptr);

  Impl::Branches branches = m.run_branches(in);

  Ac4Stats& stats = m.stats;
  stats.failures += branches.active.failures + branches.parallel.failures;
  stats.solver_only_seconds = branches.active.solver_seconds + branches.parallel.solver_seconds;
  stats.iterations = branches.active.iterations;
  stats.cost_active = branches.active.cost;
  stats.cost_parallel = branches.have_parallel ? branches.parallel.cost : kInf;
  stats.cost_actor = branches.have_actor ? branches.actor.cost : kInf;
  stats.active_seconds = branches.active.wall_seconds;
  stats.parallel_seconds = branches.parallel.wall_seconds;
  stats.actor_seconds = branches.actor.wall_seconds;

  // The cheapest candidate that has a plan; ties go to the first.
  const Branch* chosen = nullptr;
  const char* name = "fallback";
  const std::array<std::pair<const char*, const Branch*>, 3> order = {{
      {"mpc_active", &branches.active},
      {"mpc_parallel", branches.have_parallel ? &branches.parallel : nullptr},
      {"actor", branches.have_actor ? &branches.actor : nullptr},
  }};
  for (const auto& [label, branch] : order) {
    if (branch && branch->has_plan && (!chosen || branch->cost < chosen->cost)) chosen = branch, name = label;
  }

  Command command;
  if (!chosen) {
    stats.status = "no candidate";
    const Vec& guess = branches.active.first_guess;
    const std::array<double, 2> fallback = (!m.chosen.valid || guess.size() < 2)
                                               ? std::array<double, 2>{in.applied[0], in.applied[kSteerHistory]}
                                               : std::array<double, 2>{guess[0], guess[1]};
    command = {fallback[0], fallback[1]};
    m.forget_plans();
  } else {
    stats.status = "Solve_Succeeded";
    stats.slack_max = chosen->slack;
    command = {chosen->controls[0], chosen->controls[1]};
    // The recorded plan: the planned commands re-simulated from the measured state, a trajectory the model can drive.
    const Vec warm = m.active->simulate(chosen->states.data(), chosen->controls);
    for (int k = 0; k <= kHorizon; ++k) {
      const double* r = &chosen->refs[static_cast<size_t>(k) * kRefDim];
      const double px = warm[static_cast<size_t>(k) * kStateDim + kX], py = warm[static_cast<size_t>(k) * kStateDim + kY];
      const double ox = px - r[0], oy = py - r[1];
      m.plan_world[k] = {px, py};
      m.plan_frenet[k] = {r[4] + r[2] * ox + r[3] * oy, -r[3] * ox + r[2] * oy};
    }
    m.chosen.valid = true;
    m.chosen.states = chosen->states;
    m.chosen.controls = chosen->controls;
    m.chosen.last_idx = chosen->indices[kHorizon];
    m.chosen.history = in.history;
  }
  stats.candidate = name;
  ++m.wins[name];
  ++m.iteration;
  m.issued = {command.speed, command.steer};
  m.have_issued = true;
  stats.solve_seconds = seconds_since(start);
  return command;
}

std::vector<double> Ac4Mpc::model_step(const std::vector<double>& z, const Command& command) const {
  if (z.size() != static_cast<size_t>(kStateDim)) throw std::invalid_argument("z must hold the state and history");
  const double u[2] = {command.steer, command.speed};
  std::vector<double> next(kStateDim);
  step_model(z.data(), u, next.data());
  return next;
}

std::array<double, 2> Ac4Mpc::policy_act(const std::vector<double>& xa, int idx, const std::vector<double>& past) {
  if (!impl_->active) throw std::runtime_error("set_track must be called first");
  return impl_->active->policy().act(xa.data(), idx, past.data());
}

double Ac4Mpc::policy_value(const std::vector<double>& xa, int idx, const std::vector<double>& past,
                            std::vector<double>* grad) {
  if (!impl_->active) throw std::runtime_error("set_track must be called first");
  if (grad) grad->assign(kAugDim, 0.0);
  return impl_->active->policy().value(xa.data(), idx, past.data(), grad ? grad->data() : nullptr);
}

int Ac4Mpc::policy_nearest(double x, double y, int prev) const {
  if (!impl_->active) throw std::runtime_error("set_track must be called first");
  return impl_->active->policy().nearest(x, y, prev);
}

int Ac4Mpc::policy_history_length() const { return impl_->meta.history_len; }
int Ac4Mpc::policy_waypoints() const { return impl_->meta.n_waypoints; }
double Ac4Mpc::steer_estimate() const { return impl_->steer_estimate; }
const Ac4Stats& Ac4Mpc::stats() const { return impl_->stats; }
const std::vector<std::array<double, 2>>& Ac4Mpc::plan_world() const { return impl_->plan_world; }
const std::vector<std::array<double, 2>>& Ac4Mpc::plan_frenet() const { return impl_->plan_frenet; }
const std::map<std::string, long>& Ac4Mpc::candidate_wins() const { return impl_->wins; }
int Ac4Mpc::horizon() const { return kHorizon; }
double Ac4Mpc::control_dt() const { return kControlDt; }
int Ac4Mpc::history_length() const { return kHistory; }
double Ac4Mpc::speed_max() const { return kSpeedMax; }
std::array<double, 3> Ac4Mpc::step_limits() const { return {kSteerStep, kSpeedStepUp, kSpeedStepDown}; }
double Ac4Mpc::steer_max() const { return kSteerMax; }
double Ac4Mpc::clearance_margin() const { return kClearanceMargin; }
double Ac4Mpc::cog_offset() const { return kLr; }

}  // namespace progress_mpc
