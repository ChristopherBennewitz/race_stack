// AC4MPC-RTI (Reiter et al., "AC4MPC", Alg. 2/3) on the progress MPC: the C++ port of f1tenth_parallel_sim's
// `mpc.ac4mpc` (its `native` variant), with the PPO networks run by libtorch.
//
// The OCP is the progress MPC's, with the critic's linearized value as an extra terminal term (generated_ac4mpc/).
// Per step an actor rollout, an active and a parallel acados solver each make a candidate plan; `ac4eval` scores
// them on the model (OCP cost, the critic at the end) and the cheapest is applied. The three branches run on their
// own threads, each with its own libtorch module and acados solver.
//
// Modes, as in the sim: "critic" (one solver, the critic's gradient in its terminal cost), "actor" (the actor's
// rollout seeds and competes with the solvers) and "full" (both).
#pragma once

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "progress_mpc/progress_mpc.hpp"

namespace progress_mpc {

enum class Ac4Mode { kCritic, kActor, kFull };

struct Ac4Config {
  Ac4Mode mode = Ac4Mode::kFull;
  double value_weight = 0.2;      // the paper's beta: scale on the critic in the QP (1: gamma^N times its value)
  int reinit_period = 5;          // steps between re-seeding the parallel solver with the actor's rollout
  double eval_correction = 0.0;   // alpha: actor feedback on a candidate's commands in ac4eval
  int eval_rollout = 0;           // R: actor steps past the horizon in ac4eval
  bool threads = true;            // branches on their own threads (actor and full modes)
  int torch_threads = 1;          // libtorch's intra-op threads per branch; one is fastest for these small networks
  double track_spacing = 0.1;     // see ProgressMpc
};

struct Ac4Stats : Stats {
  std::string candidate;                   // the branch that won: mpc_active, mpc_parallel, actor, or fallback
  double cost_active = 0.0, cost_parallel = 0.0, cost_actor = 0.0;  // ac4eval of each candidate (inf: none)
  double actor_seconds = 0.0, active_seconds = 0.0, parallel_seconds = 0.0;  // wall time of each branch
};

class Ac4Mpc {
 public:
  // `policy_path`: the TorchScript module from `scripts/export_policy.py` of the sim repo.
  Ac4Mpc(const std::string& policy_path, const Ac4Config& config = Ac4Config());
  ~Ac4Mpc();
  Ac4Mpc(const Ac4Mpc&) = delete;
  Ac4Mpc& operator=(const Ac4Mpc&) = delete;

  // As ProgressMpc.
  void set_track(const std::vector<std::array<double, 2>>& centerline, const std::vector<double>& width_left,
                 const std::vector<double>& width_right);
  bool has_track() const;
  double track_length() const;
  void reset(int centerline_index = -1);
  void report_applied(double steer, double speed);
  Command step(const std::array<double, 6>& state, const double* steer = nullptr);
  std::vector<double> model_step(const std::vector<double>& z, const Command& command) const;
  double steer_estimate() const;
  const Ac4Stats& stats() const;
  const std::vector<std::array<double, 2>>& plan_world() const;
  const std::vector<std::array<double, 2>>& plan_frenet() const;
  const std::map<std::string, long>& candidate_wins() const;
  int horizon() const;
  double control_dt() const;
  int history_length() const;
  double speed_max() const;
  // The most the commands change per control period: steer, speed up, speed down (the actuation manager's rates).
  std::array<double, 3> step_limits() const;
  double steer_max() const;
  double clearance_margin() const;
  double cog_offset() const;

  // The policy on one augmented MPC state (vehicle state, histories, last speed command), for tests: the action
  // `{steer, speed}`, the value and its gradient, and the policy's centerline projection. `past` holds
  // policy history_len - 1 rows of (vel, yaw rate, speed command, steer command).
  std::array<double, 2> policy_act(const std::vector<double>& xa, int idx, const std::vector<double>& past);
  double policy_value(const std::vector<double>& xa, int idx, const std::vector<double>& past,
                      std::vector<double>* grad);
  int policy_nearest(double x, double y, int prev) const;
  int policy_history_length() const;
  int policy_waypoints() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace progress_mpc
