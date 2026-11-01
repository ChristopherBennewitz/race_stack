// The PPO actor and critic on the car's MPC states, run with libtorch. Mirrors `mpc.ac4mpc._Policy` of the sim repo.
//
// The networks are a TorchScript module exported by the sim repo's `scripts/export_policy.py`; the observation
// (the lookahead waypoints in the body frame, the history window) is built here from the MPC's augmented state,
// and the critic's gradient over that state is the autograd gradient over the observation chained by hand through
// the observation's geometry.
#pragma once

#include <torch/script.h>

#include <array>
#include <string>
#include <vector>

#include "track.hpp"

namespace progress_mpc {

// What the exporter recorded about the checkpoint (its `meta` buffer).
struct PolicyMeta {
  int n_waypoints = 0, history_len = 0;
  double control_dt = 0.0, discount = 0.0, progress_weight = 0.0;
  std::array<double, 2> scale{}, bias{};  // (speed, steer): action = bias + scale * clamp(mean, -1, 1)
};

// The track as the policy sees it: float32 copies for the projection (as the sim's `Track.to_frenet` computes it),
// float64 for the observation.
struct PolicyTrack {
  int count = 0, window_half = 0;
  std::vector<float> cxf, cyf, nxf, nyf, wlf, wrf;
  std::vector<double> cx, cy, wl, wr;
  void build(const Track& track, double frenet_window = 5.0);
};

constexpr int kHistoryRowDim = 4;  // vel, yaw rate, speed command, steer command

// One thread's view of the policy: its own copy of the module and scratch buffers.
class PolicyRunner {
 public:
  // `state_dim` is the MPC state without the last speed command (the vehicle state and the command histories);
  // the augmented state `xa` that the calls take has that one more.
  PolicyRunner(torch::jit::Module module, const PolicyMeta& meta, const PolicyTrack* track, int state_dim,
               int steer_index);

  // `Track.to_frenet` around `prev`, in float32 (all points when prev < 0).
  int nearest(double x, double y, int prev) const;

  // The deterministic action `{steer, speed}` at augmented state `xa`, with `past` (history_len - 1 rows) before
  // the state's own history row.
  std::array<double, 2> act(const double* xa, int idx, const double* past);

  // The critic's value, and with `grad` (size state_dim + 1) its gradient over `xa`.
  double value(const double* xa, int idx, const double* past, double* grad);

  const PolicyMeta& meta() const { return meta_; }

 private:
  void observe(const double* xa, int idx, const double* past);
  int full_search(float px, float py) const;

  torch::jit::Module module_;
  PolicyMeta meta_;
  const PolicyTrack* track_;
  int nz_, steer_index_;
  std::vector<double> scalars_, waypoints_, history_;
};

// Loads the exporter's module and reads its metadata.
torch::jit::Module load_policy_module(const std::string& path, PolicyMeta& meta);

}  // namespace progress_mpc
