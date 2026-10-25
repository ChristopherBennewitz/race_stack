// Progress-maximizing MPC on acados SQP-RTI: the C++ port of f1tenth_parallel_sim's `mpc.progress_mpc`.
//
// The OCP (car model, horizon, weights, command limits) is baked into the generated acados code; the track
// is a runtime input. States are the simulator's [x, y, yaw, vx, vy, yaw_rate, steer] at the centre of
// gravity. Commands are [steer, speed] and respect the actuation manager's rate limits.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace progress_mpc {

constexpr int kVehicleStateDim = 7;

struct Command {
  double steer = 0.0;
  double speed = 0.0;
};

// What the last `step` did; mirrors the Python controller's `last_*` attributes.
struct Stats {
  double solve_seconds = 0.0;         // references + solver
  double solver_only_seconds = 0.0;   // the acados call alone
  int iterations = -1;
  std::string status = "not solved";
  double slack_max = 0.0;             // largest track-limit slack of the accepted plan (m)
  long failures = 0;                  // solver failures since construction
};

class ProgressMpc {
 public:
  // `track_spacing`: the centerline is resampled to this spacing (m); the reference search windows assume 0.1.
  // Zero or less keeps the given points as they are (they should then be ~0.1 m apart).
  explicit ProgressMpc(double track_spacing = 0.1);
  ~ProgressMpc();
  ProgressMpc(const ProgressMpc&) = delete;
  ProgressMpc& operator=(const ProgressMpc&) = delete;

  // A closed centerline in driving direction, with the track half-widths to its left and right.
  // Throws std::invalid_argument on fewer than 4 points, mismatched sizes or a degenerate path.
  void set_track(const std::vector<std::array<double, 2>>& centerline, const std::vector<double>& width_left,
                 const std::vector<double>& width_right);
  bool has_track() const;
  double track_length() const;

  // Forgets the plan and the command history. `centerline_index` seeds the reference search (-1: search all).
  void reset(int centerline_index = -1);

  // The command the actuation manager actually applied during the last control period. Call once per period,
  // after `step`; it anchors the rate limits and carries the steering transport delay.
  void report_applied(double steer, double speed);

  // One SQP-RTI iteration for the measured state [x, y, yaw, vx, vy, yaw_rate]; returns the next command.
  // `steer` is the road-wheel angle; when absent it comes from the internal steering model.
  Command step(const std::array<double, 6>& state, const double* steer = nullptr);

  // The model over one control period, with the command histories: z = [state(7), steering history, speed history] -> z'.
  std::vector<double> model_step(const std::vector<double>& z, const Command& command) const;

  // The road-wheel angle `step` falls back on, advanced by `report_applied` through the steering model.
  double steer_estimate() const;
  const Stats& stats() const;
  // Last accepted plan: (horizon + 1) rows of [x, y] in the map frame, and of [s, lateral] along the centerline.
  const std::vector<std::array<double, 2>>& plan_world() const;
  const std::vector<std::array<double, 2>>& plan_frenet() const;
  int horizon() const;
  double control_dt() const;
  int history_length() const;
  // Command limits baked into the OCP.
  double speed_max() const;
  double steer_max() const;
  // Distance from the rear axle (base_link) to the centre of gravity the state refers to (m).
  double cog_offset() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace progress_mpc
