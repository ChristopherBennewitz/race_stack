#include "policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include <torch/torch.h>

namespace progress_mpc {
namespace {
constexpr int kX = 0, kY = 1, kYaw = 2, kVx = 3, kYawRate = 5;
}  // namespace

void PolicyTrack::build(const Track& track, double frenet_window) {
  count = track.count;
  cx.resize(count), cy.resize(count), wl = track.w_left, wr = track.w_right;
  cxf.resize(count), cyf.resize(count), nxf.resize(count), nyf.resize(count), wlf.resize(count), wrf.resize(count);
  double shortest = std::numeric_limits<double>::infinity();
  for (int i = 0; i < count; ++i) {
    cx[i] = track.center[i][0], cy[i] = track.center[i][1];
    // The sim's normal points left of the driving direction: (-ty, tx).
    cxf[i] = static_cast<float>(cx[i]), cyf[i] = static_cast<float>(cy[i]);
    nxf[i] = static_cast<float>(-track.tangent[i][1]), nyf[i] = static_cast<float>(track.tangent[i][0]);
    wlf[i] = static_cast<float>(wl[i]), wrf[i] = static_cast<float>(wr[i]);
    const auto& b = track.center[wrap(i + 1, count)];
    shortest = std::min(shortest, std::hypot(b[0] - cx[i], b[1] - cy[i]));
  }
  window_half = std::max(1, static_cast<int>(std::ceil(frenet_window / shortest)));
}

torch::jit::Module load_policy_module(const std::string& path, PolicyMeta& meta) {
  torch::jit::Module module = torch::jit::load(path, torch::kCPU);
  module.eval();
  const torch::Tensor m = module.attr("meta").toTensor().to(torch::kDouble).contiguous();
  if (m.numel() != 9) throw std::runtime_error(path + ": unexpected policy metadata; re-export it");
  const double* v = m.data_ptr<double>();
  meta.n_waypoints = static_cast<int>(v[0]);
  meta.history_len = static_cast<int>(v[1]);
  meta.control_dt = v[2];
  meta.discount = v[3];
  meta.progress_weight = v[4];
  meta.scale = {v[5], v[6]};
  meta.bias = {v[7], v[8]};
  return module;
}

PolicyRunner::PolicyRunner(torch::jit::Module module, const PolicyMeta& meta, const PolicyTrack* track, int state_dim,
                           int steer_index)
    : module_(std::move(module)), meta_(meta), track_(track), nz_(state_dim), steer_index_(steer_index) {
  scalars_.resize(2);
  waypoints_.resize(static_cast<size_t>(meta.n_waypoints) * 4);
  history_.resize(static_cast<size_t>(meta.history_len) * kHistoryRowDim);
}

int PolicyRunner::full_search(float px, float py) const {
  int best = 0;
  float best_d = std::numeric_limits<float>::infinity();
  for (int i = 0; i < track_->count; ++i) {
    const float dx = track_->cxf[i] - px, dy = track_->cyf[i] - py;
    const float d = dx * dx + dy * dy;
    if (d < best_d) best_d = d, best = i;
  }
  return best;
}

int PolicyRunner::nearest(double x, double y, int prev) const {
  const PolicyTrack& t = *track_;
  const float px = static_cast<float>(x), py = static_cast<float>(y);
  if (prev < 0) return full_search(px, py);
  int best = 0;
  float best_d = std::numeric_limits<float>::infinity();
  for (int o = -t.window_half; o <= t.window_half; ++o) {
    const int i = wrap(prev + o, t.count);
    const float dx = t.cxf[i] - px, dy = t.cyf[i] - py;
    const float d = dx * dx + dy * dy;
    if (d < best_d) best_d = d, best = i;
  }
  const float dx = px - t.cxf[best], dy = py - t.cyf[best];
  const float e = dx * t.nxf[best] + dy * t.nyf[best];
  const float half_width = e >= 0.0f ? t.wlf[best] : t.wrf[best];
  const float dist = std::sqrt(dx * dx + dy * dy);
  if (std::fabs(e) > half_width + 0.25f || dist > half_width + 0.5f) return full_search(px, py);
  return best;
}

void PolicyRunner::observe(const double* xa, int idx, const double* past) {
  const PolicyTrack& t = *track_;
  const double x = xa[kX], y = xa[kY], yaw = xa[kYaw];
  const double cs = std::cos(yaw), sn = std::sin(yaw);
  for (int j = 0; j < meta_.n_waypoints; ++j) {
    const int i = wrap(idx + j + 1, t.count);
    const double rx = t.cx[i] - x, ry = t.cy[i] - y;
    double* w = &waypoints_[static_cast<size_t>(j) * 4];
    w[0] = rx * cs + ry * sn;
    w[1] = -rx * sn + ry * cs;
    w[2] = t.wl[i];
    w[3] = t.wr[i];
  }
  const size_t rows = static_cast<size_t>(meta_.history_len - 1) * kHistoryRowDim;
  std::memcpy(history_.data(), past, sizeof(double) * rows);
  double* newest = &history_[rows];
  newest[0] = xa[kVx], newest[1] = xa[kYawRate], newest[2] = xa[nz_], newest[3] = xa[steer_index_];
  scalars_[0] = xa[kVx];
  scalars_[1] = xa[kYawRate];
}

std::array<double, 2> PolicyRunner::act(const double* xa, int idx, const double* past) {
  observe(xa, idx, past);
  torch::NoGradGuard no_grad;
  const auto opts = torch::TensorOptions().dtype(torch::kDouble);
  const torch::Tensor scalars = torch::from_blob(scalars_.data(), {1, 2}, opts);
  const torch::Tensor waypoints = torch::from_blob(waypoints_.data(), {1, meta_.n_waypoints, 4}, opts);
  const torch::Tensor history = torch::from_blob(history_.data(), {1, meta_.history_len, kHistoryRowDim}, opts);
  const torch::Tensor mean = module_.run_method("act", scalars, waypoints, history).toTensor().contiguous();
  const double* m = mean.data_ptr<double>();
  const double speed = meta_.bias[0] + meta_.scale[0] * std::clamp(m[0], -1.0, 1.0);
  const double steer = meta_.bias[1] + meta_.scale[1] * std::clamp(m[1], -1.0, 1.0);
  return {steer, speed};
}

double PolicyRunner::value(const double* xa, int idx, const double* past, double* grad) {
  observe(xa, idx, past);
  const auto opts = torch::TensorOptions().dtype(torch::kDouble);
  if (!grad) {
    torch::NoGradGuard no_grad;
    const torch::Tensor scalars = torch::from_blob(scalars_.data(), {1, 2}, opts);
    const torch::Tensor waypoints = torch::from_blob(waypoints_.data(), {1, meta_.n_waypoints, 4}, opts);
    const torch::Tensor history = torch::from_blob(history_.data(), {1, meta_.history_len, kHistoryRowDim}, opts);
    return module_.run_method("value", scalars, waypoints, history).toTensor().item<double>();
  }
  torch::Tensor scalars = torch::from_blob(scalars_.data(), {1, 2}, opts).clone().set_requires_grad(true);
  torch::Tensor waypoints =
      torch::from_blob(waypoints_.data(), {1, meta_.n_waypoints, 4}, opts).clone().set_requires_grad(true);
  torch::Tensor history =
      torch::from_blob(history_.data(), {1, meta_.history_len, kHistoryRowDim}, opts).clone().set_requires_grad(true);
  const torch::Tensor out = module_.run_method("value", scalars, waypoints, history).toTensor();
  const auto grads = torch::autograd::grad({out.sum()}, {scalars, waypoints, history});
  const torch::Tensor gs = grads[0].contiguous(), gw = grads[1].contiguous(), gh = grads[2].contiguous();
  const double* g_scalars = gs.data_ptr<double>();
  const double* g_wp = gw.data_ptr<double>();
  const double* g_hist = gh.data_ptr<double>();

  std::fill(grad, grad + nz_ + 1, 0.0);
  // The scalars and the newest history row are the state's own velocity, yaw rate and last commands.
  const double* newest = g_hist + static_cast<size_t>(meta_.history_len - 1) * kHistoryRowDim;
  grad[kVx] += g_scalars[0] + newest[0];
  grad[kYawRate] += g_scalars[1] + newest[1];
  grad[nz_] += newest[2];
  grad[steer_index_] += newest[3];
  // The waypoints move with the car's pose.
  const PolicyTrack& t = *track_;
  const double yaw = xa[kYaw], cs = std::cos(yaw), sn = std::sin(yaw);
  for (int j = 0; j < meta_.n_waypoints; ++j) {
    const int i = wrap(idx + j + 1, t.count);
    const double rx = t.cx[i] - xa[kX], ry = t.cy[i] - xa[kY];
    const double gx = g_wp[static_cast<size_t>(j) * 4], gy = g_wp[static_cast<size_t>(j) * 4 + 1];
    grad[kX] += gx * -cs + gy * sn;
    grad[kY] += gx * -sn + gy * -cs;
    grad[kYaw] += gx * (-rx * sn + ry * cs) + gy * (-rx * cs - ry * sn);
  }
  return out.item<double>();
}

}  // namespace progress_mpc
