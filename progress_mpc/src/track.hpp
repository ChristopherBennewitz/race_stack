// The centerline both controllers drive on: building it from the stack's points, and projecting planned positions onto it.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace progress_mpc {

constexpr int kRefDim = 7;  // centre x, y; tangent x, y; arc length at centre; left, right width
constexpr int kVehicleSearchLo = -15, kVehicleSearchHi = 45;  // centerline points around the last match
constexpr int kStageSearch = 10;  // points a later stage's reference may advance past the previous one

inline int wrap(int i, int n) { return ((i % n) + n) % n; }

// A closed centerline with per-point tangent, arc length and widths, at uniform spacing.
struct Track {
  int count = 0;
  double length = 0.0;
  std::vector<std::array<double, 2>> center, tangent;
  std::vector<double> cum_s, w_left, w_right;
};

inline Track build_track(std::vector<std::array<double, 2>> pts, std::vector<double> wl, std::vector<double> wr,
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

// Projects `n` planned positions (x, y at `positions[k * stride]`) onto the centerline: the first within a window
// around `prev` (everywhere when prev < 0), later stages may only advance, by a bounded step. Writes each stage's
// reference parameters and centerline index (either may be null) and returns the first stage's index.
inline int project(const Track& track, const double* positions, int stride, int n, int prev, double* refs,
                   int* indices) {
  int idx = prev, first = prev;
  double s_prev = 0.0;
  for (int k = 0; k < n; ++k) {
    const double px = positions[k * stride], py = positions[k * stride + 1];
    int lo = 0, hi = track.count;
    if (idx >= 0) {
      lo = (k == 0) ? kVehicleSearchLo : 0;
      hi = (k == 0) ? kVehicleSearchHi : kStageSearch + 1;
    }
    double best = std::numeric_limits<double>::infinity();
    int best_idx = 0;
    for (int o = lo; o < hi; ++o) {
      const int c = (idx >= 0) ? wrap(idx + o, track.count) : o;
      const double dx = track.center[c][0] - px, dy = track.center[c][1] - py;
      const double d = dx * dx + dy * dy;
      if (d < best) {
        best = d;
        best_idx = c;
      }
    }
    idx = best_idx;
    if (k == 0) first = idx;
    double s = track.cum_s[idx];
    if (k > 0) s += track.length * std::nearbyint((s_prev - s) / track.length);
    s_prev = s;
    if (refs) {
      double* r = refs + k * kRefDim;
      r[0] = track.center[idx][0];
      r[1] = track.center[idx][1];
      r[2] = track.tangent[idx][0];
      r[3] = track.tangent[idx][1];
      r[4] = s;
      r[5] = track.w_left[idx];
      r[6] = track.w_right[idx];
    }
    if (indices) indices[k] = idx;
  }
  return first;
}

}  // namespace progress_mpc
