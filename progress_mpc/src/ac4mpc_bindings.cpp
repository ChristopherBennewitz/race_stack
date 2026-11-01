#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <stdexcept>

#include "progress_mpc/ac4mpc.hpp"

namespace py = pybind11;
using progress_mpc::Ac4Config;
using progress_mpc::Ac4Mode;
using progress_mpc::Ac4Mpc;
using progress_mpc::Command;

namespace {

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

std::vector<double> flat(const Array& a, py::ssize_t expected, const char* name) {
  if (expected >= 0 && a.size() != expected) throw std::invalid_argument(std::string(name) + " has the wrong size");
  return std::vector<double>(a.data(), a.data() + a.size());
}

py::array_t<double> rows(const std::vector<std::array<double, 2>>& points) {
  py::array_t<double> out({static_cast<py::ssize_t>(points.size()), static_cast<py::ssize_t>(2)});
  auto view = out.mutable_unchecked<2>();
  for (size_t i = 0; i < points.size(); ++i) {
    view(i, 0) = points[i][0];
    view(i, 1) = points[i][1];
  }
  return out;
}

Ac4Mode parse_mode(const std::string& mode) {
  if (mode == "critic") return Ac4Mode::kCritic;
  if (mode == "actor") return Ac4Mode::kActor;
  if (mode == "full") return Ac4Mode::kFull;
  throw std::invalid_argument("mode must be 'critic', 'actor' or 'full'");
}

}  // namespace

PYBIND11_MODULE(_ac4mpc, m) {
  m.doc() = "AC4MPC-RTI (actor and critic guided progress MPC) in C++, with libtorch and branches on threads.";

  py::class_<Ac4Mpc>(m, "Ac4Mpc")
      .def(
          py::init([](const std::string& policy, const std::string& mode, double value_weight, int reinit_period,
                      double eval_correction, int eval_rollout, bool threads, int torch_threads,
                      double track_spacing) {
            Ac4Config cfg;
            cfg.mode = parse_mode(mode);
            cfg.value_weight = value_weight;
            cfg.reinit_period = reinit_period;
            cfg.eval_correction = eval_correction;
            cfg.eval_rollout = eval_rollout;
            cfg.threads = threads;
            cfg.torch_threads = torch_threads;
            cfg.track_spacing = track_spacing;
            if (reinit_period < 1) throw std::invalid_argument("reinit_period must be at least 1");
            return std::make_unique<Ac4Mpc>(policy, cfg);
          }),
          py::arg("policy"), py::arg("mode") = "full", py::arg("value_weight") = 0.2, py::arg("reinit_period") = 5,
          py::arg("eval_correction") = 0.0, py::arg("eval_rollout") = 0, py::arg("threads") = true,
          py::arg("torch_threads") = 1, py::arg("track_spacing") = 0.1,
          "`policy`: TorchScript module from the sim repo's scripts/export_policy.py. `mode`: critic, actor or full.")
      .def(
          "set_track",
          [](Ac4Mpc& self, const Array& centerline, const Array& width_left, const Array& width_right) {
            if (centerline.ndim() != 2 || centerline.shape(1) != 2)
              throw std::invalid_argument("centerline must have shape (n, 2)");
            const auto n = centerline.shape(0);
            std::vector<std::array<double, 2>> pts(n);
            auto view = centerline.unchecked<2>();
            for (py::ssize_t i = 0; i < n; ++i) pts[i] = {view(i, 0), view(i, 1)};
            const auto left = flat(width_left, n, "width_left"), right = flat(width_right, n, "width_right");
            // The first call runs the networks, and their autograd engine must not be entered with the GIL held.
            py::gil_scoped_release release;
            self.set_track(pts, left, right);
          },
          py::arg("centerline"), py::arg("width_left"), py::arg("width_right"),
          "Closed centerline (n, 2) in driving direction with the track half-widths to its left and right.")
      .def_property_readonly("has_track", &Ac4Mpc::has_track)
      .def_property_readonly("track_length", &Ac4Mpc::track_length)
      .def("reset", &Ac4Mpc::reset, py::arg("centerline_index") = -1, "Forgets the plans and the command history.")
      .def("report_applied", &Ac4Mpc::report_applied, py::arg("steer"), py::arg("speed"),
           "The command the actuation manager applied during the last control period; call once per period after step().")
      .def(
          "step",
          [](Ac4Mpc& self, const Array& state, std::optional<double> steer) {
            const auto s = flat(state, 6, "state");
            const std::array<double, 6> x{s[0], s[1], s[2], s[3], s[4], s[5]};
            Command c;
            {
              py::gil_scoped_release release;
              c = steer ? self.step(x, &*steer) : self.step(x);
            }
            return py::make_tuple(c.steer, c.speed);
          },
          py::arg("state"), py::arg("steer") = py::none(),
          "One AC4MPC-RTI iteration for the CoG state [x, y, yaw, vx, vy, yaw_rate]; returns (steer, speed).")
      .def(
          "model_step",
          [](const Ac4Mpc& self, const Array& z, double steer, double speed) {
            const auto next = self.model_step(flat(z, 7 + self.history_length(), "z"), Command{steer, speed});
            return Array(static_cast<py::ssize_t>(next.size()), next.data());
          },
          py::arg("z"), py::arg("steer"), py::arg("speed"))
      .def(
          "policy_act",
          [](Ac4Mpc& self, const Array& xa, int idx, const Array& past) {
            const auto x = flat(xa, -1, "xa"), p = flat(past, -1, "past");
            std::array<double, 2> a;
            {
              py::gil_scoped_release release;
              a = self.policy_act(x, idx, p);
            }
            return py::make_tuple(a[0], a[1]);
          },
          py::arg("xa"), py::arg("idx"), py::arg("past"), "The actor's (steer, speed) at an augmented state, for tests.")
      .def(
          "policy_value",
          [](Ac4Mpc& self, const Array& xa, int idx, const Array& past) {
            std::vector<double> grad;
            const auto x = flat(xa, -1, "xa"), p = flat(past, -1, "past");
            double v;
            {
              py::gil_scoped_release release;  // the autograd engine must not be entered with the GIL held
              v = self.policy_value(x, idx, p, &grad);
            }
            return py::make_tuple(v, Array(static_cast<py::ssize_t>(grad.size()), grad.data()));
          },
          py::arg("xa"), py::arg("idx"), py::arg("past"), "The critic's value and its gradient over xa, for tests.")
      .def("policy_nearest", &Ac4Mpc::policy_nearest, py::arg("x"), py::arg("y"), py::arg("prev") = -1)
      .def_property_readonly("policy_history_length", &Ac4Mpc::policy_history_length)
      .def_property_readonly("policy_waypoints", &Ac4Mpc::policy_waypoints)
      .def_property_readonly("plan_world", [](const Ac4Mpc& s) { return rows(s.plan_world()); })
      .def_property_readonly("plan_frenet", [](const Ac4Mpc& s) { return rows(s.plan_frenet()); })
      .def_property_readonly("candidate_wins", &Ac4Mpc::candidate_wins)
      .def_property_readonly("stats",
                             [](const Ac4Mpc& s) {
                               const auto& st = s.stats();
                               py::dict d;
                               d["solve_seconds"] = st.solve_seconds;
                               d["solver_only_seconds"] = st.solver_only_seconds;
                               d["iterations"] = st.iterations;
                               d["status"] = st.status;
                               d["slack_max"] = st.slack_max;
                               d["failures"] = st.failures;
                               d["candidate"] = st.candidate;
                               d["cost_active"] = st.cost_active;
                               d["cost_parallel"] = st.cost_parallel;
                               d["cost_actor"] = st.cost_actor;
                               d["active_seconds"] = st.active_seconds;
                               d["parallel_seconds"] = st.parallel_seconds;
                               d["actor_seconds"] = st.actor_seconds;
                               return d;
                             })
      .def_property_readonly("steer_estimate", &Ac4Mpc::steer_estimate)
      .def_property_readonly("horizon", &Ac4Mpc::horizon)
      .def_property_readonly("control_dt", &Ac4Mpc::control_dt)
      .def_property_readonly("history_length", &Ac4Mpc::history_length)
      .def_property_readonly("speed_max", &Ac4Mpc::speed_max)
      .def_property_readonly("step_limits", &Ac4Mpc::step_limits)
      .def_property_readonly("steer_max", &Ac4Mpc::steer_max)
      .def_property_readonly("clearance_margin", &Ac4Mpc::clearance_margin)
      .def_property_readonly("cog_offset", &Ac4Mpc::cog_offset);
}
