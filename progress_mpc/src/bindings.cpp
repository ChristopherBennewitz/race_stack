#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <stdexcept>

#include "progress_mpc/progress_mpc.hpp"

namespace py = pybind11;
using progress_mpc::Command;
using progress_mpc::ProgressMpc;

namespace {

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

std::vector<double> flat(const Array& a, py::ssize_t expected, const char* name) {
  if (a.size() != expected) throw std::invalid_argument(std::string(name) + " has the wrong size");
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

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "Progress-maximizing MPC (acados SQP-RTI) from f1tenth_parallel_sim, in C++.";

  py::class_<ProgressMpc>(m, "ProgressMpc")
      .def(py::init<double>(), py::arg("track_spacing") = 0.1)
      .def(
          "set_track",
          [](ProgressMpc& self, const Array& centerline, const Array& width_left, const Array& width_right) {
            if (centerline.ndim() != 2 || centerline.shape(1) != 2)
              throw std::invalid_argument("centerline must have shape (n, 2)");
            const auto n = centerline.shape(0);
            std::vector<std::array<double, 2>> pts(n);
            auto view = centerline.unchecked<2>();
            for (py::ssize_t i = 0; i < n; ++i) pts[i] = {view(i, 0), view(i, 1)};
            self.set_track(pts, flat(width_left, n, "width_left"), flat(width_right, n, "width_right"));
          },
          py::arg("centerline"), py::arg("width_left"), py::arg("width_right"),
          "Closed centerline (n, 2) in driving direction with the track half-widths to its left and right.")
      .def_property_readonly("has_track", &ProgressMpc::has_track)
      .def_property_readonly("track_length", &ProgressMpc::track_length)
      .def("reset", &ProgressMpc::reset, py::arg("centerline_index") = -1,
           "Forgets the plan and the command history.")
      .def("report_applied", &ProgressMpc::report_applied, py::arg("steer"), py::arg("speed"),
           "The command the actuation manager applied during the last control period; call once per period after step().")
      .def(
          "step",
          [](ProgressMpc& self, const Array& state, std::optional<double> steer) {
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
          "One SQP-RTI iteration for the CoG state [x, y, yaw, vx, vy, yaw_rate] (map frame, body velocities); "
          "returns (steer, speed). The road-wheel angle comes from the internal steering model unless given.")
      .def(
          "model_step",
          [](const ProgressMpc& self, const Array& z, double steer, double speed) {
            const auto next = self.model_step(flat(z, 7 + self.history_length(), "z"), Command{steer, speed});
            return Array(static_cast<py::ssize_t>(next.size()), next.data());
          },
          py::arg("z"), py::arg("steer"), py::arg("speed"),
          "One control period of the model over [state(7), steering history]; returns the next z.")
      .def_property_readonly("plan_world", [](const ProgressMpc& s) { return rows(s.plan_world()); })
      .def_property_readonly("plan_frenet", [](const ProgressMpc& s) { return rows(s.plan_frenet()); })
      .def_property_readonly("stats",
                             [](const ProgressMpc& s) {
                               const auto& st = s.stats();
                               py::dict d;
                               d["solve_seconds"] = st.solve_seconds;
                               d["solver_only_seconds"] = st.solver_only_seconds;
                               d["iterations"] = st.iterations;
                               d["status"] = st.status;
                               d["slack_max"] = st.slack_max;
                               d["failures"] = st.failures;
                               return d;
                             })
      .def_property_readonly("steer_estimate", &ProgressMpc::steer_estimate)
      .def_property_readonly("horizon", &ProgressMpc::horizon)
      .def_property_readonly("control_dt", &ProgressMpc::control_dt)
      .def_property_readonly("history_length", &ProgressMpc::history_length)
      .def_property_readonly("speed_max", &ProgressMpc::speed_max)
      .def_property_readonly("steer_max", &ProgressMpc::steer_max)
      .def_property_readonly("cog_offset", &ProgressMpc::cog_offset);
}
