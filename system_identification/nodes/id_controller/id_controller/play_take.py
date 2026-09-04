#!/usr/bin/env python3
"""Play a feed-forward excitation with loose map-frame containment.

The identification signal remains feed-forward. A capped, low-bandwidth
steering correction prevents accumulated pose drift, while an occupancy-map
safety layer slows or stops before the car footprint reaches unknown/occupied
space. The nominal command, correction, requested command, and manager-applied
command are separate bag topics.

Exit codes: 0 completed, 2 bad arguments, 3 aborted.
"""
from __future__ import annotations

import math
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import Joy

from id_controller import containment
from id_controller import takes as takes_lib

EXIT_OK = 0
EXIT_ARGS = 2
EXIT_ABORT = 3


class TakePlayer(Node):
    """Publish one identification take with loose containment and hard safety."""

    def __init__(self) -> None:
        super().__init__("take_player")
        self.declare_parameter("take", "")
        self.declare_parameter("csv", "")
        self.declare_parameter("topic", "/drive")
        self.declare_parameter("pose_topic", "/car_state/odom")
        self.declare_parameter("map_topic", "/sysid/safety_map")
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("rate_hz", float(takes_lib.HZ))
        self.declare_parameter("countdown", 3.0)
        self.declare_parameter("tail_sec", 1.0)
        self.declare_parameter("require_deadman", True)
        self.declare_parameter("deadman_button", 5)
        self.declare_parameter("human_deadman_button", 4)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("containment_enabled", True)
        self.declare_parameter("wheelbase", takes_lib.L)
        for name, value in containment.DEFAULT_PARAMETERS.items():
            self.declare_parameter(name, value)

        take = str(self.get_parameter("take").value)
        csv_path = str(self.get_parameter("csv").value)
        if bool(take) == bool(csv_path):
            self.get_logger().error("set exactly one of 'take' or 'csv'")
            raise SystemExit(EXIT_ARGS)
        try:
            self.rows = (takes_lib.build(take) if take
                         else takes_lib.load_csv(csv_path))
        except (KeyError, ValueError, OSError) as exc:
            self.get_logger().error(str(exc))
            raise SystemExit(EXIT_ARGS)

        self.name = take or csv_path
        # Circular takes measure responses that determine their real radius and
        # angular progress, so track their geometry without imposing time phase.
        self.phase_independent_reference = (
            takes_lib.has_phase_independent_reference(take) if take else False)
        self.reference_steer = (takes_lib.reference_steering(take, self.rows)
                               if take else self.rows[:, 0].copy())
        self.hz = float(self.get_parameter("rate_hz").value)
        self.dt = 1.0 / self.hz
        self.frame = str(self.get_parameter("frame_id").value)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.figure_eight = (
            takes_lib.FigureEightSequencer(self.hz)
            if take == "M3_figure_eight" and not self.dry_run else None)
        self.containment_enabled = bool(
            self.get_parameter("containment_enabled").value)
        self.require_deadman = bool(self.get_parameter("require_deadman").value)
        self.deadman_button = int(self.get_parameter("deadman_button").value)
        self.human_deadman_button = int(
            self.get_parameter("human_deadman_button").value)
        self.wheelbase = float(self.get_parameter("wheelbase").value)
        self.pose_timeout = float(self.get_parameter("pose_timeout_sec").value)
        self.jump_distance = float(self.get_parameter("jump_distance").value)
        self.jump_yaw = float(self.get_parameter("jump_yaw").value)
        self.hard_clearance = float(self.get_parameter("hard_clearance").value)
        self.path_clearance = float(self.get_parameter("path_clearance").value)
        self.slow_clearance = float(self.get_parameter("slow_clearance").value)
        self.prediction_horizon = float(
            self.get_parameter("prediction_horizon").value)
        self.heading_gain = float(self.get_parameter("heading_gain").value)
        self.cross_track_gain = float(
            self.get_parameter("cross_track_gain").value)
        self.speed_softening = float(
            self.get_parameter("speed_softening").value)
        self.correction_max = float(self.get_parameter("correction_max").value)
        self.correction_tau = float(self.get_parameter("correction_tau").value)
        self.correction_rate = float(self.get_parameter("correction_rate").value)
        if not (0.0 < self.hard_clearance < self.path_clearance
                < self.slow_clearance):
            self.get_logger().error(
                "need 0 < hard_clearance < path_clearance < slow_clearance")
            raise SystemExit(EXIT_ARGS)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        path_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(
            AckermannDriveStamped, self.get_parameter("topic").value, qos)
        self.nominal_pub = self.create_publisher(
            AckermannDriveStamped, "/sysid/nominal_cmd", qos)
        self.correction_pub = self.create_publisher(
            AckermannDriveStamped, "/sysid/containment_correction", qos)
        self.path_pub = self.create_publisher(Path, "/sysid/reference_path", path_qos)
        self.create_subscription(Joy, "/joy", self._joy_cb, 10)
        self.create_subscription(
            Odometry, self.get_parameter("pose_topic").value, self._pose_cb, 20)
        self.create_subscription(
            OccupancyGrid, self.get_parameter("map_topic").value,
            self._map_cb, map_qos)

        self.deadman_held = False
        self.deadman_ok = not self.require_deadman
        self.deadman_armed = not self.require_deadman
        self.accept_deadman = False
        self.abort_reason = ""
        self.pose = None
        self.pose_received = None
        self.previous_pose = None
        self.unwrapped_yaw = None
        self.map_field = None
        self.map_error = ""
        self.reference_path = None
        self.figure_eight_path = None
        self.feedback = 0.0
        self.last_safety_scale = 1.0
        self.i = 0
        self.tail = 0
        self.n_tail = int(round(float(self.get_parameter("tail_sec").value) * self.hz))
        self.done = False
        self.aborted = False
        self.t_start = None
        self.t_end = None

        duration = len(self.rows) / self.hz
        mode = "contained" if self.containment_enabled else "OPEN-LOOP OVERRIDE"
        self.get_logger().info(
            f"{self.name}: {len(self.rows)} rows, {duration:.1f} s at "
            f"{self.hz:.0f} Hz [{mode}]"
            + (" [DRY RUN, publishing nothing]" if self.dry_run else ""))

        # Publish and validate the path before arming. This gives the operator
        # unlimited time to inspect the blue path in RViz while the car cannot
        # receive the identification command.
        if self.containment_enabled:
            self._prepare_containment()
        if self.require_deadman and not self.aborted:
            self.get_logger().info(
                f"hold joystick button {self.deadman_button} for the whole take; "
                "releasing it aborts")
            self._wait_for_deadman()
        countdown = float(self.get_parameter("countdown").value)
        if not self.aborted:
            self.get_logger().info(f"starting in {countdown:.0f} s -- clear the area")
            self._sleep_spinning(countdown)
        if self.aborted:
            self.done = True
            return
        self.t_start = time.monotonic()
        self.timer = self.create_timer(self.dt, self.tick)

    def _joy_cb(self, msg: Joy) -> None:
        held = (len(msg.buttons) > self.deadman_button
                and msg.buttons[self.deadman_button] == 1)
        human_held = (len(msg.buttons) > self.human_deadman_button
                      and msg.buttons[self.human_deadman_button] == 1)
        if self.require_deadman and human_held:
            self._abort(f"human deadman button {self.human_deadman_button} pressed")
            return
        self.deadman_held = held
        if self.require_deadman and self.deadman_armed and not held:
            self._abort(f"deadman button {self.deadman_button} released")
        if self.require_deadman and self.accept_deadman and held:
            self.deadman_ok = True
            self.deadman_armed = True

    def _pose_cb(self, msg: Odometry) -> None:
        if msg.header.frame_id and msg.header.frame_id != self.map_frame:
            self.map_error = (f"pose frame is {msg.header.frame_id!r}, expected "
                              f"{self.map_frame!r}")
            return
        q = msg.pose.pose.orientation
        current = (float(msg.pose.pose.position.x),
                   float(msg.pose.pose.position.y),
                   containment.yaw_from_quaternion(q.x, q.y, q.z, q.w),
                   float(msg.twist.twist.linear.x),
                   float(msg.twist.twist.linear.y),
                   float(msg.twist.twist.angular.z))
        if self.unwrapped_yaw is None:
            self.unwrapped_yaw = current[2]
        elif self.previous_pose is not None:
            self.unwrapped_yaw += containment.wrap_angle(
                current[2] - self.previous_pose[2])
        if self.previous_pose is not None and self.t_start is not None:
            distance = math.hypot(current[0] - self.previous_pose[0],
                                  current[1] - self.previous_pose[1])
            yaw_jump = abs(containment.wrap_angle(current[2] - self.previous_pose[2]))
            if distance > self.jump_distance or yaw_jump > self.jump_yaw:
                self._abort(
                    f"localization jump: {distance:.2f} m, {yaw_jump:.2f} rad")
        self.previous_pose = current
        self.pose = current
        self.pose_received = time.monotonic()

    def _map_cb(self, msg: OccupancyGrid) -> None:
        # Localization uses a saved static map. Building the distance field is
        # intentionally a one-time pre-take operation; repeating Dijkstra in a
        # control callback would add command latency whenever /map republishes.
        if self.map_field is not None:
            return
        if msg.header.frame_id and msg.header.frame_id != self.map_frame:
            self.map_error = (f"map frame is {msg.header.frame_id!r}, expected "
                              f"{self.map_frame!r}")
            return
        q = msg.info.origin.orientation
        try:
            self.map_field = containment.DistanceField.from_grid(
                msg.data, msg.info.width, msg.info.height, msg.info.resolution,
                msg.info.origin.position.x, msg.info.origin.position.y,
                containment.yaw_from_quaternion(q.x, q.y, q.z, q.w))
        except (ValueError, IndexError) as exc:
            self.map_error = f"invalid occupancy map: {exc}"

    def _abort(self, reason: str) -> None:
        if self.aborted:
            return
        self.aborted = True
        self.abort_reason = reason
        self.done = True
        self.get_logger().error(f"ABORT: {reason}")

    def _complete(self, reason: str = "take complete") -> None:
        """Stop command playback with a successful outcome."""
        if self.done:
            return
        self.t_end = time.monotonic()
        if hasattr(self, "timer"):
            self.timer.cancel()
        self.done = True
        self.get_logger().info(reason)

    def _wait_for_deadman(self) -> None:
        if self.deadman_held:
            self.get_logger().info(
                f"release button {self.deadman_button}, inspect the blue path, "
                "then press it again to arm")
        while rclpy.ok() and self.deadman_held and not self.aborted:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.accept_deadman = True
        self.get_logger().info(
            f"path ready; press button {self.deadman_button} to arm")
        while rclpy.ok() and not self.deadman_ok and not self.aborted:
            rclpy.spin_once(self, timeout_sec=0.1)

    def _sleep_spinning(self, sec: float) -> None:
        end = time.monotonic() + sec
        while rclpy.ok() and not self.aborted and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _prepare_containment(self) -> None:
        timeout = float(self.get_parameter("localization_wait_sec").value)
        self.get_logger().info("waiting for map-frame pose and occupancy map...")
        end = time.monotonic() + timeout
        while (rclpy.ok() and not self.aborted and not self.map_error
               and time.monotonic() < end
               and (self.pose is None or self.map_field is None)):
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.map_error:
            self._abort(self.map_error)
            return
        if self.pose is None or self.map_field is None:
            missing = "pose" if self.pose is None else "occupancy map"
            self._abort(f"no {missing} after {timeout:.1f} s")
            return
        local_path = containment.integrate_reference(
            self.rows, self.reference_steer, self.dt, self.wheelbase)
        self.reference_path = containment.transform_path(
            local_path, self.pose[0], self.pose[1], self.pose[2])
        minimum = self.map_field.path_clearance(self.reference_path)
        self._publish_reference_path()
        if minimum < self.path_clearance:
            self._abort(
                f"reference path has only {minimum:.2f} m map clearance; need "
                f"{self.path_clearance:.2f} m -- reposition or rotate the car")
            return
        self.get_logger().info(
            f"reference path accepted; minimum map clearance {minimum:.2f} m")

    def _publish_reference_path(self, path=None) -> None:
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        for x, y, yaw in self.reference_path if path is None else path:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.z = math.sin(0.5 * yaw)
            pose.pose.orientation.w = math.cos(0.5 * yaw)
            msg.poses.append(pose)
        self.path_pub.publish(msg)

    def _start_figure_eight_lobe(self, steer: float, speed: float) -> None:
        """Anchor and safety-check one M3 circle at the measured crossover pose."""
        if self.pose is None or self.map_field is None:
            self._abort("cannot anchor figure-eight lobe without pose and map")
            return
        duration = takes_lib.revolution_seconds(abs(steer), abs(speed))
        n_steps = max(1, int(round(duration * self.hz)))
        commands = np.column_stack((
            np.full(n_steps + 1, steer), np.full(n_steps + 1, speed)))
        local_path = containment.integrate_reference(
            commands, commands[:, 0], self.dt, self.wheelbase)
        self.figure_eight_path = containment.transform_path(
            local_path, self.pose[0], self.pose[1], self.pose[2])
        minimum = self.map_field.path_clearance(self.figure_eight_path)
        if minimum < self.path_clearance:
            self._abort(
                f"next figure-eight lobe has only {minimum:.2f} m map clearance; "
                f"need {self.path_clearance:.2f} m")
            return
        self.feedback = 0.0
        self._publish_reference_path(self.figure_eight_path)
        self.get_logger().info(
            f"figure-eight lobe anchored; minimum map clearance {minimum:.2f} m")

    def _publish_ackermann(self, publisher, steer: float, speed: float) -> None:
        if self.dry_run:
            return
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame
        msg.drive.steering_angle = float(steer)
        msg.drive.speed = float(speed)
        publisher.publish(msg)

    def send(self, steer: float, speed: float) -> None:
        self._publish_ackermann(self.pub, steer, speed)

    def _contained_command(self, nominal_steer: float,
                           nominal_speed: float) -> tuple[float, float, float, float]:
        if self.pose is None or self.pose_received is None:
            self._abort("localization disappeared")
            return nominal_steer, 0.0, 0.0, 0.0
        age = time.monotonic() - self.pose_received
        if age > self.pose_timeout:
            self._abort(f"localization stale for {age:.2f} s")
            return nominal_steer, 0.0, 0.0, 0.0
        x, y, yaw, measured_speed, lateral_speed, yaw_rate = self.pose
        current_clearance = self.map_field.clearance(x, y)
        if current_clearance < self.hard_clearance:
            self._abort(
                f"car centre clearance {current_clearance:.2f} m below hard "
                f"limit {self.hard_clearance:.2f} m")
            return nominal_steer, 0.0, 0.0, current_clearance

        tracking_path = (self.figure_eight_path
                         if self.figure_eight_path is not None
                         else self.reference_path)
        cross_track, heading_error, _ = containment.tracking_errors(
            tracking_path, x, y, yaw, min(self.i, len(tracking_path) - 1), self.hz,
            phase_independent=self.phase_independent_reference)
        tracking_speed = (measured_speed if abs(measured_speed) > 0.10
                          else nominal_speed)
        raw = containment.steering_correction(
            cross_track, heading_error, tracking_speed,
            self.heading_gain, self.cross_track_gain, self.speed_softening)
        raw = float(np.clip(raw, -self.correction_max, self.correction_max))
        alpha = self.dt / (self.correction_tau + self.dt)
        target = self.feedback + alpha * (raw - self.feedback)
        max_step = self.correction_rate * self.dt
        self.feedback += float(np.clip(target - self.feedback, -max_step, max_step))
        steer = float(np.clip(nominal_steer + self.feedback,
                              -takes_lib.S_MAX, takes_lib.S_MAX))

        predicted_speed = (measured_speed if abs(measured_speed) > abs(nominal_speed)
                           else nominal_speed)
        commanded_clearance = containment.predicted_clearance(
            self.map_field, x, y, yaw, predicted_speed, steer,
            self.prediction_horizon, self.wheelbase)
        measured_clearance = containment.predicted_twist_clearance(
            self.map_field, x, y, yaw, measured_speed, lateral_speed,
            yaw_rate, self.prediction_horizon)
        clearance = min(commanded_clearance, measured_clearance)
        scale = float(np.clip(
            (clearance - self.hard_clearance)
            / (self.slow_clearance - self.hard_clearance), 0.0, 1.0))
        speed = nominal_speed * scale
        if scale < 0.999 and (self.last_safety_scale >= 0.999
                              or abs(scale - self.last_safety_scale) > 0.15):
            self.get_logger().warn(
                f"boundary speed scale {scale:.2f}; predicted clearance "
                f"{clearance:.2f} m", throttle_duration_sec=0.5)
        self.last_safety_scale = scale
        return steer, speed, steer - nominal_steer, clearance

    def _next_nominal_command(self) -> tuple[float, float] | None:
        """Return the next table command, or the pose-driven M3 command."""
        if self.figure_eight is None:
            if self.i >= len(self.rows):
                return None
            steer, speed = self.rows[self.i]
            return float(steer), float(speed)
        if self.unwrapped_yaw is None:
            self._abort("no yaw available for figure-eight lobe tracking")
            return None
        previous_lobe = self.figure_eight.lobe_index
        previous_start = self.figure_eight.lobe_start_yaw
        command = self.figure_eight.next_command(self.unwrapped_yaw)
        if self.figure_eight.timed_out:
            self._abort(
                f"figure-eight lobe {previous_lobe + 1} did not complete before timeout")
            return None
        started_lobe = (command is not None
                        and self.figure_eight.lobe_start_yaw != previous_start)
        if started_lobe:
            self._start_figure_eight_lobe(*command)
            if self.done:
                return None
        if self.figure_eight.lobe_index != previous_lobe:
            message = f"figure-eight lobe {previous_lobe + 1} complete"
            if not self.figure_eight.complete:
                message += f"; starting lobe {self.figure_eight.lobe_index + 1}"
            self.get_logger().info(message)
        return command

    def tick(self) -> None:
        if self.aborted:
            return
        nominal = self._next_nominal_command()
        if self.done:
            return
        if nominal is not None:
            nominal_steer, nominal_speed = nominal
            steer, speed, correction, clearance = (
                self._contained_command(nominal_steer, nominal_speed)
                if self.containment_enabled
                else (nominal_steer, nominal_speed, 0.0, math.inf))
            if self.done:
                return
            self._publish_ackermann(
                self.nominal_pub, nominal_steer, nominal_speed)
            self._publish_ackermann(
                self.correction_pub, correction, speed - nominal_speed)
            self.send(steer, speed)
            self.i += 1
            if self.i % max(1, int(self.hz)) == 0:
                clearance_text = (f"  clear={clearance:.2f}m"
                                  if math.isfinite(clearance) else "")
                self.get_logger().info(
                    f"t={self.i / self.hz:5.1f}s  nominal=({nominal_steer:+.3f},"
                    f" {nominal_speed:.2f})  correction={correction:+.3f}  "
                    f"requested=({steer:+.3f}, {speed:.2f})"
                    f"{clearance_text}", throttle_duration_sec=0.9)
            return
        if self.tail < self.n_tail:
            self.send(float(np.clip(self.rows[-1][0] + self.feedback,
                                    -takes_lib.S_MAX, takes_lib.S_MAX)), 0.0)
            self.tail += 1
            return
        self._complete()

    def report_rate(self) -> None:
        if self.t_start is None or self.t_end is None or self.i == 0:
            return
        elapsed = self.t_end - self.t_start
        expected = (self.i + self.tail) / self.hz
        achieved = (self.i + self.tail) / elapsed if elapsed > 0 else 0.0
        self.get_logger().info(
            f"achieved {achieved:.1f} Hz over {elapsed:.1f} s "
            f"(nominal {self.hz:.0f} Hz / {expected:.1f} s)")
        if abs(achieved - self.hz) > 0.05 * self.hz:
            self.get_logger().warn(
                "rate drifted more than 5% -- extraction assumes a uniform grid, "
                "pass the achieved rate to --control-hz or redo the take")


def main() -> None:
    rclpy.init()
    node = None
    code = EXIT_OK
    try:
        node = TakePlayer()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        code = EXIT_ABORT if node.aborted else EXIT_OK
    except SystemExit as exc:
        code = int(exc.code) if exc.code is not None else EXIT_OK
    except KeyboardInterrupt:
        code = EXIT_ABORT
    finally:
        if node is not None:
            for _ in range(5):
                node.send(0.0, 0.0)
                time.sleep(0.02)
            node.report_rate()
            node.destroy_node()
        rclpy.try_shutdown()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
