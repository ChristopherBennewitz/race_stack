import torch

from controller.rl.observation import Centerline, HistoryBuffer
from controller.rl.policy import load_policy


class RL_Controller:
    """Runs a trained PPO/RMA policy (no ROS). Driven by the controller manager at the policy's control rate.

    Inputs: the track centerline (set once) and the car state from the EKF (pose, speed, yaw rate).
    Output: speed + steering angle, which the actuation manager rate-limits exactly as the sim's
    actuation limiter does.
    """

    def __init__(self, checkpoint, device, direction, logger):
        self.policy = load_policy(checkpoint, device)
        self.shape = self.policy.shape
        self.history = HistoryBuffer(self.shape.history_len)
        self.direction = direction
        self.centerline = None
        self.prev_cmd = (0.0, 0.0)
        self.first_step = True

        logger.info(
            f"Loaded {checkpoint}: {self.shape.n_waypoints} waypoints, history {self.shape.history_len}, "
            f"{self.shape.control_hz} Hz"
        )
        low = (self.policy.action_bias - self.policy.action_scale).tolist()
        high = (self.policy.action_bias + self.policy.action_scale).tolist()
        logger.info(
            f"Action bounds [vel_cmd, steer_cmd]: low {low}, high {high}. These must agree with the actuation "
            "manager's limits; the policy's previous-action input is normalized with them."
        )

    @property
    def control_hz(self):
        return self.shape.control_hz

    def set_centerline(self, xy, d_left, d_right):
        self.centerline = Centerline(xy, d_left, d_right, direction=self.direction)

    def main_loop(self, x, y, yaw, vel, yaw_rate):
        waypoints = torch.from_numpy(self.centerline.lookahead(x, y, yaw, self.shape.n_waypoints))
        # The newest history row pairs the current measurement with the previously issued command. On the
        # first step the sim's history is still all zeros (rows are only pushed after an action), so mirror that.
        if self.first_step:
            history = self.history.buf
            self.first_step = False
        else:
            history = self.history.push(vel, yaw_rate, *self.prev_cmd)
        vel_cmd, steer_cmd = self.policy(vel, yaw_rate, waypoints, history).tolist()
        self.prev_cmd = (vel_cmd, steer_cmd)
        return vel_cmd, steer_cmd
