import numpy as np


def pose_in_frame(x, y, yaw, frame_x, frame_y, frame_yaw):
    """Re-expresses a pose given in a child frame in its parent frame.

    `frame_*` is the child frame's pose in the parent (e.g. `odom` in `map`). The state estimate is in `odom`,
    which starts wherever the car was switched on, while the centerline is in `map`.
    """
    c, s = np.cos(frame_yaw), np.sin(frame_yaw)
    return c * x - s * y + frame_x, s * x + c * y + frame_y, yaw + frame_yaw
