"""Shared, explicit steering scaling for path-following controllers."""

import numpy as np


def scale_steering(steer, lookup_speed, measured_speed, start_scale_speed,
                   end_scale_speed, downscale_factor, boost_per_mps,
                   boost_max):
    """Apply the configured high-speed downscale and measured-speed boost."""
    speed_range = max(0.1, end_scale_speed - start_scale_speed)
    downscale = 1 - np.clip(
        (lookup_speed - start_scale_speed) / speed_range,
        0.0,
        1.0,
    ) * downscale_factor
    boost = np.clip(
        1 + measured_speed * boost_per_mps,
        1.0,
        boost_max,
    )
    return steer * downscale * boost
