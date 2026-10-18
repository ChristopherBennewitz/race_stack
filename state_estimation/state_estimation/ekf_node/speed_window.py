"""Forward speed from a stream of poses, over consecutive non-overlapping windows."""
import math


class TrailingSpeedWindow:
    """Turns poses into forward-speed samples, one per completed window.

    The speed is the displacement along the mean heading over the window divided by its duration, so it
    is the mean speed over the window and lags the present by half a window. Windows do not overlap, so
    consecutive samples share no pose and their noise is independent. A pose whose stamp does not advance
    (a republished latest transform) is ignored. A window stretched by a gap beyond ``max_stretch`` windows
    is dropped and restarted, since its mean no longer describes a short span.

    Args:
        window:      target window length, seconds
        max_stretch: longest accepted window, in multiples of ``window``
    """

    def __init__(self, window, max_stretch=2.5):
        self.window = float(window)
        self.max_stretch = float(max_stretch)
        self._anchor = None
        self._last_t = None

    def push(self, t, x, y, yaw):
        """Add a pose. Returns ``(speed, duration)`` when it completes a window, else None."""
        if self._last_t is not None and t <= self._last_t:
            return None
        self._last_t = t
        if self._anchor is None:
            self._anchor = (t, x, y, yaw)
            return None
        t0, x0, y0, yaw0 = self._anchor
        duration = t - t0
        if duration < self.window:
            return None
        self._anchor = (t, x, y, yaw)
        if duration > self.max_stretch * self.window:
            return None
        heading = math.atan2(math.sin(yaw0) + math.sin(yaw), math.cos(yaw0) + math.cos(yaw))
        speed = ((x - x0) * math.cos(heading) + (y - y0) * math.sin(heading)) / duration
        return speed, duration
