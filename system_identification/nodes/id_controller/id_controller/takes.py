#!/usr/bin/env python3
"""Feed-forward excitation takes for the F1TENTH identification campaign.

One take = one recording = one bag. Every take is a fixed table of
``(steer_cmd [rad], vel_cmd [m/s])`` sampled at ``HZ``. The player adds a
separately logged, low-bandwidth containment correction to these commands.

The manifest is split in two because the two sets need different spaces:

``TAKES``     19 bounded room takes -- actual fit is checked against ``/map``.
``CORRIDOR``   5 takes that need a long straight -- everything longitudinal.

Standalone use (no ROS needed, only numpy)::

    python3 -m id_controller.takes --list
    python3 -m id_controller.takes --csv out/

Because ``steer_gain`` is not known, every bounded maneuver runs on a circle (a
held steering command circles at *some* radius regardless of the gain), while
anything that has to travel in a straight line is budgeted by distance instead.
"""
from __future__ import annotations

import numpy as np

HZ = 40
DT = 1.0 / HZ
L = 0.32                      # measured wheelbase, m

# Conservative symmetric limit inside NUC2's calibrated [-0.346, +0.353] rad
# range.  Keeping generated takes inside it avoids asymmetric manager clipping.
S_MAX = 0.34
V_CEILING = 5.0     # guard against a typo, not a physical limit


def _n(sec: float) -> int:
    return max(1, int(round(sec * HZ)))


def hold(sec: float, steer: float, vel: float) -> np.ndarray:
    k = _n(sec)
    return np.stack([np.full(k, steer), np.full(k, vel)], axis=-1)


def ramp(sec: float, steer: float, v0: float, v1: float) -> np.ndarray:
    k = _n(sec)
    return np.stack([np.full(k, steer), np.linspace(v0, v1, k)], axis=-1)


def cat(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate([p for p in parts if len(p)], axis=0)


def revolution_seconds(delta: float, v: float, gain: float = 1.0) -> float:
    """Time for one full turn at the WORST CASE (largest) radius, i.e. lowest gain."""
    return 2.0 * np.pi * (L / np.tan(gain * delta)) / v


# --- M1: kinematic steer_gain, one take per (angle, direction) ------------------
M1_DELTAS = (0.20, 0.25, 0.30, 0.34)
M1_V = 0.6


def m1(delta: float, sign: float, revs: float = 1.25) -> np.ndarray:
    sec = revs * revolution_seconds(delta, M1_V)
    return cat(hold(1.0, sign * delta, 0.0),
               hold(sec, sign * delta, M1_V),
               hold(0.5, sign * delta, 0.0))


# --- M2: skidpad, ramp to the slide -------------------------------------------
def m2(delta: float = 0.34, sign: float = 1.0, v0: float = 0.8, v1: float = 3.6,
       sec: float = 16.0) -> np.ndarray:
    return cat(hold(1.0, sign * delta, 0.0),
               ramp(sec, sign * delta, v0, v1),
               hold(1.5, sign * delta, v1))


# --- M3: figure-eight, lobe = one revolution at the worst-case radius -----------
def m3(delta: float = 0.34, v: float = 1.4, laps: int = 3) -> np.ndarray:
    lobe = revolution_seconds(delta, v)
    parts = [hold(0.5, 0.0, v)]
    for _ in range(laps):
        parts += [hold(lobe, +delta, v), hold(lobe, -delta, v)]
    return cat(*parts)


# --- M4: steering chirp about a circular bias ----------------------------------
def m4(bias: float = 0.24, amp: float = 0.10, f0: float = 0.2, f1: float = 3.0,
       sec: float = 24.0, v: float = 1.8) -> np.ndarray:
    k = _n(sec)
    t = np.arange(k) * DT
    ph = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * sec))
    return np.stack([bias + amp * np.sin(ph), np.full(k, v)], axis=-1)


# --- M5: speed steps on a circle ----------------------------------------------
def m5(delta: float = 0.30, sign: float = 1.0,
       levels: tuple = (1.0, 2.6, 1.0, 3.2, 0.8), sec: float = 2.5) -> np.ndarray:
    parts = [hold(1.0, sign * delta, levels[0])]
    for v in levels:
        parts.append(hold(sec, sign * delta, v))
    return cat(*parts)


# --- M6: straight-line bias runs, budgeted by distance -------------------------
def m6(v: float = 1.5, sec: float = 2.6) -> np.ndarray:
    return cat(hold(0.5, 0.0, v), hold(sec, 0.0, v), hold(0.6, 0.0, 0.0))


# --- M7: rate-limit doublets, ridden around a circle to stay bounded -----------
def m7(bias: float = 0.17, amp: float = 0.17, half: float = 0.15, v: float = 1.1,
       n: int = 12, gap: float = 0.6) -> np.ndarray:
    parts = [hold(0.8, bias, v)]
    for i in range(n):
        s = 1.0 if i % 2 == 0 else -1.0
        parts += [hold(half, bias + s * amp, v),
                  hold(half, bias - s * amp, v),
                  hold(gap, bias, v)]
    return cat(*parts)


# ---------------------------------------------------------------------------
# CORRIDOR TAKES: a long straight run that a small room cannot provide.
#
# This is where the longitudinal parameters live. In the box the car never passes
# 2.4 m/s because it has to stop again; a 10 m straight reaches 3.0 and a 12 m
# straight 4.7, and `motor_kp` / `brake_kp` are the two worst-determined
# parameters in the fit (+-55% and +-41%) precisely because nothing has ever
# excited them.
#
# Width needed is under 0.7 m for all of these -- a hallway is enough. The stated
# distance includes 0.5 m of car and the full coast to a stop.
# ---------------------------------------------------------------------------

def c1(v: float = 3.0, t_up: float = 3.0, t_hold: float = 1.0,
       t_dn: float = 2.5) -> np.ndarray:
    """One clean accel/brake step. 10.0 m, reaches 3.0 m/s."""
    return cat(hold(0.5, 0.0, 0.0), hold(t_up, 0.0, v),
               hold(t_hold, 0.0, v), hold(t_dn, 0.0, 0.0))


def c2(v: float = 5.0, t_up: float = 2.0, t_hold: float = 1.0,
       t_dn: float = 2.5) -> np.ndarray:
    """A faster step for a 12 m run. 11.7 m, reaches 4.7 m/s."""
    return cat(hold(0.5, 0.0, 0.0), hold(t_up, 0.0, v),
               hold(t_hold, 0.0, v), hold(t_dn, 0.0, 0.0))


def c3(v_lo: float = 1.3, v_hi: float = 3.4, t_lo: float = 3.0,
       t_hi: float = 1.8) -> np.ndarray:
    """A step from a SETTLED low speed, which a staircase cannot afford here.

    The velocity loop is first order with a time constant near 1 s, so a level
    has to be held ~3 s before the response is steady. A four-level staircase at
    1.2 s per level -- the obvious design -- never settles anywhere, and then the
    loop gain and the time constant trade against each other along a flat
    direction. The corridor's real budget is the integral of speed (~11 m), so it
    buys ONE clean settled step, not four muddy ones. Get multiple transients by
    repeating the take, not by subdividing it.
    """
    return cat(hold(0.5, 0.0, 0.0), hold(t_lo, 0.0, v_lo),
               hold(t_hi, 0.0, v_hi), hold(2.5, 0.0, 0.0))


def c4(v: float = 3.0, sec: float = 2.5) -> np.ndarray:
    """Straight bias run with real dwell -- hundreds of steer_bias samples, not tens."""
    return cat(hold(0.5, 0.0, v), hold(sec, 0.0, v), hold(2.0, 0.0, 0.0))


def c5(amp: float = 0.05, f0: float = 1.0, f1: float = 4.0, sec: float = 3.0,
       v: float = 3.0) -> np.ndarray:
    """High-band steering chirp along the straight.

    The low-frequency end of a sweep needs time the corridor does not have, but
    the delay-vs-lag discriminator is the AMPLITUDE ROLL-OFF, which lives at the
    high end. Three seconds at 1-4 Hz is seven cycles of exactly the band that
    separates them, and it complements M4's low band rather than repeating it.
    Amplitude is small so the car stays in the lane.
    """
    k = _n(sec)
    t = np.arange(k) * DT
    ph = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * sec))
    steer = amp * np.sin(ph)
    vel = np.full(k, v)
    return cat(hold(1.5, 0.0, v), np.stack([steer, vel], axis=-1), hold(2.0, 0.0, 0.0))


# --- manifests ----------------------------------------------------------------

TAKES = {}
for _d in M1_DELTAS:
    for _s, _lbl in ((+1.0, "L"), (-1.0, "R")):
        TAKES[f"M1_circle_{_d:.2f}_{_lbl}"] = (lambda d=_d, s=_s: m1(d, s))
TAKES["M2_skidpad_L"] = lambda: m2(sign=+1.0)
TAKES["M2_skidpad_R"] = lambda: m2(sign=-1.0)
TAKES["M3_figure_eight"] = m3
TAKES["M4_chirp_on_circle"] = m4
TAKES["M5_speed_steps_L"] = lambda: m5(sign=+1.0)
TAKES["M5_speed_steps_R"] = lambda: m5(sign=-1.0)
for _v, _sec in ((1.0, 2.6), (1.5, 2.6), (2.2, 2.0), (3.0, 1.3)):
    TAKES[f"M6_straight_{_v:.1f}"] = (lambda v=_v, sec=_sec: m6(v, sec))
TAKES["M7_doublets_on_circle"] = m7

CORRIDOR = {
    "C1_accel_brake_3.0": c1,
    "C2_accel_brake_4.7": c2,
    "C3_settled_step": c3,
    "C4_straight_bias_3.0": c4,
    "C5_chirp_highband": c5,
}

ALL = {**TAKES, **CORRIDOR}


def has_phase_independent_reference(name: str) -> bool:
    """Whether a named take follows closed circular geometry.

    Progress around these paths depends on the steering response being measured,
    so it must not be inferred from elapsed time. The figure-eight is excluded:
    its crossing still needs the time-local association.
    """
    return (name.startswith(("M1_circle_", "M2_skidpad_", "M5_speed_steps_"))
            or name in {"M4_chirp_on_circle", "M7_doublets_on_circle"})


def completes_at_radial_limit(name: str) -> bool:
    """Whether reaching the radial limit successfully completes the take."""
    return name.startswith("M2_skidpad_")


def build(name: str) -> np.ndarray:
    """Return the (N, 2) command table for a named take, validated."""
    if name not in ALL:
        raise KeyError(
            f"unknown take {name!r}; known takes:\n  " + "\n  ".join(sorted(ALL)))
    cmds = np.asarray(ALL[name](), dtype=float)
    validate(name, cmds)
    return cmds


def reference_steering(name: str, commands: np.ndarray) -> np.ndarray:
    """Steering used to form the containment path, excluding fast excitation."""
    if name == "M4_chirp_on_circle":
        return np.full(len(commands), 0.24)
    if name == "M7_doublets_on_circle":
        return np.full(len(commands), 0.17)
    if name == "C5_chirp_highband":
        return np.zeros(len(commands))
    return np.asarray(commands[:, 0], dtype=float).copy()


def validate(name: str, cmds: np.ndarray) -> None:
    """A take that clips against s_max is measuring the hard stop, not what it targets."""
    peak = float(np.abs(cmds[:, 0]).max())
    if peak > S_MAX:
        raise ValueError(f"{name}: steering peak {peak:.3f} exceeds s_max {S_MAX}")
    vmax = float(cmds[:, 1].max())
    if vmax > V_CEILING:
        raise ValueError(f"{name}: speed {vmax:.2f} above ceiling {V_CEILING}")


def load_csv(path: str) -> np.ndarray:
    """Read a take CSV with header ``t,steer_cmd,vel_cmd`` into an (N, 2) table."""
    import csv as _csv
    with open(path) as fh:
        rows = list(_csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} has no rows")
    return np.array([[float(r["steer_cmd"]), float(r["vel_cmd"])] for r in rows])


def write_all(out_dir: str = "takes") -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)
    for name in ALL:
        cmds = build(name)
        t = np.arange(len(cmds)) * DT
        np.savetxt(os.path.join(out_dir, f"{name}.csv"),
                   np.column_stack([t, cmds]),
                   delimiter=",", header="t,steer_cmd,vel_cmd", comments="", fmt="%.6f")
        print(f"{name:<24}{len(cmds) / HZ:>7.1f} s")


def _summary(name: str) -> str:
    cmds = build(name)
    return (f"{name:<24}{len(cmds) / HZ:>7.1f} s  "
            f"steer |max| {np.abs(cmds[:, 0]).max():.3f} rad  "
            f"v max {cmds[:, 1].max():.2f} m/s")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="print the manifest")
    ap.add_argument("--corridor", action="store_true",
                    help="with --list, show the 10-12 m straight set instead")
    ap.add_argument("--csv", metavar="DIR", help="write every take to DIR as CSV")
    args = ap.parse_args()

    if args.csv:
        write_all(args.csv)
        return

    manifest = CORRIDOR if args.corridor else TAKES
    total = 0.0
    for name in manifest:
        print(_summary(name))
        total += len(build(name)) / HZ
    label = "corridor" if args.corridor else "mapped room"
    print(f"\n{len(manifest)} takes, {total:.0f} s of driving ({label})")


if __name__ == "__main__":
    main()
