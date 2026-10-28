#!/usr/bin/env python3
"""Slide-ready plots of what the MPC did over many laps, reconstructed from a rosbag.

    uv run --project ../f1tenth_parallel_sim python progress_mpc/scripts/plot_run.py mpc_run4 \
        --map stack_master/maps/hangar_25_10_v --theme both

Needs `/state_estimation/odom_map`, `/drive` and `/controller/diagnostics` in the bag and the track's
`global_waypoints.json`. Writes PNG (200 dpi, 16:9) and SVG per figure, a lap table (`laps.csv`) and the numbers behind
the titles (`stats.json`) into `--out` (default `mpc_plots/<bag name>/`), for the light and/or dark slide theme:

    00_summary            the four main panels on one slide
    01_laps_on_track      every lap's driven path on the track, the track limits and the MPC's corridor
    02_lateral_vs_distance  lateral offset along the track for every lap, the mean path and the lap-to-lap spread
    03_speed_vs_distance  speed along the track for every lap
    04_lap_times          lap times of the complete laps
    05_commands           steering and speed commands over time

A lap is a pass from the start/finish line (s = 0 of the planner's centerline) to the next; only complete laps enter the
statistics. The first and last, partial, laps are not drawn except the last one on the map when it ends at a wall.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import warnings

import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_data as rd  # noqa: E402

# The validated reference palette (dataviz skill): blue + orange pass every check on both surfaces.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#8a8984", grid="#e7e6e2", track="#bdbcb6",
                  corridor="#d6d5cf", lap="#2a78d6", mean="#184f95", accent="#eb6834", cmd="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#8c8b83", grid="#2c2c2a", track="#5a5955",
                 corridor="#3d3c39", lap="#3987e5", mean="#cde2fb", accent="#d95926", cmd="#d95926"),
    # matplotlib's own look: white, black, boxed axes, the default blue and orange
    "plain": dict(surface="white", ink="black", ink2="black", muted="gray", grid="#dddddd", track="dimgray",
                  corridor="gray", lap="tab:blue", mean="black", accent="tab:red", cmd="tab:orange"),
}
SLIDE = (13.333, 7.5)
PLAIN = (10.0, 6.0)
SCALE = 1.0  # text sizes below are for the 13.3 inch slide; the plain theme scales them down


def fs(size: float) -> float:
    return size * SCALE
CLEARANCE = 0.384  # the MPC keeps the centre of gravity this far inside the track limits (collision radius + margin)


def setup_matplotlib(theme: dict, plain: bool = False):
    import matplotlib as mpl

    if plain:
        mpl.rcdefaults()
        mpl.rcParams.update({"svg.fonttype": "none", "figure.dpi": 100, "savefig.dpi": 150})
        return
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
        "figure.facecolor": theme["surface"], "axes.facecolor": theme["surface"], "savefig.facecolor": theme["surface"],
        "axes.edgecolor": theme["muted"], "axes.labelcolor": theme["ink2"], "text.color": theme["ink"],
        "xtick.color": theme["ink2"], "ytick.color": theme["ink2"], "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": theme["grid"], "grid.linewidth": 1.0,
        "axes.axisbelow": True, "lines.solid_capstyle": "round", "svg.fonttype": "none",
    })


class Context:
    """Everything the figures draw from."""

    def __init__(self, bag_path: Path, map_dir: Path, direction: str, clearance: float):
        self.bag_name = bag_path.name
        self.track = rd.Track.load(map_dir, direction)
        self.bag = rd.read_bag(bag_path)
        self.clearance = clearance
        self.laps_all, self.info = rd.laps_from_bag(self.bag, self.track)
        self.laps = [lap for lap in self.laps_all if lap.complete]
        if len(self.laps) < 2:
            raise SystemExit(f"{len(self.laps)} complete lap(s) in {bag_path}: need at least two to compare laps")
        self.grid = np.arange(0.0, self.track.length, 0.1)
        self.lateral = np.array([rd.on_grid(lap, self.grid, lap.lateral) for lap in self.laps])
        self.speed = np.array([rd.on_grid(lap, self.grid, lap.speed) for lap in self.laps])
        self.mean_lateral = np.nanmean(self.lateral, axis=0)
        self.spread = np.nanstd(self.lateral, axis=0, ddof=1)  # lap-to-lap 1-sigma at each point
        self.mean_speed = np.nanmean(self.speed, axis=0)
        w_left = np.interp(self.grid, self.track.s, self.track.w_left, period=self.track.length)
        w_right = np.interp(self.grid, self.track.s, self.track.w_right, period=self.track.length)
        self.limit_left, self.limit_right = w_left, -w_right
        self.corridor_left, self.corridor_right = w_left - clearance, -(w_right - clearance)
        self.final = self.laps_all[-1] if not self.laps_all[-1].complete and self.laps_all[-1].number > self.laps_all[0].number else None
        times = np.array([lap.duration for lap in self.laps])
        deviation = np.sqrt(np.nanmean((self.lateral - self.mean_lateral) ** 2, axis=1))
        outside = np.mean((self.lateral > self.corridor_left) | (self.lateral < self.corridor_right))
        self.stats = {
            "bag": self.bag_name, "track_length_m": round(self.track.length, 2), "complete_laps": len(self.laps),
            "lap_times_s": [round(float(t), 2) for t in times], "lap_time_mean_s": round(float(times.mean()), 2),
            "lap_time_std_s": round(float(times.std(ddof=1)), 2), "lap_time_best_s": round(float(times.min()), 2),
            "lap_time_spread_pct": round(float(100 * times.std(ddof=1) / times.mean()), 1),
            "lateral_spread_median_cm": round(float(100 * np.nanmedian(self.spread)), 1),
            "lateral_spread_p90_cm": round(float(100 * np.nanpercentile(self.spread, 90)), 1),
            "lateral_spread_max_cm": round(float(100 * np.nanmax(self.spread)), 1),
            "per_lap_rms_from_mean_path_cm": [round(float(100 * d), 1) for d in deviation],
            "mean_speed_mps": round(float(np.mean([lap.speed.mean() for lap in self.laps])), 2),
            "max_speed_mps": round(float(max(lap.speed.max() for lap in self.laps)), 2),
            "share_of_samples_outside_mpc_corridor_pct": round(float(100 * outside), 1),
        }

    @property
    def subtitle(self) -> str:
        return (f"{self.bag_name} · {len(self.laps)} complete laps of the {self.track.length:.1f} m track · position from "
                f"the localization-corrected estimator")


def title(fig, text: str, sub: str, theme: dict) -> None:
    fig.text(0.045, 0.955, text, fontsize=fs(24), fontweight="normal" if SCALE < 1 else "bold", color=theme["ink"], va="top", ha="left")
    fig.text(0.045, 0.895, sub, fontsize=fs(14), color=theme["ink2"], va="top", ha="left")


def lap_label(ax, x, y, text, theme, **kw) -> None:
    ax.text(x, y, text, color=theme["ink2"], fontsize=fs(14), va=kw.pop("va", "bottom"), ha=kw.pop("ha", "left"), **kw)


# ---------------------------------------------------------------------------------------------------------- panels

def draw_map(ax, ctx: Context, theme: dict, compact: bool = False) -> None:
    t = ctx.track
    left = t.xy + t.normal * t.w_left[:, None]
    right = t.xy - t.normal * t.w_right[:, None]
    inner_l = t.xy + t.normal * (t.w_left - ctx.clearance)[:, None]
    inner_r = t.xy - t.normal * (t.w_right - ctx.clearance)[:, None]
    for edge in (left, right):
        ax.plot(*np.vstack([edge, edge[:1]]).T, color=theme["track"], lw=2.0, zorder=1)
    for edge in (inner_l, inner_r):
        ax.plot(*np.vstack([edge, edge[:1]]).T, color=theme["corridor"], lw=1.4, ls=(0, (4, 3)), zorder=1)
    for i, lap in enumerate(ctx.laps):
        ax.plot(*lap.xy.T, color=theme["lap"], lw=1.2, alpha=0.55, zorder=3, label="each lap" if i == 0 else None)
    mean_xy = np.stack([np.interp(ctx.grid, t.s, t.xy[:, k], period=t.length) for k in range(2)], axis=1)
    nrm = np.stack([np.interp(ctx.grid, t.s, t.normal[:, k], period=t.length) for k in range(2)], axis=1)
    path = mean_xy + nrm * ctx.mean_lateral[:, None]
    ax.plot(*path.T, color=theme["mean"], lw=2.4, zorder=4, label="mean path")
    if ctx.final is not None:
        ax.plot(*ctx.final.xy.T, color=theme["accent"], lw=1.6, zorder=5, label="last lap (ends at the wall)")
        ax.plot(*ctx.final.xy[-1], marker="o", ms=10, color=theme["accent"], mec=theme["surface"], mew=2, zorder=6)
    # start / finish line
    p0, n0 = t.xy[0], t.normal[0]
    a, b = p0 + n0 * t.w_left[0], p0 - n0 * t.w_right[0]
    ax.plot([a[0], b[0]], [a[1], b[1]], color=theme["ink"], lw=2.0, zorder=2)
    ax.annotate("start / finish", xy=b, xytext=(10, -22), textcoords="offset points", color=theme["ink2"], fontsize=fs(14))
    ax.plot([], [], color=theme["track"], lw=2.0, label="track limits")
    ax.plot([], [], color=theme["corridor"], lw=1.4, ls=(0, (4, 3)), label="MPC corridor")
    ax.set_aspect("equal"); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    if compact:
        if ctx.final is not None:  # the lateral panel's legend names the rest
            ax.annotate("last lap ends\nat the wall", xy=ctx.final.xy[-1], xytext=(-8, 8), textcoords="offset points",
                        color=theme["accent"], fontsize=fs(12), ha="right", va="bottom")
    else:
        ax.legend(loc="center left", bbox_to_anchor=(1.03, 0.5), frameon=False, labelcolor=theme["ink2"], handlelength=2.2, fontsize=fs(13))


def draw_lateral(ax, ctx: Context, theme: dict, label_laps: bool = True, compact: bool = False) -> None:
    g = ctx.grid
    ax.fill_between(g, ctx.corridor_right, ctx.corridor_left, color=theme["grid"], alpha=0.7, lw=0, zorder=0)
    ax.plot(g, ctx.limit_left, color=theme["track"], lw=2.0, zorder=1); ax.plot(g, ctx.limit_right, color=theme["track"], lw=2.0, zorder=1)
    for i, lat in enumerate(ctx.lateral):
        ax.plot(g, lat, color=theme["lap"], lw=1.2, alpha=0.55, zorder=3, label="each lap" if i == 0 else None)
    ax.plot(g, ctx.mean_lateral, color=theme["mean"], lw=2.4, zorder=4, label="mean of all laps")
    ax.plot([], [], color=theme["track"], lw=2.0, label="track limits")
    ax.fill_between([], [], [], color=theme["grid"], alpha=0.7, label="MPC corridor")
    ax.axhline(0, color=theme["muted"], lw=0.8, zorder=2)
    ax.set_xlim(0, ctx.track.length); ax.set_ylabel("lateral offset (m, left +)")
    lim = max(np.nanmax(np.abs(ctx.limit_left)), np.nanmax(np.abs(ctx.limit_right))) * 1.05
    ax.set_ylim(-lim, lim)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=5, frameon=False, labelcolor=theme["ink2"], fontsize=fs(11 if compact else 13), handlelength=1.4 if compact else 1.8, columnspacing=0.9 if compact else 1.4)


def draw_spread(ax, ctx: Context, theme: dict) -> None:
    ax.fill_between(ctx.grid, 0, 100 * ctx.spread, color=theme["lap"], alpha=0.25, lw=0)
    ax.plot(ctx.grid, 100 * ctx.spread, color=theme["lap"], lw=2.0)
    med = ctx.stats["lateral_spread_median_cm"]
    ax.axhline(med, color=theme["ink2"], lw=1.2, ls=(0, (5, 3)))
    ax.text(0.30, 0.93, f"dashed: median {med:.1f} cm", transform=ax.transAxes, color=theme["ink2"], fontsize=fs(14), va="top", ha="left")
    ax.set_xlim(0, ctx.track.length); ax.set_ylim(0, None); ax.set_ylabel("spread, 1σ (cm)")


def draw_speed(ax, ctx: Context, theme: dict) -> None:
    for i, v in enumerate(ctx.speed):
        ax.plot(ctx.grid, v, color=theme["lap"], lw=1.2, alpha=0.55, label="each lap" if i == 0 else None)
    ax.plot(ctx.grid, ctx.mean_speed, color=theme["mean"], lw=2.4, label="mean of all laps")
    ax.set_xlim(0, ctx.track.length); ax.set_ylim(0, None); ax.set_ylabel("speed (m/s)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, frameon=False, labelcolor=theme["ink2"], fontsize=fs(13))


def draw_lap_times(ax, ctx: Context, theme: dict) -> None:
    times = [lap.duration for lap in ctx.laps]
    xs = np.arange(1, len(times) + 1)
    bars = ax.bar(xs, times, width=0.62, color=theme["lap"], zorder=3)
    for b in bars:  # 4 px rounded data-ends would need patches; plain bars with a surface gap keep the spec simple
        b.set_edgecolor(theme["surface"]); b.set_linewidth(2)
    for x, tm in zip(xs, times):
        ax.text(x, tm + 0.12, f"{tm:.1f} s", ha="center", va="bottom", fontsize=fs(16), color=theme["ink"], fontweight="bold", zorder=6,
                bbox=dict(facecolor=theme["surface"], edgecolor="none", pad=1.5))
    mean = float(np.mean(times))
    ax.axhline(mean, color=theme["ink2"], lw=1.4, ls=(0, (5, 3)), zorder=4)
    ax.text(len(times) + 0.62, mean, f"mean\n{mean:.1f} s", color=theme["ink2"], fontsize=fs(14), va="center", ha="left",
            bbox=dict(facecolor=theme["surface"], edgecolor="none", pad=2))
    ax.set_xticks(xs); ax.set_xticklabels([f"lap {k}" for k in xs]); ax.set_ylim(0, max(times) * 1.18)
    ax.set_xlim(0.4, len(times) + 1.4); ax.set_ylabel("lap time (s)"); ax.grid(axis="x", visible=False)


def draw_commands(axes, ctx: Context, theme: dict) -> None:
    from matplotlib.lines import Line2D

    drive, odom = ctx.bag["drive"], ctx.bag["odom"]
    for ax in axes:
        for lap in ctx.laps:
            ax.axvline(lap.t0, color=theme["muted"], lw=0.8, ls=(0, (2, 3)), zorder=1)
    top, bottom = axes
    top.plot(drive[:, 0], drive[:, 2], color=theme["lap"], lw=1.4)
    for level in (-0.3, 0.3):
        top.axhline(level, color=theme["track"], lw=1.2)
    top.text(drive[-1, 0], 0.3, "steering limit ±0.3 rad ", color=theme["ink2"], fontsize=fs(14), ha="right", va="bottom")
    top.set_ylabel("steering (rad)"); top.set_ylim(-0.38, 0.38)
    bottom.plot(odom[:, 0], odom[:, 4], color=theme["lap"], lw=1.6)
    bottom.plot(drive[:, 0], drive[:, 1], color=theme["cmd"], lw=1.4, alpha=0.9)
    bottom.set_ylabel("speed (m/s)"); bottom.set_xlabel("time (s)"); bottom.set_ylim(0, None)
    bottom.legend(handles=[Line2D([], [], color=theme["lap"], lw=2.4, label="measured speed"),
                           Line2D([], [], color=theme["cmd"], lw=2.4, label="speed command")],
                  loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, frameon=False, labelcolor=theme["ink2"], fontsize=fs(13))
    top.set_xlim(0, odom[-1, 0]); bottom.set_xlim(0, odom[-1, 0])


# ---------------------------------------------------------------------------------------------------------- figures

def figure(name: str, ctx: Context, theme_name: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    global SCALE
    theme = THEMES[theme_name]
    plain = theme_name == "plain"
    SCALE = 0.6 if plain else 1.0
    setup_matplotlib(theme, plain)
    s = ctx.stats
    fig = plt.figure(figsize=PLAIN if plain else SLIDE)
    if name == "00_summary":
        gs = fig.add_gridspec(2, 2, left=0.07, right=0.985, top=0.78, bottom=0.085, wspace=0.17, hspace=0.55)
        draw_map(fig.add_subplot(gs[0, 0]), ctx, theme, compact=True)
        ax_lat = fig.add_subplot(gs[0, 1]); draw_lateral(ax_lat, ctx, theme, compact=True); ax_lat.set_xlabel("distance along the track (m)"); ax_lat.set_ylabel("lateral offset (m)", fontsize=fs(13))
        draw_lap_times(fig.add_subplot(gs[1, 0]), ctx, theme)
        ax_sp = fig.add_subplot(gs[1, 1]); draw_speed(ax_sp, ctx, theme); ax_sp.set_xlabel("distance along the track (m)")
        title(fig, f"{s['complete_laps']} laps of {s['lap_time_mean_s']:.1f} s (σ {s['lap_time_std_s']:.1f} s), "
                   f"the line repeats to {s['lateral_spread_median_cm']:.0f} cm (1σ)", ctx.subtitle, theme)
    elif name == "01_laps_on_track":
        ax = fig.add_axes([0.04, 0.03, 0.62, 0.82]); draw_map(ax, ctx, theme)
        title(fig, f"The same line, lap after lap: {s['complete_laps']} laps overlaid on the track", ctx.subtitle, theme)
    elif name == "02_lateral_vs_distance":
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.2], left=0.075, right=0.985, top=0.79, bottom=0.095, hspace=0.12)
        ax1 = fig.add_subplot(gs[0]); draw_lateral(ax1, ctx, theme); ax1.tick_params(labelbottom=False)
        ax2 = fig.add_subplot(gs[1], sharex=ax1); draw_spread(ax2, ctx, theme); ax2.set_xlabel("distance along the track (m)")
        title(fig, f"Path stability: laps differ by {s['lateral_spread_median_cm']:.0f} cm (1σ, median); "
                   f"at most {s['lateral_spread_max_cm']:.0f} cm", ctx.subtitle, theme)
    elif name == "03_speed_vs_distance":
        ax = fig.add_axes([0.075, 0.105, 0.91, 0.66]); draw_speed(ax, ctx, theme); ax.set_xlabel("distance along the track (m)")
        title(fig, f"Speed profile repeats: mean {s['mean_speed_mps']:.1f} m/s, peak {s['max_speed_mps']:.1f} m/s", ctx.subtitle, theme)
    elif name == "04_lap_times":
        ax = fig.add_axes([0.075, 0.105, 0.91, 0.70]); draw_lap_times(ax, ctx, theme)
        title(fig, f"Lap times: mean {s['lap_time_mean_s']:.1f} s, spread {s['lap_time_std_s']:.1f} s "
                   f"({s['lap_time_spread_pct']:.1f} %), best {s['lap_time_best_s']:.1f} s", ctx.subtitle, theme)
    elif name == "05_commands":
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], left=0.075, right=0.985, top=0.80, bottom=0.105, hspace=0.30)
        a1 = fig.add_subplot(gs[0]); a2 = fig.add_subplot(gs[1], sharex=a1); draw_commands((a1, a2), ctx, theme)
        title(fig, "What the controller commanded over the whole drive", ctx.subtitle + " · dotted lines mark laps", theme)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}_{theme_name}.{ext}", dpi=150 if plain else 200)
    plt.close(fig)


FIGURES = ("00_summary", "01_laps_on_track", "02_lateral_vs_distance", "03_speed_vs_distance", "04_lap_times", "05_commands")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--map", type=Path, required=True, help="map folder with global_waypoints.json (stack_master/maps/<name>)")
    parser.add_argument("--direction", choices=("ccw", "cw"), default="ccw")
    parser.add_argument("--clearance", type=float, default=CLEARANCE, help="MPC clearance inside the track limits (m)")
    parser.add_argument("--theme", choices=("light", "dark", "both", "plain"), default="light")
    parser.add_argument("--out", type=Path, help="default: mpc_plots/<bag name>")
    args = parser.parse_args()

    out = args.out or Path("mpc_plots") / args.bag.name
    out.mkdir(parents=True, exist_ok=True)
    ctx = Context(args.bag, args.map, args.direction, args.clearance)
    for theme_name in (("light", "dark") if args.theme == "both" else (args.theme,)):
        for name in FIGURES:
            figure(name, ctx, theme_name, out)
    with open(out / "laps.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lap", "start_s", "end_s", "lap_time_s", "mean_speed_mps", "lateral_rms_cm", "complete"])
        for lap in ctx.laps_all:
            writer.writerow([lap.number, f"{lap.t0:.2f}", f"{lap.t1:.2f}", f"{lap.duration:.2f}", f"{lap.speed.mean():.2f}",
                             f"{100 * np.sqrt(np.mean(lap.lateral ** 2)):.1f}", int(lap.complete)])
    (out / "stats.json").write_text(json.dumps(ctx.stats, indent=2) + "\n")
    print(json.dumps(ctx.stats, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
