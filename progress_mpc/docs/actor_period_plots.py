"""Plots of the actor_period / reseed_from_previous experiment (laptop pinned to 2 cores, 2026-11-15), one PNG per plot.

    uv run --project ../f1tenth_parallel_sim python progress_mpc/docs/actor_period_plots.py

Reads actor_period_results.json and writes next to it:
    plot_actor_timing_2core.png      AC4MPC step time p50 / p99 / max per variant
    plot_actor_branches_2core.png    mean time of the three branches per variant
    plot_actor_paths.png             path driven in 300 steps from each of the 8 start points, per variant
    plot_actor_command_diff.png      command difference to the default with the same measured states (shadow run)
    plot_actor_same_winner.png       share of steps on which the variant picks the same candidate as the default
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
R = json.loads((HERE / "actor_period_results.json").read_text())
INK, MUTED, GRID = "#111111", "#555555", "#e5e5e5"
RAMP = ["#9ec5f4", "#3987e5", "#184f95"]
COLOR = {"default": "#e0662f", "reseed": "#2a78d6", "actor_period 2": "#1baf7a", "actor_period 3": "#eda100", "actor_period 5": "#d55181",
         "default (nudged 1e-7)": "#8a8f98"}
NAMES = {"default": "default", "reseed": "reseed", "actor_period 2": "actor\nperiod 2", "actor_period 3": "actor\nperiod 3",
         "actor_period 5": "actor\nperiod 5"}
VARIANTS = ["default", "reseed", "actor_period 2", "actor_period 3", "actor_period 5"]
plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})


def new(title, size=(7.6, 5.0)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_title(title, loc="left", fontweight="bold", color=INK)
    return fig, ax


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(HERE / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


x = np.arange(len(VARIANTS))
labels = [NAMES[v] for v in VARIANTS]

# ---- step time
fig, ax = new("AC4MPC step time, pinned to 2 cores")
bw = 0.26
for j, m in enumerate(("p50", "p99", "max")):
    vals = [R["timing"][v][m] for v in VARIANTS]
    ax.bar(x + (j - 1) * bw, vals, bw * 0.92, color=RAMP[j], label=m)
    for xi, vi in zip(x + (j - 1) * bw, vals):
        ax.text(xi, vi + 0.15, f"{vi:.1f}", ha="center", fontsize=8.3, color=INK)
ax.set_xticks(x, labels, color=INK, fontsize=9.5)
ax.set_ylim(0, 12.5)
ax.set_ylabel("step time (ms)")
ax.legend(frameon=False, ncol=3, loc="upper right")
grid(ax)
save(fig, "plot_actor_timing_2core.png")

# ---- branches
fig, ax = new("Mean time of each branch, pinned to 2 cores")
names = ("actor", "active", "parallel")
cols = {"actor": "#1baf7a", "active": "#3987e5", "parallel": "#184f95"}
for j, b in enumerate(names):
    vals = [R["branches"][v][b] for v in VARIANTS]
    ax.bar(x + (j - 1) * bw, vals, bw * 0.92, color=cols[b], label=f"{b} branch")
    for xi, vi in zip(x + (j - 1) * bw, vals):
        ax.text(xi, vi + 0.06, f"{vi:.1f}", ha="center", fontsize=8.3, color=INK)
ax.set_xticks(x, labels, color=INK, fontsize=9.5)
ax.set_ylim(0, 4.6)
ax.set_ylabel("branch time per step (ms)")
ax.legend(frameon=False, ncol=3, loc="upper right")
grid(ax)
save(fig, "plot_actor_branches_2core.png")

# ---- paths per start
fig, ax = new("Path driven in 300 steps, per start point", size=(9.0, 5.0))
starts = R["starts"]
for v in VARIANTS + ["default (nudged 1e-7)"]:
    ls, lw, mk = ("--", 1.6, "x") if "nudged" in v else ("-", 1.8, "o")
    ax.plot(starts, R["paths"][v], ls=ls, lw=lw, marker=mk, ms=4.5, color=COLOR[v], label=v.replace("_", " "))
ax.set_xticks(starts)
ax.set_xlabel("start point (centerline index)")
ax.set_ylabel("path (m)")
ax.legend(frameon=False, ncol=3, fontsize=8.8, loc="lower center")
ax.set_ylim(19, 25.2)
grid(ax)
save(fig, "plot_actor_paths.png")

# ---- command difference
fig, ax = new("Command difference to the default (same measured states)")
x2 = np.arange(len(VARIANTS) - 1)
vs = VARIANTS[1:]
bw2 = 0.36
ax.bar(x2 - bw2 / 2, [R["shadow"][v]["dsteer_mean"] for v in vs], bw2 * 0.92, color="#3987e5", label="steering (rad)")
ax.bar(x2 + bw2 / 2, [R["shadow"][v]["dspeed_mean"] for v in vs], bw2 * 0.92, color="#1baf7a", label="speed (m/s)")
for xi, v in zip(x2, vs):
    ax.text(xi - bw2 / 2, R["shadow"][v]["dsteer_mean"] + 0.0008, f"{R['shadow'][v]['dsteer_mean']:.3f}", ha="center", fontsize=8.5, color=INK)
    ax.text(xi + bw2 / 2, R["shadow"][v]["dspeed_mean"] + 0.0008, f"{R['shadow'][v]['dspeed_mean']:.3f}", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(x2, [NAMES[v] for v in vs], color=INK, fontsize=9.5)
ax.set_ylabel("mean absolute difference")
ax.legend(frameon=False, loc="upper left")
grid(ax)
save(fig, "plot_actor_command_diff.png")

# ---- same winner
fig, ax = new("Same candidate chosen as the default")
vals = [R["shadow"][v]["same_winner_pct"] for v in vs]
ax.bar(x2, vals, 0.6, color=[COLOR[v] for v in vs])
for xi, vi in zip(x2, vals):
    ax.text(xi, vi + 0.8, f"{vi:.0f}%", ha="center", color=INK)
ax.set_xticks(x2, [NAMES[v] for v in vs], color=INK, fontsize=9.5)
ax.set_ylim(0, 100)
ax.set_ylabel("share of steps (%)")
grid(ax)
save(fig, "plot_actor_same_winner.png")
print("wrote 5 plots")
