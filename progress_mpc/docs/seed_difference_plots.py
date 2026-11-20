"""Plots of how different the commands that re-seed the parallel solver are, default vs reseed_from_previous (laptop, 2026-11-15).

    uv run --project ../f1tenth_parallel_sim python progress_mpc/docs/seed_difference_plots.py

Reads seed_difference.json and writes next to it:
    plot_seed_difference_steer.png   steering: seed difference and the difference of the solved plans, along the horizon
    plot_seed_difference_speed.png   the same for the speed command
Measured with Ac4Mpc(debug_compare_seeds=True) at the re-seed steps; see the "_about" key of the JSON.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
R = json.loads((HERE / "seed_difference.json").read_text())
INK, MUTED, GRID = "#111111", "#555555", "#e5e5e5"
SEED, PLAN = "#2a78d6", "#e0662f"
plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})
k = np.arange(40)
for name, unit, fname in (("steer", "steering command difference (rad)", "plot_seed_difference_steer.png"),
                          ("speed", "speed command difference (m/s)", "plot_seed_difference_speed.png")):
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.set_title(f"Re-seed commands, default vs reseed: {name}", loc="left", fontweight="bold", color=INK)
    for key, color, label in (("seed_diff", SEED, "seed commands"), ("plan_diff", PLAN, "plan the solver makes of them")):
        d = R[f"{key}_{name}"]
        ax.plot(k, d["mean"], color=color, lw=2.2, label=f"{label}: mean")
        ax.plot(k, d["p95"], color=color, lw=1.6, ls="--", label=f"{label}: p95")
    ax.set_xlabel("horizon step (0 = first command, 25 ms per step)")
    ax.set_ylabel(unit)
    ax.set_xlim(0, 39)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.savefig(HERE / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
print("wrote 2 plots")
