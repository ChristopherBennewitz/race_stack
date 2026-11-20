"""Plots of the solve times in solve_time.md (laptop, 2026-11-15), one PNG per plot.

    uv run --project ../f1tenth_parallel_sim python progress_mpc/docs/solve_time_plots.py

Writes next to this file:
    plot_qp_settings.png        solver time of the plain MPC per qp_tol
    plot_ac4mpc_all_cores.png   AC4MPC step p50 / p99 / max per setting, all cores
    plot_ac4mpc_2_cores.png     the same pinned to 2 cores (stand-in for the car's P-cores)
    plot_reseed_steps.png       consecutive AC4MPC steps, the re-seed steps shaded
    plot_reseed_cycle.png       mean step time by position in the 5-step re-seed cycle
    plot_reseed_cdf.png         distribution of the AC4MPC step times
The last three read reseed_step_times.json (per-step timings of the run behind them).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
INK, MUTED, GRID = "#111111", "#555555", "#e5e5e5"
BEFORE, AFTER, REF, GOOD = "#e0662f", "#2a78d6", "#9aa0a8", "#1d8a5a"
RAMP = ["#9ec5f4", "#3987e5", "#184f95"]  # p50, p99, max
plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})


def new(title, size=(7.2, 5.0)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_title(title, loc="left", fontweight="bold", color=INK)
    return fig, ax


def grid(ax, axis):
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(HERE / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------ QP settings
fig, ax = new("QP tolerance: solver time of the plain MPC",
              size=(7.6, 4.6))
tols = ["1e-6\n(acados default)", "1e-5", "1e-4\n(previous export)", "1e-3", "1e-2\n(export default)", "1e-1"]
vals = [4.16, 1.40, 1.37, 1.33, 1.28, 1.19]
paths = [18.39, 18.31, 18.41, 18.50, 18.43, 17.76]
y = np.arange(len(tols))[::-1]
ax.barh(y, vals, color=[REF, AFTER, "#184f95", AFTER, GOOD, BEFORE], height=0.62)
for yi, v, p in zip(y, vals, paths):
    ax.text(v + 0.08, yi, f"{v:.2f} ms   path {p:.1f} m", va="center", fontsize=9.5, color=INK)
ax.set_title("QP tolerance: solver time of the plain MPC", loc="left", fontweight="bold", color=INK, pad=24)
ax.text(0, 1.025, "path: distance driven, evaluated over 250 steps (6.25 s) on the THETRACK lap (31 m), 8 start points",
        transform=ax.transAxes, fontsize=9.5, color=INK, va="bottom")
ax.set_yticks(y, ["qp_tol " + t for t in tols], fontsize=9, color=INK)
ax.set_xlim(0, 6.3)
ax.set_xlabel("solver time per step (ms)")
grid(ax, "x")
save(fig, "plot_qp_settings.png")

# ------------------------------------------------------------------------------------------------------ AC4MPC step
variants = ["tol 1e-4\ndefault", "tol 1e-4\n+ reseed", "tol 1e-2\ndefault", "tol 1e-2\n+ reseed"]
step = {
    "all cores": [(3.4, 7.1, 8.7), (3.1, 4.9, 6.7), (3.3, 6.8, 8.4), (2.9, 4.6, 6.0)],
    "2 cores": [(5.1, 9.3, 12.2), (5.0, 7.5, 9.8), (5.0, 9.1, 11.6), (4.8, 7.2, 8.7)],
}
for key, fname, title in (("all cores", "plot_ac4mpc_all_cores.png", "AC4MPC step time, all cores"),
                          ("2 cores", "plot_ac4mpc_2_cores.png", "AC4MPC step time, pinned to 2 cores")):
    fig, ax = new(title)
    x = np.arange(len(variants))
    bw = 0.26
    for j, m in enumerate(("p50", "p99", "max")):
        v = [row[j] for row in step[key]]
        ax.bar(x + (j - 1) * bw, v, bw * 0.92, color=RAMP[j], label=m)
        for xi, vi in zip(x + (j - 1) * bw, v):
            ax.text(xi, vi + 0.15, f"{vi:.1f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, variants, fontsize=9.5, color=INK)
    ax.set_ylim(0, 14)
    ax.set_ylabel("step time (ms)")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    grid(ax, "y")
    save(fig, fname)

# ------------------------------------------------------------------------------------------------------ re-seed steps
raw = json.loads((HERE / "reseed_step_times.json").read_text())
steps = {k: np.array(raw[k]["step"]) for k in ("before", "after")}
ms = {k: np.array(raw[k]["ms"]) for k in ("before", "after")}
colors = {"before": BEFORE, "after": AFTER}
labels = {"before": "default", "after": "reseed_from_previous"}
fig, ax = new("Consecutive AC4MPC steps", size=(9.0, 4.8))
lo, hi = 20, 80
for xi in range(lo, hi):
    if xi % 5 == 0:
        ax.axvspan(xi - 0.5, xi + 0.5, color="#eeeeee", zorder=0)
for k in ("before", "after"):
    sel = steps[k][:280] < hi
    ax.plot(steps[k][:280][sel], ms[k][:280][sel], color=colors[k], lw=1.8, marker="o", ms=3.5, label=labels[k])
ax.set_xlim(lo - 0.5, hi - 0.5)
ax.set_ylim(0, 9)
ax.set_xlabel("control step (grey: re-seed steps, every 5th)")
ax.set_ylabel("step time (ms)")
ax.legend(frameon=False, loc="upper right", ncol=2)
grid(ax, "y")
save(fig, "plot_reseed_steps.png")

fig, ax = new("Step time by position in the re-seed cycle")
phase = np.arange(5)
w = 0.38
for j, k in enumerate(("before", "after")):
    mean = [ms[k][steps[k] % 5 == p].mean() for p in phase]
    p99 = [np.percentile(ms[k][steps[k] % 5 == p], 99) for p in phase]
    ax.bar(phase + (j - 0.5) * w, mean, w * 0.92, color=colors[k], label=f"{labels[k]}: mean")
    ax.plot(phase + (j - 0.5) * w, p99, ls="none", marker="_", ms=16, mew=2, color=INK)
ax.plot([], [], ls="none", marker="_", ms=10, mew=2, color=INK, label="p99")
ax.set_xticks(phase, ["re-seed\nstep", "+1", "+2", "+3", "+4"])
ax.set_ylim(0, 11)
ax.set_xlabel("position in the 5-step cycle")
ax.set_ylabel("step time (ms)")
ax.legend(frameon=False, loc="upper right")
grid(ax, "y")
save(fig, "plot_reseed_cycle.png")

fig, ax = new("Distribution of AC4MPC step times")
for k in ("before", "after"):
    s = np.sort(ms[k])
    ax.plot(s, 100 * np.arange(1, len(s) + 1) / len(s), color=colors[k], lw=2, label=labels[k])
    p99 = np.percentile(s, 99)
    ax.plot([p99], [99], marker="o", color=colors[k], ms=6)
    ax.annotate(f"p99 {p99:.1f} ms", (p99, 99), xytext=(p99 + 0.15, 90 if k == "before" else 84), fontsize=9.5, color=colors[k],
                arrowprops=dict(arrowstyle="-", color=colors[k], lw=0.8))
ax.set_xlim(2, 9)
ax.set_ylim(40, 100.5)
ax.set_xlabel("step time (ms)")
ax.set_ylabel("share of steps at or below (%)")
ax.legend(frameon=False, loc="lower right")
grid(ax, "both")
save(fig, "plot_reseed_cdf.png")
print("wrote 6 plots")
