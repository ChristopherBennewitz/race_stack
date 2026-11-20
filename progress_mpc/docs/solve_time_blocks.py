"""Solve times of the QP settings and of the AC4MPC step as blocks (SVG + PNG), same style as ac4mpc_diagram.py.

    uv run --project ../f1tenth_parallel_sim python progress_mpc/docs/solve_time_blocks.py

The numbers are the ones in solve_time.md (laptop, 2026-11-15): benchmark_solver.py for the plain solver, the C++ Ac4Mpc in
`full` mode with the real checkpoint for the AC4MPC step (mean of two repetitions).
"""

import subprocess
from pathlib import Path

HERE = Path(__file__).parent
GREY, BLUE, GREEN, TAN = "#e4e4e4", "#b5c1d3", "#b3c6ad", "#d5b9a4"
W, H = 1000, 590
out: list[str] = []


def box(x, y, w, h, fill, title, *lines, size=18, line_size=12.5, bold_last=False):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="#000" stroke-width="1.5"/>')
    n = len(lines)
    ty = y + 34
    out.append(f'<text x="{x + w / 2}" y="{ty}" text-anchor="middle" font-size="{size}" font-weight="bold">{title}</text>')
    for i, line in enumerate(lines):
        weight = ' font-weight="bold"' if bold_last and i == n - 1 else ""
        out.append(f'<text x="{x + w / 2}" y="{ty + 22 + 19 * i}" text-anchor="middle" font-size="{line_size}"{weight}>{line}</text>')


def text(x, y, s, size=12.5, anchor="start", color="#333", bold=False):
    weight = ' font-weight="bold"' if bold else ""
    out.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}"{weight}>{s}</text>')


# ---------------------------------------------------------------- the solver alone
text(30, 38, "QP settings: solver time per step of the plain MPC", size=15, color="#000", bold=True)
text(30, 58, "laptop, benchmark_solver.py, 8 starts x 250 steps. cond_N 8 (partial condensing) is the big win; the tolerance below 1e-5 changes little.", size=12, color="#444")
rows = [  # (time, setting lines, fill, note)
    ("8.05 ms", ("default cond_N", "tol 1e-6"), GREY, "1.0x"),
    ("2.46 ms", ("default cond_N", "tol 1e-4"), BLUE, "3.3x"),
    ("4.17 ms", ("cond_N 8", "tol 1e-6"), BLUE, "1.9x"),
    ("1.39 ms", ("cond_N 8", "tol 1e-5"), BLUE, "5.8x"),
    ("1.37 ms", ("cond_N 8", "tol 1e-4"), BLUE, "5.9x"),
    ("1.33 ms", ("cond_N 8", "tol 1e-3"), BLUE, "6.1x"),
    ("1.27 ms", ("cond_N 8", "tol 1e-2", "(export default)"), GREEN, "6.3x"),
]
bw, gap, x0, y0 = 125, 12, 26, 80
for i, (t, lines, fill, speedup) in enumerate(rows):
    x = x0 + i * (bw + gap)
    box(x, y0, bw, 118, fill, t, *lines)
    text(x + bw / 2, y0 + 138, f"{speedup} faster" if speedup != "1.0x" else "reference", size=12, anchor="middle")

# ---------------------------------------------------------------- the AC4MPC step
text(30, 268, "AC4MPC: time of one control step (actor + two solvers, three threads)", size=15, color="#000", bold=True)
text(30, 288, "laptop, real checkpoint, mode full, 8 starts x 300 steps, mean of two runs. The NUC7 is ~3.5x slower.", size=12, color="#444")
variants = [
    ("tol 1e-4", "default", TAN, (3.4, 7.1, 8.7), (5.1, 9.3, 12.2)),
    ("tol 1e-4", "+ reseed_from_previous", BLUE, (3.1, 4.9, 6.7), (5.0, 7.5, 9.8)),
    ("tol 1e-2", "default (exported now)", TAN, (3.3, 6.8, 8.4), (5.0, 9.1, 11.6)),
    ("tol 1e-2", "+ reseed_from_previous", GREEN, (2.9, 4.6, 6.0), (4.8, 7.2, 8.7)),
]
bw2, gap2, x02, y02 = 228, 12, 26, 310
for i, (tol, name, fill, allc, two) in enumerate(variants):
    x = x02 + i * (bw2 + gap2)
    box(x, y02, bw2, 200, fill, tol, name, "",
        "all cores", f"p50 {allc[0]:.1f}   p99 {allc[1]:.1f}   max {allc[2]:.1f} ms", "",
        "2 cores", f"p50 {two[0]:.1f}   p99 {two[1]:.1f}   max {two[2]:.1f} ms", line_size=12.5)

text(30, 548, "Driving outcome is unchanged in every variant: path 22.75 to 22.88 m, track violation 0 m, 0 solver failures.",
     size=12, color="#444")
text(30, 568, "reseed_from_previous: the parallel solver's re-seed uses the previous step's actor plan, shifted, instead of a fresh rollout.",
     size=12, color="#444")

svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
    'font-family="DejaVu Sans, Arial, sans-serif" fill="#000">\n'
    f'<rect width="{W}" height="{H}" fill="#fff"/>\n' + "\n".join(out) + "\n</svg>\n"
)
(HERE / "solve_time_blocks.svg").write_text(svg)
subprocess.run(["uvx", "cairosvg", str(HERE / "solve_time_blocks.svg"), "-o", str(HERE / "solve_time_blocks.png"), "-s", "2"], check=True)
