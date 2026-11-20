"""AC4MPC warm-start diagram (SVG + PNG), in the style of f1tenth_parallel_sim/artifacts/rl/policy_diagram.py.

    uv run --project ../f1tenth_parallel_sim python progress_mpc/docs/ac4mpc_diagram.py

What `shift()` does to a plan: how the previous plan becomes the next warm start.
"""

import math
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
BG, BLUE, PURPLE = "#ffffff", "#b5c1d3", "#c8b3c8"
NEW = "#e9c9a6"
W, H = 1000, 300
out: list[str] = []


def box(x, y, w, h, fill, title, *lines, size=15, dashed=False):
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="#000" stroke-width="1.5"{dash}/>')
    n = len(lines)
    ty = y + h / 2 - 9.5 * n + (6 if n else 5)
    out.append(f'<text x="{x + w / 2}" y="{ty}" text-anchor="middle" font-size="{size}" font-weight="bold">{title}</text>')
    for i, line in enumerate(lines):
        out.append(f'<text x="{x + w / 2}" y="{ty + 19 * (i + 1)}" text-anchor="middle" font-size="12.5">{line}</text>')


def arrow(*pts, dashed=False, color="#000"):
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    out.append(f'<polyline points="{" ".join(f"{x},{y}" for x, y in pts)}" fill="none" stroke="{color}" stroke-width="1.5"{dash}/>')
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    a = math.atan2(y1 - y0, x1 - x0)
    p = [(x1, y1)] + [(x1 - 10 * math.cos(a + s * 0.45), y1 - 10 * math.sin(a + s * 0.45)) for s in (1, -1)]
    out.append(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in p)}" fill="{color}"/>')


def text(x, y, s, size=12.5, anchor="middle", color="#333", bold=False):
    weight = ' font-weight="bold"' if bold else ""
    out.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}"{weight}>{s}</text>')


# ---------------------------------------------------------------- shift()
text(60, 756, "shift(): how a plan becomes the next warm start", size=15, anchor="start", color="#000", bold=True)
text(60, 777, "A plan is 40 commands, one per 25 ms. One step later 39 of them are still valid.", size=12, anchor="start", color="#444")
cw, gap, x0 = 66, 8, 200
labels_old = ["u0", "u1", "u2", "u3", "u4", "u5", "u6", "…", "u39"]
labels_new = ["u1", "u2", "u3", "u4", "u5", "u6", "…", "u39", "u40"]
y1, y2 = 800, 885
text(60, y1 + 25, "plan at step k", anchor="start", color="#000", bold=True)
text(60, y2 + 25, "guess at step k+1", anchor="start", color="#000", bold=True)
for i, (a, b) in enumerate(zip(labels_old, labels_new)):
    x = x0 + i * (cw + gap)
    box(x, y1, cw, 38, BLUE, a, size=13)
    box(x, y2, cw, 38, NEW if b == "u40" else BLUE, b, size=13)
    if i >= 1:
        arrow((x + cw / 2, y1 + 38), (x - (cw + gap) + cw / 2 + 6, y2), color="#666")
text(x0 + cw / 2, y1 - 8, "u0: applied to the car", size=11.5, anchor="start")
text(x0 + 8 * (cw + gap) + cw / 2, y2 + 56, "new: the actor's command at the old plan's", size=11.5, anchor="end")
text(x0 + 8 * (cw + gap) + cw / 2, y2 + 71, "end state, limited to the allowed step", size=11.5, anchor="end")
box(x0, 970, 470, 40, PURPLE, "re-simulate from the measured state x0", size=13)
box(x0 + 560, 970, 190, 40, BLUE, "one RTI iteration", size=13)
arrow((x0 + 235, y2 + 38), (x0 + 235, 970))
arrow((x0 + 470, 990), (x0 + 560, 990))

svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
    'font-family="DejaVu Sans, Arial, sans-serif" fill="#000">\n'
    f'<rect width="{W}" height="{H}" fill="{BG}"/>\n<g transform="translate(0,-730)">\n' + "\n".join(out) + "\n</g>\n</svg>\n"
)
(HERE / "ac4mpc_diagram.svg").write_text(svg)
subprocess.run(["uvx", "cairosvg", str(HERE / "ac4mpc_diagram.svg"), "-o", str(HERE / "ac4mpc_diagram.png"), "-s", "2"], check=True)
