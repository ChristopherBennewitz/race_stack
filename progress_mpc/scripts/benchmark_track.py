#!/usr/bin/env python3
"""The progress MPC closed loop on a track with the car's measured delays: which MPC settings drive it cleanly?

Needs the f1tenth_parallel_sim environment (torch, CasADi, acados_template, ACADOS_SOURCE_DIR set) and the track in the
sim's `configs/maps/<name>/` (copy `stack_master/maps/<name>/`; drop the repeated closing point of
`centerline_waypoints` in `global_waypoints.json`, which the sim's Track cannot read). Run from anywhere:

    uv run --project ../f1tenth_parallel_sim python progress_mpc/scripts/benchmark_track.py --round 2
    PYTHONPATH=<build>/python ... --round 3        # the deployed C++ controller (needs the built progress_mpc module)

--round 1: margin / speed cap / horizon / compensation, 6 starts x 40 s.  --round 2: margins, horizons, the exported QP
settings, 10 starts x 60 s.  --round 3: the deployed C++ controller against the Python reference.  Results and the
decision: progress_mpc/docs/track_clearance.md.

Plant: the sim's vehicle model at 400 Hz, commands through the actuation manager (100 Hz tick, rate limits).
Controller: 40 Hz, sees the state delayed per channel (the new EKF's measured lag) plus 0..10 ms of message age, and its
command is available after the solve time. Footprint contact with the track limit ends a run (the car stops at a wall).
"""
import argparse, os, sys, tempfile, time, json, math, concurrent.futures as cf
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SIM = Path(os.environ.get("SIM_REPO", REPO.parent / "f1tenth_parallel_sim")).resolve()
MAP = os.environ.get("TRACK_MAP", "hangar_25_10_v")

SIM_DT = 1.0 / 400.0
CONTROL_EVERY = 10                      # plant ticks per controller tick (40 Hz)
SOLVE_TICKS = 3                         # command available this many ticks after the controller tick (~7.5 ms)
OBS_DELAY = dict(pos=0.015, yaw=0.020, vel=0.016, wz=0.018)   # measured stage delays (sim configs use +5 ms hold)
AGE_MAX = 0.010                         # message age uniform in [0, 10 ms] (100 Hz hold)
T_END = float(os.environ.get("T_END", 40.0))
HALF_LENGTH, HALF_WIDTH = 0.284, 0.148

VARIANTS = {
    "ideal (no delays)":      dict(cfg={}, delays=False, comp=False),
    "base":                   dict(cfg={}, delays=True, comp=False),
    "base + comp":            dict(cfg={}, delays=True, comp=True),
    "margin 0.07":            dict(cfg=dict(track_margin=0.07), delays=True, comp=False),
    "speed cap 3.0":          dict(cfg=dict(max_speed=3.0), delays=True, comp=False),
    "margin 0.07 + cap 3.5":  dict(cfg=dict(track_margin=0.07, max_speed=3.5), delays=True, comp=False),
    "margin 0.07 + cap 3.5 + comp": dict(cfg=dict(track_margin=0.07, max_speed=3.5), delays=True, comp=True),
    "horizon 30":             dict(cfg=dict(horizon_nodes=30), delays=True, comp=False),
    "margin 0.07 + horizon 30": dict(cfg=dict(track_margin=0.07, horizon_nodes=30), delays=True, comp=False),
}

QP = dict(qp=True)
VARIANTS2 = {
    "margin 0.07":                       dict(cfg=dict(track_margin=0.07), delays=True, comp=False),
    "margin 0.07 + exported QP":         dict(cfg=dict(track_margin=0.07), delays=True, comp=False, **QP),
    "margin 0.10 + exported QP":         dict(cfg=dict(track_margin=0.10), delays=True, comp=False, **QP),
    "margin 0.15 + exported QP":         dict(cfg=dict(track_margin=0.15), delays=True, comp=False, **QP),
    "margin 0.10 + horizon 50 + QP":     dict(cfg=dict(track_margin=0.10, horizon_nodes=50), delays=True, comp=False, **QP),
    "margin 0.10 + horizon 60 + QP":     dict(cfg=dict(track_margin=0.10, horizon_nodes=60), delays=True, comp=False, **QP),
    "margin 0.10 + cap 4.0 + QP":        dict(cfg=dict(track_margin=0.10, max_speed=4.0), delays=True, comp=False, **QP),
    "margin 0.10 + comp + QP":           dict(cfg=dict(track_margin=0.10), delays=True, comp=True, **QP),
}
VARIANTS3 = {
    "C++ deployed, comp on":      dict(cfg=dict(track_margin=0.10, max_speed=4.0), delays=True, comp=True, cpp=True),
    "C++ deployed, comp off":     dict(cfg=dict(track_margin=0.10, max_speed=4.0), delays=True, comp=False, cpp=True),
    "Python same settings, comp": dict(cfg=dict(track_margin=0.10, max_speed=4.0), delays=True, comp=True, qp=True),
    "C++ 5 m/s (no cap), comp on":  dict(cfg=dict(track_margin=0.10), delays=True, comp=True, cpp=True),
    "C++ 5 m/s (no cap), comp off": dict(cfg=dict(track_margin=0.10), delays=True, comp=False, cpp=True),
    "C++ deployed, no delays":    dict(cfg=dict(track_margin=0.10, max_speed=4.0), delays=False, comp=False, cpp=True),
}
ROUNDS = {"1": VARIANTS, "2": VARIANTS2, "3": VARIANTS3}
if os.environ.get("ROUND", "1") in ROUNDS:
    VARIANTS = ROUNDS[os.environ.get("ROUND", "1")]

def run_variant(name, starts):
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(SIM / "src"))
    os.chdir(SIM)
    from f1tenth_sim.config import CarConfig
    from f1tenth_sim.sim import BatchedSim
    from mpc.progress_mpc import ProgressMPC, ProgressMPCConfig
    spec = VARIANTS[name]
    cfg = ProgressMPCConfig.from_toml(SIM / "configs" / "progress_mpc.toml").model_copy(update=dict(
        track_dir=SIM / "configs" / "maps" / MAP, track_name=MAP,
        output_dir=Path(tempfile.mkdtemp()) / "m", **spec["cfg"]))
    if spec.get("qp"):
        class Exported(ProgressMPC):
            def _ocp(self):
                ocp = super()._ocp()
                ocp.solver_options.qp_solver_cond_N = 8
                ocp.solver_options.qp_tol = 1e-4
                return ocp
        mpc = Exported(cfg)
    else:
        mpc = ProgressMPC(cfg)
    cpp = None
    if spec.get("cpp"):
        sys.path.insert(0, str(REPO / "controller"))
        from progress_mpc import ProgressMpc
        cpp = ProgressMpc(0.1)
        cpp.set_track(np.asarray(mpc._center), np.asarray(mpc._w_left), np.asarray(mpc._w_right))
    sys.path.insert(0, str(REPO / "controller"))
    from controller.mpc.mpc_controller import extrapolate_state
    car = CarConfig.from_toml(SIM / "configs" / "f1tenth_car.toml")
    center, tangent, wl, wr, cum_s, L = mpc._center, mpc._tangent, mpc._w_left, mpc._w_right, mpc._cum_s, mpc._length
    rng = np.random.default_rng(0)
    results = []
    for i0 in starts:
        sim = BatchedSim(car, 1, sim_dt=SIM_DT, substeps=1, device="cpu")
        p0, t0 = center[i0], tangent[i0]
        z0 = np.array([p0[0], p0[1], np.arctan2(t0[1], t0[0]), 0.5, 0.0, 0.0, 0.0])
        sim.state = torch.as_tensor(z0[None, :], dtype=torch.float32)
        mpc.reset(None)
        if cpp is not None:
            cpp.reset()
        hist = [sim.state[0].numpy().copy()]
        pending = []                                  # (available_tick, steer, speed)
        cur = (0.0, 0.0)
        prog, s_prev = 0.0, cum_s[i0]
        stats = dict(speed=[], slack=[], sat=[], lat=[], head=[], fail0=(cpp.stats["failures"] if cpp is not None else mpc.total_solver_failures))
        crashed, t_crash = False, T_END
        n_ticks = int(T_END / SIM_DT)
        for k in range(n_ticks):
            if k % CONTROL_EVERY == 0:
                def at(delay):
                    pos = max(k - delay / SIM_DT, 0.0); i = int(pos); f = pos - i
                    a = hist[i]; b = hist[min(i + 1, len(hist) - 1)]
                    return a + f * (b - a)
                if spec["delays"]:
                    age = rng.uniform(0.0, AGE_MAX)
                    obs = np.zeros(7)
                    obs[0:2] = at(OBS_DELAY["pos"] + age)[0:2]; obs[2] = at(OBS_DELAY["yaw"] + age)[2]
                    obs[3:5] = at(OBS_DELAY["vel"] + age)[3:5]; obs[5] = at(OBS_DELAY["wz"] + age)[5]
                    obs[6] = hist[k][6]
                    if spec["comp"]:
                        if spec.get("cpp"):      # the controller's own numbers: stage lag 12 ms + measured age + 7.5 ms compute
                            dt = 0.012 + age + 0.0075
                        else:
                            dt = 0.017 + age + SOLVE_TICKS * SIM_DT
                        obs[0], obs[1], obs[2] = extrapolate_state(*obs[:6], dt)
                else:
                    obs = hist[k].copy()
                if cpp is not None:
                    steer, speed = cpp.step(np.asarray(obs[:6]))
                    cpp.report_applied(steer, speed)
                    stats["slack"].append(cpp.stats["slack_max"])
                else:
                    steer, speed = mpc.step(obs)
                    mpc.report_applied(steer, speed)
                    stats["slack"].append(mpc.last_geom_slack_max)
                pending.append((k + SOLVE_TICKS, steer, speed)); stats["sat"].append(abs(steer) >= 0.29)
            while pending and pending[0][0] <= k:
                _, st, sp = pending.pop(0); cur = (st, sp)
            sim.drive(torch.tensor([cur[0]], dtype=torch.float32), torch.tensor([cur[1]], dtype=torch.float32))
            state = sim.state[0].numpy().copy(); hist.append(state)
            if k % CONTROL_EVERY == 0:
                x, y, yaw = state[0], state[1], state[2]
                j = int(np.argmin(np.sum((center - [x, y]) ** 2, axis=1)))
                ds = cum_s[j] - s_prev; ds = (ds + L / 2) % L - L / 2; prog += ds; s_prev = cum_s[j]
                off = np.array([x, y]) - center[j]; lat = -tangent[j][1] * off[0] + tangent[j][0] * off[1]
                head = math.atan2(math.sin(yaw - math.atan2(tangent[j][1], tangent[j][0])), math.cos(yaw - math.atan2(tangent[j][1], tangent[j][0])))
                stats["lat"].append(lat); stats["head"].append(head); stats["speed"].append(state[3])
                c, sn = math.cos(yaw), math.sin(yaw)
                for ax, ay in ((HALF_LENGTH, HALF_WIDTH), (HALF_LENGTH, -HALF_WIDTH), (-HALF_LENGTH, HALF_WIDTH), (-HALF_LENGTH, -HALF_WIDTH)):
                    px, py = x + c * ax - sn * ay, y + sn * ax + c * ay
                    jj = int(np.argmin(np.sum((center - [px, py]) ** 2, axis=1))); o = np.array([px, py]) - center[jj]
                    lt = -tangent[jj][1] * o[0] + tangent[jj][0] * o[1]
                    if lt > wl[jj] or -lt > wr[jj]:
                        crashed = True
                if crashed:
                    t_crash = k * SIM_DT; break
        lat = np.array(stats["lat"]); 
        results.append(dict(start=int(i0), crashed=bool(crashed), t=float(t_crash), progress=float(prog),
                            v_mean=float(np.mean(stats["speed"])), v_max=float(np.max(stats["speed"])),
                            slack05=float(np.mean(np.array(stats["slack"]) > 0.05)), sat=float(np.mean(stats["sat"])),
                            lat_rms=float(np.sqrt(np.mean(lat ** 2))), head_rms_deg=float(np.degrees(np.sqrt(np.mean(np.array(stats["head"]) ** 2)))),
                            fails=int((cpp.stats["failures"] if cpp is not None else mpc.total_solver_failures) - stats["fail0"])))
    return name, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", choices=list(ROUNDS), default="1")
    parser.add_argument("--seconds", type=float, help="simulated time per start (default 40 s, 60 s for rounds 2 and 3)")
    parser.add_argument("--map", default=MAP, help="folder in the sim's configs/maps")
    parser.add_argument("variants", nargs="*", help="run only these variants of the round")
    args = parser.parse_args()
    os.environ["ROUND"], os.environ["TRACK_MAP"] = args.round, args.map     # the worker processes read these
    os.environ["T_END"] = str(args.seconds or (40.0 if args.round == "1" else 60.0))
    VARIANTS = ROUNDS[args.round]
    sys.argv = [sys.argv[0], *args.variants]
    starts = [10, 35, 60, 85, 110, 135, 160, 185, 210, 235] if os.environ.get("ROUND") in ("2", "3") else [10, 53, 96, 139, 182, 225]
    names = args.variants or list(VARIANTS)
    t0 = time.time()
    out = {}
    with cf.ProcessPoolExecutor(max_workers=min(len(names), 9), mp_context=__import__("multiprocessing").get_context("spawn")) as pool:
        for name, res in pool.map(run_variant, names, [starts] * len(names)):
            out[name] = res
            print(f"done {name} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open(REPO / "progress_mpc" / "docs" / f"track_benchmark_round{args.round}.json", "w"), indent=1)
    print(f"\n{'variant':30s} {'crashes':>7s} {'t until crash / end (mean s)':>28s} {'progress m':>11s} {'v mean/max':>11s} {'slack>5cm':>9s} {'steer@lock':>10s} {'lat rms':>8s} {'head rms':>8s} {'fails':>5s}")
    for name, res in out.items():
        m = lambda k: np.mean([r[k] for r in res])
        print(f"{name:30s} {sum(r['crashed'] for r in res):4d}/{len(res)} {m('t'):22.1f}       {m('progress'):11.1f} {m('v_mean'):5.2f}/{max(r['v_max'] for r in res):4.2f} {100*m('slack05'):8.0f}% {100*m('sat'):9.0f}% {m('lat_rms'):8.2f} {m('head_rms_deg'):7.1f}° {sum(r['fails'] for r in res):5d}")
