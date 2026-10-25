#!/usr/bin/env python3
"""Smoothness against fidelity for the global filter's pose-related noise (no fitting).

The one-step-ahead likelihood cannot choose ``R_slam`` and the position process noise: scan-matching error is correlated
in time, so the likelihood always prefers a filter that follows the previous pose (``ekf_tune.py --global`` shows it).
They are a trade-off, so this tool lays the trade-off out. It replays the global filter over every bag and pose track
for a grid of ``R_slam`` scales and ``Q[x], Q[y]`` values (everything else from the local fit, ``defaults.NOISE``) and
reports, per setting:

- jitter: RMS acceleration (m/s^2) implied by the second difference of the filter's position at 100 Hz. The local
  filter, which never sees SLAM and is smooth by construction, is the target: a good global filter is not noisier
  than it. Part of the number is the real acceleration, so read it relative to the local row.
- tracking: RMS size (m) of the SLAM-pose innovations, i.e. how far the filter's own prediction is from each pose.
  Small means the filter follows SLAM closely; it must stay near the pose's own noise (a few cm) to be anchored.
- re-seeds and gated poses: each re-seed is a jump in the output.
- pose NIS: 3 when the pose variance matches the innovations.

    f1tenth_parallel_sim/.venv/bin/python misc/ekf_global_sweep.py sysid_bags/sysid_* --pose-dir sysid_poses
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekf_replay import Engine, FilterConfig, SlamConfig, load_pose_file, replay  # noqa: E402
from state_estimation.ekf_node import defaults  # noqa: E402

_state: dict = {}


def jitter(rows):
    """RMS acceleration (m/s^2) from the second difference of the filter's base_link position, over consecutive ticks."""
    t, xy = rows[:, 0], rows[:, 11:13]
    ok = (np.abs(np.diff(t)[1:] - 0.01) < 1e-6) & (np.abs(np.diff(t)[:-1] - 0.01) < 1e-6)
    accel = (xy[2:] - 2 * xy[1:-1] + xy[:-2]) / 0.01 ** 2
    return float(np.sqrt((np.linalg.norm(accel[ok], axis=1) ** 2).mean()))


def settings(r_scales, q_positions):
    """Name -> (FilterConfig, SlamConfig): the local fit's dynamics with the pose-related noise varied."""
    local = defaults.NOISE
    out = {}
    for q in q_positions:
        for scale in r_scales:
            config = FilterConfig()  # the local fit: Q, IMU and VESC noise
            config.Q = np.diag(np.array(local['Q'], dtype=float))
            config.Q[0, 0] = config.Q[1, 1] = q
            slam = SlamConfig(R=np.diag(np.array(defaults.SLAM['R_slam_pose']) * scale))
            out[f'Qpos {q:.0e}  R_slam x{scale:g}'] = (config, slam)
    return out


def _init(cache_dir, pose_dir):
    _state['engine'] = Engine()
    _state['cache'], _state['poses'] = cache_dir, pose_dir


def _run(task):
    import ekf_tune

    bag_path, name, config, slam = task
    bag = _state.setdefault('bags', {}).get(bag_path)
    if bag is None:
        bag = _state['bags'][bag_path] = ekf_tune.load_bag(bag_path, _state['cache'])
    if slam is None:  # the local filter, the smoothness target
        rows, stats = replay(bag, config, _state['engine'])
        return name, bag_path, jitter(rows), None
    pose = load_pose_file(Path(_state['poses']) / (Path(bag_path).name + '.csv'))
    slam = SlamConfig(R=slam.R, delay=slam.delay, gate=slam.gate, reseed_after=slam.reseed_after, pose=pose)
    rows, stats = replay(bag, config, _state['engine'], slam)
    s = stats['slam']
    inn = s['innovations']
    return name, bag_path, jitter(rows), {
        'track': float(np.sqrt((inn[:, :2] ** 2).sum(1).mean())), 'nis': float(np.mean(s['nis'])), 'gated': s['gated'],
        'reseeds': s['reseeds'], 'vx': float(np.sqrt(((rows[:, 1] - rows[:, 4]) ** 2).mean())),
        'wz': float(np.sqrt(((rows[:, 3] - rows[:, 6]) ** 2).mean()))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bags', nargs='+')
    ap.add_argument('--pose-dir', required=True)
    ap.add_argument('--r-scales', type=float, nargs='+', default=[0.3, 1.0, 3.0, 10.0], help='multiples of the node\'s R_slam_pose')
    ap.add_argument('--q-positions', type=float, nargs='+', default=[4.9e-5, 5e-4, 5e-3], help='Q[x] = Q[y] values')
    ap.add_argument('--cache-dir', default=str(Path.home() / '.cache' / 'ekf_tune'))
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    bags = [b for b in args.bags if Path(b).is_dir()]

    grid = settings(args.r_scales, args.q_positions)
    grid['old global defaults'] = (FilterConfig.for_global(), SlamConfig())
    tasks = [(b, name, c, s) for name, (c, s) in grid.items() for b in bags]
    tasks += [(b, 'local filter (target)', FilterConfig(), None) for b in bags]
    with mp.get_context('spawn').Pool(args.workers, initializer=_init, initargs=(args.cache_dir, args.pose_dir)) as pool:
        results = pool.map(_run, tasks, chunksize=1)

    print(f'{"setting":34s} {"jitter":>7s} {"tracking":>9s} {"NIS/3":>6s} {"gated":>6s} {"reseeds":>8s}   vx     wz     (all {len(bags)} takes)')
    for name in list(grid) + ['local filter (target)']:
        rows = [r for r in results if r[0] == name]
        jit = np.sqrt(np.mean([r[2] ** 2 for r in rows]))
        if rows[0][3] is None:
            print(f'{name:34s} {jit:7.0f}')
            continue
        m = [r[3] for r in rows]
        print(f'{name:34s} {jit:7.0f} {np.mean([x["track"] for x in m]) * 100:7.1f} cm {np.mean([x["nis"] for x in m]):6.2f} '
              f'{sum(x["gated"] for x in m):6d} {sum(x["reseeds"] for x in m):8d}  {np.mean([x["vx"] for x in m]):.3f}  '
              f'{np.mean([x["wz"] for x in m]):.3f}')


if __name__ == '__main__':
    main()
