#!/usr/bin/env python3
"""The smooth map-frame pose (local odometry + low-passed map->odom) against the global EKF, over a time constant.

Replays the local filter on each take once, then feeds its pose and the SLAM poses to ``MapOdomCorrection`` for each
``tau`` and reports what ``ekf_global_sweep.py`` reports for the global EKF, so the rows are comparable:

- jitter: RMS acceleration (m/s^2) implied by the second difference of the output position at 100 Hz (local filter: the
  target; the global EKF with its adopted settings: 89).
- tracking: RMS distance (cm) between each SLAM pose and what the output said at that moment (of the output).
- lag: the RMS of the same, restricted to the drift the local odometry accumulates, is not separable from pose noise;
  read tracking together with jitter. A tau that is too large shows as tracking growing.
- rejected poses and jumps (each jump is a step in the output).

    f1tenth_parallel_sim/.venv/bin/python misc/ekf_correction_sweep.py sysid_bags/sysid_* --pose-dir sysid_poses
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekf_global_sweep import jitter  # noqa: E402
from ekf_replay import Engine, FilterConfig, SlamConfig, load_pose_file, replay, wrap  # noqa: E402
from state_estimation.ekf_node import defaults  # noqa: E402
from state_estimation.ekf_node.map_odom_correction import MapOdomCorrection  # noqa: E402


def yaw_jitter(rows):
    """RMS angular acceleration (rad/s^2) of the output heading, from its second difference at 100 Hz."""
    d = np.diff(rows[:, 0])
    ok = (np.abs(d[1:] - 0.01) < 1e-6) & (np.abs(d[:-1] - 0.01) < 1e-6)
    yaw = np.unwrap(rows[:, 13])
    return float(np.sqrt((((yaw[2:] - 2 * yaw[1:-1] + yaw[:-2]) / 0.01 ** 2)[ok] ** 2).mean()))


def output_tracking(rows, pose):
    """RMS distance (m) between each SLAM pose and the output at the pose's content time. It includes the pose's own
    noise, so compare settings with each other, not with zero; a lag shows up as growth."""
    t = pose[:, 1] - defaults.SLAM['pose_delay']
    inside = (t > rows[0, 0]) & (t < rows[-1, 0])
    x = np.interp(t[inside], rows[:, 0], rows[:, 11])
    y = np.interp(t[inside], rows[:, 0], rows[:, 12])
    return float(np.sqrt(((x - pose[inside, 2]) ** 2 + (y - pose[inside, 3]) ** 2).mean()))


def speeds(rows):
    """RMSE (vx, wz) of the output twist against the reference."""
    return float(np.sqrt(((rows[:, 1] - rows[:, 4]) ** 2).mean())), float(np.sqrt(((rows[:, 3] - rows[:, 6]) ** 2).mean()))


def correct(rows, pose, tau, latency, smooth=0.0):
    """Runs the correction over a local replay's rows (pose at columns 11:14); returns (output x, y, yaw per row, stats); ``stats['start']`` is the first row with a pose."""
    c = defaults.CORRECTION
    corr = MapOdomCorrection(tau, c['pose_delay'], c['gate_distance'], c['gate_angle'], c['reseed'], smooth)
    out = np.zeros((len(rows), 3))
    inn, outcomes, i, start = [], {}, 0, None
    received = pose[:, 1] + latency
    for k, row in enumerate(rows):
        corr.add_local(row[0], row[11:14])
        while i < len(pose) and received[i] <= row[0]:
            outcome = corr.add_pose(pose[i, 1], pose[i, 2:5])
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            if outcome in ('updated', 'gated') and corr.last_innovation is not None:
                inn.append(corr.last_innovation)
            i += 1
        out[k] = corr.output(row[11:14], row[0])
        if start is None and corr.transform is not None:
            start = k  # the node publishes nothing before the first pose; the output starts in the map frame here
    inn = np.array(inn)
    return out, {'track': float(np.sqrt((inn[:, :2] ** 2).sum(1).mean())) if len(inn) else float('nan'),
                 'yaw': float(np.sqrt((inn[:, 2] ** 2).mean())) if len(inn) else float('nan'),
                 'rejected': corr.rejected, 'jumps': corr.jumps, 'start': start or 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bags', nargs='+')
    ap.add_argument('--pose-dir', required=True)
    ap.add_argument('--taus', type=float, nargs='+', default=[defaults.CORRECTION['tau']])
    ap.add_argument('--r-yaw', type=float, nargs='+', default=[None], help='IMU orientation variances to try (default: the local filter\'s)')
    ap.add_argument('--smooth', type=float, nargs='+', default=[defaults.CORRECTION['smooth']], help='extra output lag on the correction, seconds')
    ap.add_argument('--latency', type=float, default=0.007, help='pose receive time minus stamp, seconds')
    ap.add_argument('--cache-dir', default=str(Path.home() / '.cache' / 'ekf_tune'))
    args = ap.parse_args()
    import ekf_tune

    engine = Engine()
    bags = [(Path(b), ekf_tune.load_bag(b, args.cache_dir)) for b in args.bags if Path(b).is_dir()]
    poses = {b.name: load_pose_file(Path(args.pose_dir) / (b.name + '.csv')) for b, _ in bags}
    print(f'{len(bags)} takes')
    for r_yaw in args.r_yaw:
        config = FilterConfig()
        if r_yaw is not None:
            config.R_imu[0, 0] = r_yaw
        takes = [(b.name, replay(bag, config, engine)[0], poses[b.name]) for b, bag in bags]
        label = 'default' if r_yaw is None else f'R_yaw {r_yaw:g}'
        print(f'local filter, {label}: jitter {np.sqrt(np.mean([jitter(r) ** 2 for _, r, _ in takes])):.0f}, '
              f'yaw jitter {np.sqrt(np.mean([yaw_jitter(r) ** 2 for _, r, _ in takes])):.1f}, '
              f'vx {np.mean([speeds(r)[0] for _, r, _ in takes]):.3f} wz {np.mean([speeds(r)[1] for _, r, _ in takes]):.3f}', flush=True)
        header = f'{"":34s} {"jitter":>7s} {"yaw jit":>8s} {"tracking":>9s} {"vx":>6s} {"wz":>6s} {"rejected":>9s} {"jumps":>6s}'
        print(header)
        for tau in args.taus:
            for smooth in args.smooth:
                res = []
                for _, rows, pose in takes:
                    out, s = correct(rows, pose, tau, args.latency, smooth)
                    shown = rows[s['start']:].copy()
                    shown[:, 11:14] = out[s['start']:]
                    s.update(track=output_tracking(shown, pose), jitter=jitter(shown), yaw_jitter=yaw_jitter(shown), vx=speeds(shown)[0], wz=speeds(shown)[1])
                    res.append(s)
                rms = lambda key: np.sqrt(np.mean([s[key] ** 2 for s in res]))
                label = f'  correction tau {tau:g} s, smooth {smooth:g} s'
                print(f'{label:34s} {rms("jitter"):7.0f} {rms("yaw_jitter"):8.1f} {np.mean([s["track"] for s in res]) * 100:7.1f} cm '
                      f'{np.mean([s["vx"] for s in res]):6.3f} {np.mean([s["wz"] for s in res]):6.3f} '
                      f'{sum(s["rejected"] for s in res):9d} {sum(s["jumps"] for s in res):6d}', flush=True)
        if r_yaw is None:  # the global EKF with the adopted settings, for the same columns
            local = defaults.NOISE
            res = []
            for (name, bag), (_, _, pose) in zip(bags, takes):
                config = FilterConfig()
                config.Q = np.diag(np.array(defaults.NOISE_GLOBAL['Q'], dtype=float))
                slam = SlamConfig(R=np.diag(defaults.SLAM['R_slam_pose']), pose=pose)
                rows, stats = replay(bag, config, engine, slam)
                res.append({'jitter': jitter(rows), 'yaw_jitter': yaw_jitter(rows), 'track': output_tracking(rows, pose),
                            'vx': speeds(rows)[0], 'wz': speeds(rows)[1]})
            rms = lambda key: np.sqrt(np.mean([s[key] ** 2 for s in res]))
            print(f'{"  global EKF (adopted)":34s} {rms("jitter"):7.0f} {rms("yaw_jitter"):8.1f} {np.mean([s["track"] for s in res]) * 100:7.1f} cm '
                  f'{np.mean([s["vx"] for s in res]):6.3f} {np.mean([s["wz"] for s in res]):6.3f}', flush=True)


if __name__ == '__main__':
    main()
