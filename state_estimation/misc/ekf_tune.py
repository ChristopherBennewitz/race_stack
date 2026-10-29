#!/usr/bin/env python3
"""Fits the local EKF's noise (Q, the IMU and VESC R) to recorded bags by maximum likelihood.

The filter's one-step-ahead prediction errors (innovations) are Gaussian with covariance S when the noise is
right, so the right noise minimizes the summed negative log-likelihood ``0.5 (y' S^-1 y + log det S)`` over every
sensor update. That needs no ground truth. CMA-ES searches the log10 of the diagonal noise within ``--decades`` of
the node's defaults.

Speed: a candidate is scored on a bag by ``ekf_fast`` (the whole replay as one compiled CasADi call, ~0.1 s per bag)
in a pool of worker processes, one task per (candidate, bag). ``ekf_replay.replay`` is the reference implementation:
the fast path is checked against it before the search and the final report uses it.

Bags the search never saw (``--holdout``) check the result: lower NLL, NIS closer to its degrees of freedom, and
velocity error against the SLAM-derived reference (not independent of the live estimator for the sysid bags; see
``ekf_replay``).

A likelihood can be raised by making the filter copy a sensor: with a tiny R on the wheel speed, the innovations
of that sensor vanish. Such solutions show up as a parameter on its bound and as a velocity error against the
reference that gets *worse*. ``--fix NAME[=VARIANCE]`` holds a parameter (at its default, or at VARIANCE) and
``--floor NAME=VARIANCE`` sets a lower bound; names are as in the parameter table.

    f1tenth_parallel_sim/.venv/bin/python misc/ekf_tune.py sysid_bags/sysid_* --holdout figure_8 free_drive_3 \\
        --fix 'R_vesc[vx]' 'R_imu[yaw]' 'R_imu[yaw_rate]' --out local.json

Bags are parsed once and cached (``--cache-dir``). Prints the tuned ``defaults`` values to paste.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekf_replay import Engine, FilterConfig, read_bag, reference_velocity, replay, summarize  # noqa: E402
from state_estimation.ekf_node import defaults  # noqa: E402

BAD = 1e9  # the cost of a candidate whose filter diverged
PARAMETERS = (
    [f'Q[{n}]' for n in ('x', 'y', 'yaw', 'vx', 'vy', 'yaw_rate', 'steer')]
    + [f'R_imu[{n}]' for n in ('yaw', 'yaw_rate', 'ax', 'ay')]
    + ['R_vesc[vx]', 'R_vesc[yaw_rate]']
)
N_VARIANCES = len(PARAMETERS)
BIAS_SCALE = 0.5  # m/s^2: one CMA-ES unit of accelerometer bias


def nominal_vector():
    """log10 of the local filter's defaults, in ``PARAMETERS`` order."""
    n = defaults.NOISE
    return np.log10(np.array(list(n['Q']) + list(n['R_imu']) + [n['R_vesc'][3], n['R_vesc'][5]]))


def to_config(theta, fit_bias):
    """The local filter's ``FilterConfig`` for a parameter vector (log10 variances, then, optionally, the bias offsets)."""
    v = 10.0 ** np.asarray(theta[:N_VARIANCES])
    config = FilterConfig()
    config.Q = np.diag(v[:7])
    config.R_imu = np.diag(v[7:11])
    r_vesc = np.array(defaults.NOISE['R_vesc'], dtype=float)
    r_vesc[3], r_vesc[5] = v[11], v[12]
    config.R_vesc = np.diag(r_vesc)
    if fit_bias:
        config.accel_bias = np.array(defaults.NOISE['imu_accel_bias']) + BIAS_SCALE * np.asarray(theta[N_VARIANCES:N_VARIANCES + 2])
    return config


def parse_constraints(fix, floor, names=PARAMETERS, nominal=None):
    """``({index: log10 variance}, {index: log10 floor})`` from ``--fix NAME[=V]`` and ``--floor NAME=V`` arguments."""
    nominal = nominal_vector() if nominal is None else nominal

    def index(name):
        if name not in names:
            raise SystemExit(f'unknown parameter {name!r}; choose from {", ".join(names)}')
        return names.index(name)

    fixed, floors = {}, {}
    for item in fix or []:
        name, _, value = item.partition('=')
        i = index(name)
        fixed[i] = np.log10(float(value)) if value else nominal[i]
    for item in floor or []:
        name, _, value = item.partition('=')
        if not value:
            raise SystemExit(f'--floor needs NAME=VARIANCE, got {item!r}')
        floors[index(name)] = np.log10(float(value))
    return fixed, floors


def load_bag(path, cache_dir):
    """One bag's replay inputs and reference velocity, parsed once and cached next to its modification time."""
    path = Path(path)
    cache = Path(cache_dir) / f'{path.name}.{int(path.stat().st_mtime)}.npz'
    keys = ('imu', 'cmd', 'odom', 'vesc', 'live')
    if cache.exists():
        with np.load(cache) as z:
            bag = {k: z[k] for k in keys}
            bag['reference'] = tuple(z['reference'])
        return bag
    bag = read_bag(path)
    bag['reference'] = reference_velocity(bag['odom'])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, reference=np.array(bag['reference']), **{k: bag[k] for k in keys})
    return bag


# --- worker processes: each prepares the training bags once and compiles its own filter ---------------------------
_replays: list = []
_spec: dict = {}


def _init_worker(paths, spec):
    import ekf_fast

    global _replays, _spec
    _spec = spec
    engine = Engine()
    fast = ekf_fast.FastFilter(engine)
    _replays = [ekf_fast.FastReplay(load_bag(p, spec['cache_dir']), FilterConfig(), engine, fast) for p in paths]


def _score(task):
    """(summed NLL, samples) of one candidate on one training bag."""
    index, theta = task
    try:
        result = _replays[index].run(to_config(theta, _spec['fit_bias']))
    except RuntimeError:
        return BAD, 1
    nll = sum(result['nll'].values())
    return (nll if np.isfinite(nll) else BAD), max(sum(result['count'].values()), 1)


def check_fast_path(path, spec, engine):
    """The compiled evaluator must reproduce ``replay`` before the search is allowed to rely on it."""
    import ekf_fast

    bag = load_bag(path, spec['cache_dir'])
    config = to_config(nominal_vector() + 0.3, False)
    result = ekf_fast.FastReplay(bag, config, engine, ekf_fast.FastFilter(engine)).run(config)
    _, stats = replay(bag, config, engine)
    reference, nis, counts = stats['nll'], {k: float(np.sum(v)) for k, v in stats['nis'].items()}, stats['count']
    for sensor in reference:
        for name, a, b in (('NLL', result['nll'][sensor], reference[sensor]), ('NIS', result['nis_sum'][sensor], nis[sensor])):
            if abs(a - b) > 1e-4 * abs(b):
                raise SystemExit(f'the fast evaluator disagrees with ekf_replay on the {name} of {Path(path).name} '
                                 f'({sensor}: {a} vs {b})')
        if result['count'][sensor] != counts[sensor]:
            raise SystemExit(f'the fast evaluator counts {result["count"][sensor]} {sensor} updates, ekf_replay {counts[sensor]}')


def evaluate(paths, config, cache_dir, engine, label):
    """Prints the report for ``config`` on ``paths``; returns pooled (NLL/sample, rmse)."""
    errors, nll, count = [], 0.0, 0
    for path in paths:
        bag = load_bag(path, cache_dir)
        rows, stats = replay(bag, config, engine)
        summarize(f'{label} {Path(path).name.replace("sysid_", "")}', rows, stats)
        errors.append(rows[:, 1:4] - rows[:, 4:7])
        nll += sum(stats['nll'].values())
        count += sum(stats['count'].values())
    rmse = np.sqrt((np.vstack(errors) ** 2).mean(0))
    print(f'{label} POOLED  NLL/sample {nll / count:.3f}  rmse vx {rmse[0]:.3f} vy {rmse[1]:.3f} wz {rmse[2]:.3f}')
    return nll / count, rmse


def search(pool, n_train, free, template, lo, hi, args, seed):
    """One CMA-ES run over the free parameters; returns (best full vector, its cost)."""
    import cma

    def full(theta_free):
        theta = template.copy()
        theta[free] = theta_free
        return theta

    es = cma.CMAEvolutionStrategy(template[free], args.sigma0,
                                  {'popsize': args.popsize, 'bounds': [list(lo[free]), list(hi[free])],
                                   'seed': seed, 'verbose': -9})
    best, best_cost = template, np.inf
    start = time.time()
    last_gain = (0, np.inf)  # (generation, cost) of the last improvement larger than the tolerance
    for generation in range(args.max_generations):
        candidates = es.ask()
        tasks = [(i, full(theta)) for theta in candidates for i in range(n_train)]
        scores = pool.map(_score, tasks, chunksize=max(1, len(tasks) // (4 * args.workers)))
        costs = []
        for k in range(len(candidates)):
            part = scores[k * n_train:(k + 1) * n_train]
            costs.append(sum(s[0] for s in part) / sum(s[1] for s in part))
        es.tell(candidates, costs)
        if min(costs) < best_cost:
            best_cost, best = min(costs), full(candidates[int(np.argmin(costs))])
        if generation % 10 == 9 or generation == 0:
            print(f'  seed {seed} gen {generation + 1:4d}  best NLL/sample {best_cost:.4f}  step {es.sigma:.3f}  '
                  f'{time.time() - start:.0f}s', flush=True)
        if best_cost < last_gain[1] - args.stall_tol:
            last_gain = (generation, best_cost)
        if generation - last_gain[0] >= args.stall_generations:
            print(f'  seed {seed} gen {generation + 1:4d}  stalled at {best_cost:.4f}', flush=True)
            break
        if es.stop():
            break
    return best, best_cost


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bags', nargs='+', help='bag directories')
    ap.add_argument('--holdout', nargs='*', default=[], metavar='SUBSTRING',
                    help='bags whose path contains one of these are kept out of the fit and used to check it')
    ap.add_argument('--fix', nargs='*', metavar='NAME[=VARIANCE]', help='hold these parameters (default value unless given)')
    ap.add_argument('--floor', nargs='*', metavar='NAME=VARIANCE', help='lower bound for these parameters')
    ap.add_argument('--decades', type=float, default=2.0, help='search each variance within this many decades of the default')
    ap.add_argument('--sigma0', type=float, default=0.5, help='initial CMA-ES step, in decades')
    ap.add_argument('--popsize', type=int, default=32)
    ap.add_argument('--max-generations', type=int, default=300)
    ap.add_argument('--restarts', type=int, default=3, help='independent searches from different seeds; the best wins')
    ap.add_argument('--fit-bias', action='store_true', help='also fit the accelerometer bias')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help='worker processes; the default is the number of physical cores assuming 2 threads per core '
                    '(more workers slow every task down once the cores are shared)')
    ap.add_argument('--stall-generations', type=int, default=25,
                    help='stop a search when it has not improved by --stall-tol in this many generations')
    ap.add_argument('--stall-tol', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cache-dir', default=str(Path.home() / '.cache' / 'ekf_tune'))
    ap.add_argument('--out', help='write the result as JSON')
    ap.add_argument('--evaluate-only', action='store_true', help='report the defaults on every bag and stop')
    args = ap.parse_args()
    args.bags = [b for b in args.bags if Path(b).is_dir()]  # a shell glob also matches the .metadata.yaml files
    if not args.bags:
        ap.error('no bag directories given')

    names = PARAMETERS
    base = nominal_vector()
    held = [b for b in args.bags if any(s in b for s in args.holdout)]
    train = [b for b in args.bags if b not in held]
    if not train:
        ap.error('no training bags left after --holdout')
    fixed, floors = parse_constraints(args.fix, args.floor, names, base)
    print(f'training on {len(train)} bags, holding out {len(held)}')
    for b in args.bags:  # parse once here so the workers only read the cache
        load_bag(b, args.cache_dir)
    spec = {'cache_dir': args.cache_dir, 'fit_bias': args.fit_bias}

    engine = Engine()
    n_params = len(names) + (2 if args.fit_bias else 0)
    nominal_config = to_config(np.concatenate([base, np.zeros(2)]), args.fit_bias)
    print('\n== defaults ==')
    evaluate(train, nominal_config, args.cache_dir, engine, 'train')
    if held:
        evaluate(held, nominal_config, args.cache_dir, engine, 'hold ')
    if args.evaluate_only:
        return
    check_fast_path(train[0], spec, engine)

    template = np.concatenate([base, np.zeros(n_params - len(names))])
    lo = np.concatenate([base - args.decades, [-4.0] * (n_params - len(names))])
    hi = np.concatenate([base + args.decades, [4.0] * (n_params - len(names))])
    for i, value in fixed.items():
        template[i] = value
    for i, value in floors.items():
        lo[i] = max(lo[i], value)
        template[i] = max(template[i], value)
    free = np.array([i for i in range(n_params) if i not in fixed])
    if fixed:
        print('fixed:', ', '.join(f'{names[i]}={10 ** v:.4g}' for i, v in fixed.items()))
    if floors:
        print('floors:', ', '.join(f'{names[i]}>={10 ** v:.4g}' for i, v in floors.items()))

    pool = mp.get_context('spawn').Pool(min(args.workers, args.popsize * len(train)), initializer=_init_worker,
                                        initargs=(train, spec))
    best, best_cost = template, np.inf
    try:
        for restart in range(args.restarts):
            candidate, cost = search(pool, len(train), free, template, lo, hi, args, args.seed + restart)
            print(f'seed {args.seed + restart}: NLL/sample {cost:.4f}', flush=True)
            if cost < best_cost:
                best, best_cost = candidate, cost
    finally:
        pool.terminate()

    tuned = to_config(best, args.fit_bias)
    print('\n== tuned ==')
    evaluate(train, tuned, args.cache_dir, engine, 'train')
    if held:
        evaluate(held, tuned, args.cache_dir, engine, 'hold ')

    print('\nparameter           default        tuned   factor')
    for i, (name, d, t) in enumerate(zip(names, base, best[:len(names)])):
        note = '  (fixed)' if i in fixed else ''
        if i not in fixed:
            on_lo, on_hi = abs(t - lo[i]) < 0.02, abs(t - hi[i]) < 0.02
            note = '  <- on the lower bound' if on_lo else '  <- on the upper bound' if on_hi else ''
        print(f'{name:16s} {10 ** d:11.4g} {10 ** t:12.4g} {10 ** (t - d):8.2f}x{note}')
    noise = {
        'R_imu': [float(v) for v in np.diag(tuned.R_imu)],
        'R_vesc': [float(v) for v in np.diag(tuned.R_vesc)],
        'Q': [float(v) for v in np.diag(tuned.Q)],
        'imu_accel_bias': [float(v) for v in tuned.accel_bias],
    }
    print('\nstate_estimation/ekf_node/defaults.py NOISE entries:')
    for key, value in noise.items():
        print(f"    '{key}': [{', '.join(f'{v:.4g}' for v in value)}],")
    if args.out:
        Path(args.out).write_text(json.dumps({'noise': noise, 'train': train, 'holdout': held,
                                              'fixed': {names[i]: float(10 ** v) for i, v in fixed.items()},
                                              'nll_per_sample': best_cost}, indent=2))
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
