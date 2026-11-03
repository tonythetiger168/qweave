"""
QL7 gate: GPU-hours to reach the baseline's final score
=======================================================
Inputs: run.json files from ql7_grpo_train.py (arm = directory name minus
the trailing _s<seed>). For the baseline arm (default 'base'):

  target S* = mean over seeds of the baseline's final full-test score - delta
  H_base    = mean over baseline seeds of GPU-hours to first reach S*

For every other arm, H_arm is the same quantity (inf if a seed never gets
there). The eval curve is smoothed with a 3-point moving average before
the crossing is found (centred, no lag), and the crossing is linearly interpolated in GPU-hours.

PASS for an arm needs ALL of:
  * ratio = H_arm / H_base <= 0.75, with the bootstrap 95% CI upper bound
    (resampling seeds) <= 1.0, and
  * final score non-inferior: paired per-question bootstrap of
    (arm - base) final accuracy, 95% CI lower bound >= -noninf (2 pt).
Anything else is reported as a negative result (FermiQLLM's "-25% RL cost"
not reproduced by that lever).
"""

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, List

import torch


def arm_of(run: Dict, path: str) -> str:
    name = os.path.basename(os.path.dirname(path)) if path.endswith('run.json') else run['arm']
    return re.sub(r'_s\d+$', '', name)


def smooth(y: List[float], w=3) -> List[float]:
    """Centred moving average (no lag; edges average what exists)."""
    r = w // 2
    return [sum(y[max(0, i - r):i + r + 1]) / len(y[max(0, i - r):i + r + 1])
            for i in range(len(y))]


def hours_to(curve: List[Dict], target: float, w=3) -> float:
    pts = [c for c in curve if c.get('tag', 'curve') == 'curve']
    if not pts:
        return math.inf
    h = [c['gpu_hours'] for c in pts]
    y = smooth([c['score'] for c in pts], w)
    for i in range(len(y)):
        if y[i] >= target:
            if i == 0:
                return h[0]
            f = (target - y[i - 1]) / max(y[i] - y[i - 1], 1e-12)
            return h[i - 1] + f * (h[i] - h[i - 1])
    return math.inf


def paired_boot(a: List[float], b: List[float], n=2000, seed=0):
    """CI of mean(a) - mean(b) in points, a/b per-question 0/1 (same order)."""
    x = torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(x), (n, len(x)), generator=g)
    m = x[idx].mean(1) * 100
    return float(x.mean() * 100), float(m.quantile(0.025)), float(m.quantile(0.975))


def seed_boot_ratio(ha: List[float], hb: List[float], n=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.tensor(ha, dtype=torch.float64)
    B = torch.tensor(hb, dtype=torch.float64)
    ra = A[torch.randint(0, len(A), (n, len(A)), generator=g)].mean(1)
    rb = B[torch.randint(0, len(B), (n, len(B)), generator=g)].mean(1)
    r = ra / rb
    r = torch.where(torch.isnan(r), torch.full_like(r, math.inf), r)
    return float(r.quantile(0.025)), float(r.quantile(0.975))


def gate(runs: Dict[str, List[Dict]], base='base', delta=1.0, target_ratio=0.75,
         noninf=2.0) -> Dict:
    if base not in runs:
        raise SystemExit(f'baseline arm {base!r} missing; arms: {sorted(runs)}')
    bf = [r['final']['score'] for r in runs[base]]
    target = sum(bf) / len(bf) - delta
    # a baseline seed that never reaches S* is censored at its total GPU-hours
    # (favours the baseline); an arm seed that never reaches it counts as inf
    hb = [min(hours_to(r['curve'], target), r['final']['gpu_hours']) for r in runs[base]]
    Hb = sum(hb) / len(hb)
    base_correct = [sum(x) / len(x) for x in zip(*[r['final']['correct'] for r in runs[base]])]
    out = {'baseline': base, 'target_score': target, 'baseline_final_mean': target + delta,
           'baseline_hours_to_target': Hb, 'baseline_total_hours':
           sum(r['final']['gpu_hours'] for r in runs[base]) / len(runs[base]),
           'target_ratio': target_ratio, 'arms': {}}
    for arm, rs in sorted(runs.items()):
        ha = (hb if arm == base else [hours_to(r['curve'], target) for r in rs])
        Ha = sum(ha) / len(ha)
        ratio = Ha / Hb if Hb > 0 else (1.0 if Ha == 0 else math.inf)
        lo, hi = seed_boot_ratio(ha, hb) if all(map(math.isfinite, ha + hb)) else (math.inf,) * 2
        corr = [sum(x) / len(x) for x in zip(*[r['final']['correct'] for r in rs])]
        d, dlo, dhi = paired_boot(corr, base_correct)
        fr = defaultdict(float)
        for r in rs:
            for k, v in r.get('fractions', {}).items():
                fr[k] += v / len(rs)
        zero = [sum(x['zero_groups'] for x in r['log']) / max(1, sum(x['groups'] for x in r['log']))
                for r in rs]
        ok = (arm != base and ratio <= target_ratio and hi <= 1.0 and dlo >= -noninf)
        out['arms'][arm] = {
            'seeds': len(rs), 'hours_to_target': ha, 'ratio': ratio, 'ratio_ci95': [lo, hi],
            'final_mean': sum(r['final']['score'] for r in rs) / len(rs),
            'final_delta_pt': d, 'final_delta_ci95': [dlo, dhi],
            'zero_signal_frac': sum(zero) / len(zero), 'phase_fractions': dict(fr),
            'select_seconds': sum(sum(x.get('select_s') or 0 for x in r['log']) for r in rs) / len(rs),
            'pass': ok}
    out['pass_any'] = any(v['pass'] for v in out['arms'].values())
    out['baseline_persistence'] = persistence(
        runs[base], int(runs[base][0].get('args', {}).get('group', 8)))
    return out


def persistence(runs: List[Dict], G: int = 8) -> Dict:
    """Offline, from baseline logs only: when a prompt is revisited, how often
    is its group zero-signal again, given it was zero-signal last time? This
    bounds what prompt selection can save before paying for selection arms:
    skipping every prompt that was zero-signal last time saves `skip_frac` of
    revisit rollouts and wrongly skips `false_skip` of them."""
    zz = zn = nz = nn = 0
    for r in runs:
        last = {}
        for rec in r['log']:
            for i, s in zip(rec.get('prompt_idx', []), rec.get('prompt_succ', [])):
                z = s in (0, G)
                if i in last:
                    if last[i]:
                        zz += z
                        zn += (not z)
                    else:
                        nz += z
                        nn += (not z)
                last[i] = z
    tot = zz + zn + nz + nn
    if not tot:
        return {'revisits': 0}
    return {'revisits': tot, 'p_zero_given_zero': zz / max(1, zz + zn),
            'p_zero_given_signal': nz / max(1, nz + nn),
            'skip_frac': (zz + zn) / tot, 'false_skip': zn / tot}


def load(paths):
    runs = defaultdict(list)
    for p in paths:
        for f in (glob.glob(os.path.join(p, '**', 'run.json'), recursive=True)
                  if os.path.isdir(p) else [p]):
            r = json.load(open(f))
            if r.get('final'):
                runs[arm_of(r, f)].append(r)
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='run.json files or directories')
    ap.add_argument('--base', default='base')
    ap.add_argument('--delta', type=float, default=1.0)
    ap.add_argument('--target-ratio', type=float, default=0.75)
    ap.add_argument('--noninf', type=float, default=2.0)
    ap.add_argument('--out', default='')
    a = ap.parse_args(argv)
    res = gate(load(a.runs), a.base, a.delta, a.target_ratio, a.noninf)
    print(f"target {res['target_score']:.2f}  baseline {res['baseline_hours_to_target']:.3f} GPU-h")
    for arm, v in res['arms'].items():
        print(f"{arm:16s} ratio {v['ratio']:.3f} CI [{v['ratio_ci95'][0]:.2f}, "
              f"{v['ratio_ci95'][1]:.2f}]  final {v['final_mean']:.1f} "
              f"({v['final_delta_pt']:+.2f} [{v['final_delta_ci95'][0]:+.2f}, "
              f"{v['final_delta_ci95'][1]:+.2f}])  zero-signal {v['zero_signal_frac']:.2f}  "
              f"{'PASS' if v['pass'] else '-'}")
    print('baseline persistence:', json.dumps(res['baseline_persistence']))
    print('QL7 gate:', 'PASS' if res['pass_any'] else 'FAIL')
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
    return res


if __name__ == '__main__':
    main()
