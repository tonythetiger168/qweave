"""
QL5 acceptance gate: compressed vs dense on MMLU and GSM8K
==========================================================
Gate (docs/FERMIQLLM_SURPASS_PLAN.md): at >= 20 % total parameter saving,
MMLU drop <= 2 pt and GSM8K drop <= 5 pt vs the uncompressed Base model.

Reads two run_eval.py reports produced with --limit 0 --log-samples and
computes, per benchmark, the score difference with a 95 % PAIRED bootstrap
confidence interval (same examples resampled for both models; MMLU pooled
over all 57 subjects, i.e. micro-average). The gate passes on the point
estimate; the report also says whether it still passes at the CI bound,
which is the claim we can make in public (surpass plan E2).

  python phase4_llm/ql5_gate.py --dense ql4_dense.json \\
      --compressed ql4_compressed.json [--ql5-report ql5_report.json]
"""

import argparse, json, os, sys
import numpy as np

GATES = {'mmlu': 2.0, 'gsm8k': 5.0}       # max allowed drop, points
MIN_SAVING = 20.0                         # % of total params


def _load_samples(report_path):
    r = json.load(open(report_path))
    sf = r.get('samples_file')
    if not sf:
        sys.exit(f"{report_path}: no per-sample file; rerun run_eval.py with --log-samples")
    return r, json.load(open(os.path.join(os.path.dirname(report_path), sf)))


def paired(a: dict, b: dict, prefix: str):
    """Aligned 0/1 score vectors over every (subtask, doc) under `prefix`."""
    xa, xb = [], []
    for t in sorted(a):
        if t == prefix or t.startswith(prefix + '_'):
            common = sorted(set(a[t]) & set(b.get(t, {})))
            xa += [a[t][d] for d in common]
            xb += [b[t][d] for d in common]
    return np.array(xa), np.array(xb)


def bootstrap_diff(xa, xb, n=10_000, seed=0):
    """(b - a) mean in points with a 95 % paired percentile CI."""
    rng = np.random.default_rng(seed)
    d = (xb - xa) * 100.0
    bs = np.concatenate([d[rng.integers(0, len(d), size=(min(500, n - i), len(d)))].mean(1)
                         for i in range(0, n, 500)])
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def gate(dense_samples, comp_samples, saving_pct=None, gates=GATES):
    out, ok_point, ok_ci = {}, True, True
    for task, max_drop in gates.items():
        xa, xb = paired(dense_samples, comp_samples, task)
        if len(xa) == 0:
            out[task] = {'error': 'no paired samples'}
            ok_point = ok_ci = False
            continue
        m, lo, hi = bootstrap_diff(xa, xb)
        p = m >= -max_drop
        c = lo >= -max_drop
        out[task] = {'n': int(len(xa)), 'dense': round(100 * xa.mean(), 2),
                     'compressed': round(100 * xb.mean(), 2),
                     'delta': round(m, 2), 'ci95': [round(lo, 2), round(hi, 2)],
                     'max_drop': max_drop, 'pass_point': bool(p), 'pass_ci': bool(c)}
        ok_point &= p; ok_ci &= c
    if saving_pct is not None:
        s = saving_pct >= MIN_SAVING
        out['param_saving_pct'] = saving_pct
        ok_point &= s; ok_ci &= s
    out['PASS'] = bool(ok_point)
    out['PASS_at_CI_bound'] = bool(ok_ci)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dense', required=True, help='ql4_dense.json')
    ap.add_argument('--compressed', required=True, help='ql4_compressed.json')
    ap.add_argument('--ql5-report', default=None, help='ql5_report.json (param saving)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    rd, sd = _load_samples(args.dense)
    rc, sc = _load_samples(args.compressed)
    saving = None
    if args.ql5_report:
        saving = json.load(open(args.ql5_report))['param_saving_pct']
    elif rd.get('params') and rc.get('params'):
        saving = round(100 * (1 - rc['params'] / rd['params']), 2)
    res = gate(sd, sc, saving)
    print(json.dumps(res, indent=2))
    for t in GATES:
        r = res[t]
        if 'delta' in r:
            print(f"{t:6s} dense {r['dense']:5.1f}  compressed {r['compressed']:5.1f}  "
                  f"delta {r['delta']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]  "
                  f"(allowed -{r['max_drop']})")
    print("QL5 ACCEPTANCE:", "PASS" if res['PASS'] else "FAIL",
          "| at 95% CI bound:", "PASS" if res['PASS_at_CI_bound'] else "FAIL")
    if args.out:
        json.dump(res, open(args.out, 'w'), indent=2)


if __name__ == '__main__':
    main()
