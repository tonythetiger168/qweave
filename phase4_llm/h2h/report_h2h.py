"""
Head-to-head report: every model vs a named same-size baseline
==============================================================
Reads the JSONs written by eval_h2h.py and prints absolute and relative
gains with paired bootstrap 95 % CIs (questions resampled, same questions
for both models). FermiQLLM 1.0 claims +10-20 % (relative) over same-size
open-source models on MATH-500 / GPQA-Diamond / BBH.

  python phase4_llm/h2h/report_h2h.py --dir /workspace/h2h --baseline qwen3-0.6b-nothink
"""

import argparse, glob, json, os, random


def paired_ci(a, b, iters=4000, seed=0):
    """95 % CI of mean(b) - mean(a) in points, a/b are 0/1 lists on the same items."""
    n = len(a)
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(100 * sum(b[i] - a[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/h2h")
    ap.add_argument("--baseline", required=True)
    a = ap.parse_args()
    res = {}
    for f in sorted(glob.glob(os.path.join(a.dir, "*.json"))):
        d = json.load(open(f))
        if "benches" in d:
            res[d["name"]] = d
    base = res[a.baseline]
    benches = ["math500", "gpqa", "bbh"]
    lines = ["| model | mode | " + " | ".join(
        f"{b} acc | Δ pt [95 % CI] | rel. gain" for b in benches) + " |",
        "|---|---|" + "---|---|---|" * len(benches)]
    summary = {}
    for name, d in res.items():
        row = [name, d["mode"]]
        summary[name] = {}
        for b in benches:
            r, rb = d["benches"].get(b), base["benches"].get(b)
            if not r or not rb:
                row += ["—", "—", "—"]; continue
            m = dict(zip(r["ids"], r["correct"])); mb = dict(zip(rb["ids"], rb["correct"]))
            ids = [i for i in rb["ids"] if i in m]
            x = [mb[i] for i in ids]; y = [m[i] for i in ids]
            acc, accb = 100 * sum(y) / len(ids), 100 * sum(x) / len(ids)
            if name == a.baseline:
                row += [f"{acc:.1f}", "baseline", "—"]
            else:
                lo, hi = paired_ci(x, y)
                rel = 100 * (acc - accb) / accb if accb else float("nan")
                row += [f"{acc:.1f}", f"{acc - accb:+.1f} [{lo:+.1f}, {hi:+.1f}]", f"{rel:+.1f} %"]
                summary[name][b] = {"acc": acc, "delta": acc - accb, "ci": [lo, hi], "rel_pct": rel}
            summary[name].setdefault("acc", {})[b] = acc
        lines.append("| " + " | ".join(row) + " |")
    md = "\n".join(lines)
    print(md)
    open(os.path.join(a.dir, "h2h_table.md"), "w").write(md + "\n")
    json.dump({"baseline": a.baseline, "summary": summary},
              open(os.path.join(a.dir, "h2h_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
