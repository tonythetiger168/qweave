"""
Gate verdicts for DiracLM components
====================================
A component goes into a DiracLM release only if a measured report says it
passed its gate *on that release's base model*. This module reads the
reports the existing phases already write and returns a uniform verdict:

    {"verdict": "PASS" | "FAIL" | "MISSING", "detail": str, "numbers": {...}}

  qsa    phase4_llm/bench_ql3.py -> ql3_bench.json
         PASS: kernel speedup >= 1.2x at 16K AND long-window PPL <= +1 %
         for the shipped top-k (QL3 gate, unchanged)
  ql7    phase4_llm/ql7_gate.py --out gate.json
         PASS: the named arm passed (<= 0.75x GPU-hours to the baseline's
         score, CI upper <= 1.0, non-inferior final score)
  ql5    any report with {"verdict": "PASS"|"FAIL"} or {"pass": bool}
         (the QL5 rework thread defines its own gate file)

Pure Python, runs in CI.
"""

import json
import os
from typing import Dict, Optional

MISSING = {"verdict": "MISSING", "detail": "no report", "numbers": {}}


def _load(path: Optional[str]) -> Optional[Dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def qsa_verdict(report: Dict, topk: int = 16, n_ctx: int = 16384,
                min_speedup: float = 1.2, max_ppl_pct: float = 1.0) -> Dict:
    """QL3 gate for the shipped top-k. bench_ql3.py writes `kernel` (top-k =
    --topk) and `kernel_2x` (2 * --topk); quality modes `qsa_uniform` and
    `qsa_uniform_2x` match them."""
    kern = [r for r in report.get("kernel", []) + report.get("kernel_2x", [])
            if r.get("topk") == topk and r.get("N") == n_ctx]
    base_topk = min((r.get("topk") for r in report.get("kernel", [])), default=None)
    mode = "qsa_uniform" if topk == base_topk else "qsa_uniform_2x"
    qual = [q for q in report.get("quality", []) if q.get("mode") == mode]
    if not kern or not qual:
        return {"verdict": "MISSING", "numbers": {},
                "detail": f"no kernel/quality rows for top-{topk} at N={n_ctx}"}
    sp, dp = kern[0]["speedup"], qual[0]["ppl_rel_pct"]
    ok = sp >= min_speedup and dp <= max_ppl_pct
    return {"verdict": "PASS" if ok else "FAIL",
            "numbers": {"model": report.get("model"), "gpu": report.get("gpu"),
                        "topk": topk, "n_ctx": n_ctx, "speedup": sp, "ppl_rel_pct": dp},
            "detail": f"top-{topk} at {n_ctx // 1024}K: {sp:.2f}x attention, "
                      f"PPL {dp:+.2f}% (gate >= {min_speedup}x, <= {max_ppl_pct}%)"}


def ql7_verdict(report: Dict, arm: Optional[str] = None) -> Dict:
    arms = report.get("arms", {})
    if arm is None:
        passing = sorted((v.get("ratio", 9e9), k) for k, v in arms.items() if v.get("pass"))
        arm = passing[0][1] if passing else None
    if arm is None or arm not in arms:
        return {"verdict": "FAIL" if arms else "MISSING", "numbers": {},
                "detail": "no arm passed the QL7 gate" if arms else "no arms in report"}
    v = arms[arm]
    nums = {k: v[k] for k in ("ratio", "ratio_ci95", "final_mean", "final_delta_pt",
                                "final_delta_ci95") if k in v}
    nums["arm"] = arm
    return {"verdict": "PASS" if v.get("pass") else "FAIL", "numbers": nums,
            "detail": f"arm {arm}: GPU-hours ratio {v.get('ratio')}"}


def generic_verdict(report: Dict) -> Dict:
    if "verdict" in report:
        ok = str(report["verdict"]).upper() == "PASS"
    elif "pass" in report:
        ok = bool(report["pass"])
    else:
        return {"verdict": "MISSING", "numbers": {}, "detail": "report has no verdict field"}
    return {"verdict": "PASS" if ok else "FAIL", "numbers": report.get("numbers", {}),
            "detail": report.get("detail", "")}


def verdict(component: str, report_path: Optional[str], config: Optional[Dict] = None,
            base_model: Optional[str] = None) -> Dict:
    rep = _load(report_path)
    if rep is None:
        return dict(MISSING, detail=f"no report at {report_path}")
    cfg = config or {}
    if component == "qsa":
        v = qsa_verdict(rep, topk=cfg.get("topk", 16))
    elif component == "ql7":
        v = ql7_verdict(rep, cfg.get("arm"))
    else:
        v = generic_verdict(rep)
    # a gate measured on another model does not count for this release
    measured_on = rep.get("model") or rep.get("base_model") or cfg.get("measured_on")
    if base_model and measured_on and measured_on != base_model and v["verdict"] == "PASS":
        v = dict(v, verdict="FAIL",
                 detail=f"measured on {measured_on}, release base is {base_model}; "
                        f"re-run the gate on the release base ({v['detail']})")
    return v
