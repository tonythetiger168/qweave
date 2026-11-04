"""
DiracLM release recipe and release gate
=======================================
A recipe (JSON, see recipes/) names the base model and the candidate
components. `resolve` keeps a component only if its gate report says PASS
on that base; `release_gate` then decides whether the resulting model may
be published, from the paired comparison against the plain base.

Release gate (frozen, docs/QL8_RELEASE_PLAN.md section 4):
  * at least one component included (otherwise there is nothing to release)
  * every benchmark in the protocol (MATH-500, GPQA-Diamond, BBH) is
    non-inferior to the plain base: paired-bootstrap 95% CI lower bound
    >= -noninf (2 pt)
  * a benchmark may be called "better than base" only if its CI lower
    bound is > 0; the card states every other result as "within X pt"

Pure Python, runs in CI.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from phase4_llm.release import gates

PROTOCOL_TASKS = ("math500", "gpqa_diamond", "bbh")
TASK_LABEL = {"math500": "MATH-500", "gpqa_diamond": "GPQA-Diamond", "bbh": "BBH"}
COMPONENTS = {
    "qsa": "QL3 quantum-inspired sparse attention (long-context prefill)",
    "ql7": "QL7 QUBO prompt selection for GRPO (RL cost)",
    "ql5": "QL5 activation-aware tensor-network compression (parameters)",
}


@dataclass
class Component:
    name: str
    enabled: bool = True
    gate_report: Optional[str] = None
    config: Dict = field(default_factory=dict)


@dataclass
class Recipe:
    name: str
    version: str
    base_model: str
    base_license: str
    license: str
    components: List[Component]
    hf_repo: str = ""
    notes: str = ""
    controls: Dict[str, str] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        return f"{self.name}-v{self.version}"


def load_recipe(path: str) -> Recipe:
    with open(path) as f:
        d = json.load(f)
    comps = [Component(name=k, **v) for k, v in d.pop("components", {}).items()]
    unknown = [c.name for c in comps if c.name not in COMPONENTS]
    if unknown:
        raise ValueError(f"unknown components {unknown}; known: {sorted(COMPONENTS)}")
    return Recipe(components=comps, **d)


@dataclass
class Resolution:
    included: Dict[str, Dict]
    excluded: Dict[str, Dict]


def resolve(recipe: Recipe, root: str = ".") -> Resolution:
    inc, exc = {}, {}
    for c in recipe.components:
        if not c.enabled:
            exc[c.name] = {"verdict": "DISABLED", "detail": "disabled in recipe", "numbers": {}}
            continue
        path = os.path.join(root, c.gate_report) if c.gate_report else None
        v = gates.verdict(c.name, path, c.config, recipe.base_model)
        if v["verdict"] == "MISSING" and c.gate_report:
            v = dict(v, detail=f"no gate report yet ({c.gate_report})")
        (inc if v["verdict"] == "PASS" else exc)[c.name] = dict(v, config=c.config)
    return Resolution(inc, exc)


def release_gate(res: Resolution, comparison: Optional[Dict], noninf: float = 2.0) -> Dict:
    """comparison = reasoning_eval.compare_dirs(base, model) output."""
    reasons, claims = [], {}
    if not res.included:
        reasons.append("no component passed its gate on this base")
    tasks = (comparison or {}).get("tasks", {})
    for t in PROTOCOL_TASKS:
        r = tasks.get(t)
        if r is None:
            reasons.append(f"{TASK_LABEL[t]}: not evaluated")
            continue
        if r["ci_lo"] < -noninf:
            reasons.append(f"{TASK_LABEL[t]}: {r['diff_pt']:+.2f} pt, CI lower "
                           f"{r['ci_lo']:+.2f} < -{noninf}")
        if r["ci_lo"] > 0:
            claims[t] = f"better than base by {r['diff_pt']:+.2f} pt [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"
        else:
            bound = max(abs(r["ci_lo"]), abs(r["ci_hi"]))
            claims[t] = (f"within {bound:.1f} pt of base ({r['diff_pt']:+.2f} pt "
                         f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}])")
    return {"ready": not reasons, "reasons": reasons, "claims": claims}
