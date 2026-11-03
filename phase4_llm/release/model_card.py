"""
DiracLM model card (Hugging Face README.md)
===========================================
Every number on the card comes from a report file; components that did not
pass their gate are listed as excluded with the measured reason, and the
card is marked "not a release candidate" when the release gate fails.
Pure Python, runs in CI.
"""

from typing import Dict, Optional

from phase4_llm.release.recipe import COMPONENTS, PROTOCOL_TASKS, TASK_LABEL, Recipe, Resolution

FERMI_CLAIMS = {
    "math500": "+10-20 % (no baseline, no per-item output)",
    "gpqa_diamond": "+10-20 % (no baseline, no per-item output)",
    "bbh": "+10-20 % (no baseline, no per-item output)",
}


def _front_matter(r: Recipe) -> str:
    tags = ["diraclm", "qweave", "quantum-inspired", "tensor-network", "sparse-attention", "grpo"]
    return "\n".join(["---", f"license: {r.license.lower()}", f"base_model: {r.base_model}",
                      "library_name: transformers", "pipeline_tag: text-generation",
                      "tags:"] + [f"- {t}" for t in tags] + ["---", ""])


def render(r: Recipe, res: Resolution, gate: Dict, comparison: Optional[Dict] = None,
           summaries: Optional[Dict[str, Dict]] = None, repro: str = "") -> str:
    L = [_front_matter(r), f"# {r.name} (v{r.version})", ""]
    if not gate.get("ready"):
        L += ["> **Not a release candidate.** The release gate did not pass:"]
        L += [f"> - {x}" for x in gate.get("reasons", [])] + [""]
    L += [f"{r.name} is a quantum-inspired language model from the "
          "[qweave](https://github.com/tonythetiger168/qweave) project, built on "
          f"[{r.base_model}](https://huggingface.co/{r.base_model}).", "",
          "**Quantum-inspired** means classical methods borrowed from quantum physics "
          "(tensor networks, entanglement diagnostics, simulated quantum annealing) running "
          "on ordinary GPUs. No quantum hardware is used and we make no quantum-advantage claim.",
          "", "## What is in this model", "",
          "| Component | Status | Measured on this base |", "|---|---|---|"]
    for name, desc in COMPONENTS.items():
        if name in res.included:
            L.append(f"| {desc} | included | {res.included[name]['detail']} |")
        elif name in res.excluded:
            v = res.excluded[name]
            L.append(f"| {desc} | not included ({v['verdict'].lower()}) | {v['detail']} |")
    L.append("")

    tasks = (comparison or {}).get("tasks", {})
    if tasks:
        L += ["## Reasoning benchmarks vs the plain base", "",
              "Same prompts, sampling and grader for both models; difference is paired over "
              "items with a 95 % bootstrap CI. Per-item results are in `eval/` (GPQA as ids and "
              "scores only, at the dataset authors' request).", "",
              f"| Benchmark | {r.controls.get('plain_base', r.base_model)} | {r.name} | "
              "Difference (pt) | 95 % CI | FermiQLLM 1.0 claim |",
              "|---|---|---|---|---|---|"]
        for t in PROTOCOL_TASKS:
            if t in tasks:
                x = tasks[t]
                L.append(f"| {TASK_LABEL[t]} | {x['a_pct']:.1f} | {x['b_pct']:.1f} | "
                         f"{x['diff_pt']:+.2f} | [{x['ci_lo']:+.2f}, {x['ci_hi']:+.2f}] | "
                         f"{FERMI_CLAIMS[t]} |")
        L += [""]
        if gate.get("claims"):
            L += ["What we claim from this table:", ""]
            L += [f"- {TASK_LABEL[t]}: {c}" for t, c in gate["claims"].items()] + [""]
    if summaries:
        L += ["## Generation settings", ""]
        for who, s in summaries.items():
            L.append(f"- {who}: backend {s.get('backend')}, thinking {s.get('thinking')}, "
                     f"max {s.get('max_tokens')} new tokens, seed {s.get('seed')}")
        L += [""]

    L += ["## Usage", "", "```python",
          "# dense attention, no extra code needed",
          "from transformers import AutoModelForCausalLM, AutoTokenizer",
          f'tok = AutoTokenizer.from_pretrained("{r.hf_repo}")',
          f'model = AutoModelForCausalLM.from_pretrained("{r.hf_repo}", torch_dtype="bfloat16")',
          "", "# with QSA long-context prefill",
          "# pip install git+https://github.com/tonythetiger168/qweave",
          "from phase4_llm.release.loader import load_diraclm",
          f'model, tok = load_diraclm("{r.hf_repo}")', "```", "",
          "## Limitations", "",
          "- QSA only changes prompts of at least `min_len` tokens (see `diraclm_config.json`); "
          "short-prompt benchmarks such as the ones above run the same dense attention as the base.",
          "- Benchmarks were run once per protocol with k samples per item; they are not "
          "leaderboard submissions.",
          "- Results on other tasks, languages and safety behaviour are inherited from the base "
          "model and were not re-measured.", "",
          "## Reproduce", "", "```bash", repro or "# see docs/QL8_RELEASE_PLAN.md in qweave",
          "```", "",
          "## License", "",
          f"{r.license}. Built on {r.base_model} ({r.base_license}); the base model's license "
          "and notices are included in this repository.", ""]
    return "\n".join(L)
