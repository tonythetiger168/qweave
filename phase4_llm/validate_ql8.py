"""
QL8 (DiracLM release) validation -- CPU, no downloads, runs in CI
=================================================================
  [1] <think> stripping, truncated replies score 0
  [2] \\boxed{} extraction with nested braces
  [3] MATH answer equivalence (normaliser + numeric fallback)
  [4] GPQA: option shuffle is deterministic and tracks the gold letter;
      letter extraction
  [5] BBH: answer extraction and grading (multiple choice, words, numbers)
  [6] per-item scoring, summary, GPQA rows keep no question text
  [7] paired bootstrap: zero difference, known shift, CI coverage
  [8] QSA gate from the real QL3 Qwen3-0.6B report: top-16 PASS, top-8
      FAIL, and a gate measured on another base does not count
  [9] QL7 gate verdict picks the passing arm
 [10] recipe -> resolve -> release gate -> model card -> package build;
      push refuses without approval and refuses public when not ready
 [11] QSA HF attention: decode against a KV cache equals dense attention
      (bottom-right causal), prefill below min_len equals SDPA causal
"""

import json
import os
import random
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from phase4_llm.release import gates  # noqa: E402
from phase4_llm.release import model_card, package  # noqa: E402
from phase4_llm.release import reasoning_eval as E  # noqa: E402
from phase4_llm.release import recipe as R  # noqa: E402

PASS = []


def check(name, cond, info=''):
    PASS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}", flush=True)


# [1]
check("1 strip_think",
      E.strip_think("<think>a \\boxed{1}</think>The answer is \\boxed{2}").strip()
      == "The answer is \\boxed{2}"
      and E.strip_think("<think>never closed \\boxed{3}") is None
      and E.strip_think("reasoning</think>final") == "final"
      and E.grade("math500", "<think>\\boxed{5}", "5")["truncated"])

# [2]
check("2 last_boxed",
      E.last_boxed("x \\boxed{1} then \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
      and E.last_boxed("\\boxed{\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}}")
      == "\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}"
      and E.last_boxed("no box") is None
      and E.last_boxed("\\fbox{7}") == "7")

# [3]
same = [("\\frac12", "\\frac{1}{2}"), ("\\dfrac{3}{4}", "\\frac{3}{4}"), ("0.5", "\\frac{1}{2}"),
        ("1/2", "\\frac{1}{2}"), ("x = 5", "5"), ("5.", "5"), ("10\\%", "10"),
        ("90^\\circ", "90"), ("\\$1,000", "1000"), ("\\text{(C)}", "(C)"),
        ("\\sqrt3", "\\sqrt{3}"), ("\\left( 3, 4 \\right)", "(3,4)"), ("2.50", "2.5"),
        ("12 \\text{ inches}", "12"), ("-\\frac{2}{4}", "-0.5")]
diff = [("3x", "3"), ("\\frac{1}{3}", "0.33"), ("(3,4)", "(4,3)"), ("2", "-2"),
        ("\\sqrt{2}", "2"), (None, "1")]
bad_same = [p for p in same if not E.math_equal(*p)]
bad_diff = [p for p in diff if E.math_equal(*p)]
check("3 math_equal", not bad_same and not bad_diff, f"wrong: {bad_same + bad_diff}")

# [4]
row = {"Question": "Q?", "Correct Answer": "right", "Incorrect Answer 1": "w1",
       "Incorrect Answer 2": "w2", "Incorrect Answer 3": "w3"}
a, b = E.gpqa_item(row, 7), E.gpqa_item(row, 7)
golds = {E.gpqa_item(row, i)["gold"] for i in range(40)}
check("4 GPQA shuffle + extraction",
      a == b and f"({a['gold']}) right" in a["prompt"] and golds == set("ABCD")
      and not a["public"]
      and E.extract_choice("I think (B) is wrong... Answer: C") == "C"
      and E.extract_choice("so \\boxed{D}") == "D"
      and E.extract_choice("**Answer:** A") == "A"
      and E.extract_choice("Answer: E") is None,
      f"gold letters over 40 items: {sorted(golds)}")

# [5]
check("5 BBH extract + grade",
      E.bbh_equal(E.extract_bbh("...So the answer is (B)."), "(B)")
      and not E.bbh_equal(E.extract_bbh("So the answer is (C)."), "(B)")
      and E.bbh_equal(E.extract_bbh("So the answer is True."), "True")
      and E.bbh_equal(E.extract_bbh("So the answer is 24."), "24")
      and E.bbh_equal(E.extract_bbh("So the answer is valid"), "valid")
      and E.bbh_equal(E.extract_bbh("So the answer is ] ] )."), "] ] )")
      and E.bbh_equal(E.extract_bbh("the answer is: apple banana cherry"), "apple banana cherry")
      and not E.bbh_equal(None, "Yes")
      and len(E.BBH_SUBTASKS) == 27)

# [6]
items = [{"id": "math500/a", "prompt": "p", "gold": "2", "public": True},
         {"id": "math500/b", "prompt": "p", "gold": "3", "public": True}]
recs = E.score_task("math500", items, [["\\boxed{2}", "\\boxed{1}"], ["<think>x", "\\boxed{3}"]])
s = E.summarize(recs)
grec = E.score_task("gpqa_diamond", [a], [["Answer: " + a["gold"]]])[0]
check("6 scoring + summary + GPQA privacy",
      [r["score"] for r in recs] == [0.5, 0.5] and s["avg_at_k"] == 50.0
      and s["truncated_rate"] == 25.0 and s["no_answer_rate"] == 25.0
      and grec["score"] == 1.0 and "prompt" not in grec and "replies" not in grec,
      json.dumps(s))

# [7]
rng = random.Random(0)
base = {f"i{i}": float(rng.random() < 0.6) for i in range(600)}
zero = E.paired_bootstrap(base, dict(base), n_boot=2000)
shift = dict(base)
flip = [k for k, v in base.items() if v == 0.0][:60]          # +10 pt exactly
for k in flip:
    shift[k] = 1.0
up = E.paired_bootstrap(base, shift, n_boot=4000)
cover = 0
for t in range(40):                     # coverage of the true diff (0) under noise
    r2 = random.Random(100 + t)
    x = {f"i{i}": float(r2.random() < 0.5) for i in range(300)}
    y = {f"i{i}": float(r2.random() < 0.5) for i in range(300)}
    bt = E.paired_bootstrap(x, y, n_boot=1000, seed=t)
    cover += bt["ci_lo"] <= 0 <= bt["ci_hi"]
check("7 paired bootstrap",
      zero["diff_pt"] == 0 and zero["ci_lo"] == 0 and zero["ci_hi"] == 0
      and up["diff_pt"] == 10.0 and 7.0 < up["ci_lo"] < 10.0 < up["ci_hi"] < 13.0
      and cover >= 34,
      f"shift {up}, coverage {cover}/40")

# [8]
ql3 = os.path.join(ROOT, "results", "0il1br2zf5k25e", "ql3", "ql3_bench.json")
rep = json.load(open(ql3))
v16, v8 = gates.qsa_verdict(rep, topk=16), gates.qsa_verdict(rep, topk=8)
other = gates.verdict("qsa", ql3, {"topk": 16}, base_model="Qwen/Qwen3-4B-Base")
same_base = gates.verdict("qsa", ql3, {"topk": 16}, base_model="Qwen/Qwen3-0.6B")
check("8 QSA gate from the measured QL3 report",
      v16["verdict"] == "PASS" and v8["verdict"] == "FAIL" and other["verdict"] == "FAIL"
      and same_base["verdict"] == "PASS"
      and gates.verdict("qsa", "/nonexistent.json")["verdict"] == "MISSING",
      f"top16: {v16['detail']} | top8: {v8['detail']}")

# [9]
q7 = {"arms": {"base": {"pass": False, "ratio": 1.0},
               "qp_sqa": {"pass": True, "ratio": 0.7, "ratio_ci95": [0.6, 0.9]},
               "qp_greedy_swap": {"pass": True, "ratio": 0.66},
               "dyn": {"pass": False, "ratio": 0.8}}, "pass_any": True}
check("9 QL7 verdict",
      gates.ql7_verdict(q7)["numbers"]["arm"] == "qp_greedy_swap"
      and gates.ql7_verdict(q7, "qp_sqa")["verdict"] == "PASS"
      and gates.ql7_verdict(q7, "dyn")["verdict"] == "FAIL"
      and gates.ql7_verdict({"arms": {"base": {"pass": False}}})["verdict"] == "FAIL")

# [10]
with tempfile.TemporaryDirectory() as td:
    art = os.path.join(td, "art")
    os.makedirs(os.path.join(art, "ql3_4b"))
    rep4 = dict(rep, model="Qwen/Qwen3-4B-Base")
    json.dump(rep4, open(os.path.join(art, "ql3_4b", "ql3_bench.json"), "w"))
    rec = R.load_recipe(os.path.join(ROOT, "phase4_llm", "release", "recipes", "diraclm-4b.json"))
    res = R.resolve(rec, art)
    ok_resolve = list(res.included) == ["qsa"] and set(res.excluded) == {"ql7", "ql5"}
    # fake eval dirs: model equal to base on MATH, +5 pt on BBH, GPQA missing
    eb, em = os.path.join(td, "eb"), os.path.join(td, "em")
    for d in (eb, em):
        os.makedirs(d)
    r3 = random.Random(1)
    for t, n, gain in (("math500", 500, 0.0), ("bbh", 2000, 0.05)):
        rb = [{"id": f"{t}/{i}", "gold": "1", "k": 1, "score": float(r3.random() < 0.5),
               "preds": ["1"], "no_answer": 0, "truncated": 0, "prompt": "q", "replies": ["r"]}
              for i in range(n)]
        rm = [dict(x, score=1.0 if (x["score"] == 1.0 or r3.random() < 2 * gain) else 0.0)
              for x in rb]
        for d, rows in ((eb, rb), (em, rm)):
            with open(os.path.join(d, f"{t}.jsonl"), "w") as f:
                f.writelines(json.dumps(x) + "\n" for x in rows)
    for d in (eb, em):
        json.dump({"backend": "vllm", "thinking": False, "max_tokens": 4096, "seed": 0},
                  open(os.path.join(d, "summary.json"), "w"))
    # gpqa: model file has question text that must be dropped on copy
    for d in (eb, em):
        with open(os.path.join(d, "gpqa_diamond.jsonl"), "w") as f:
            for i in range(198):
                f.write(json.dumps({"id": f"gpqa_diamond/{i}", "gold": "A", "k": 1, "score": 1.0,
                                    "preds": ["A"], "no_answer": 0, "truncated": 0,
                                    "prompt": "SECRET QUESTION"}) + "\n")
    wdir = os.path.join(td, "w")
    os.makedirs(wdir)
    for fn in ("model.safetensors", "config.json", "tokenizer.json", "LICENSE", "optimizer.pt"):
        open(os.path.join(wdir, fn), "w").write(fn)
    out = os.path.join(td, "rel")
    b = package.build(os.path.join(ROOT, "phase4_llm", "release", "recipes", "diraclm-4b.json"),
                      art, out, wdir, eb, em, repro="bash scripts/runpod_ql8.sh")
    card = open(os.path.join(out, "README.md")).read()
    man = json.load(open(os.path.join(out, "MANIFEST.json")))
    gp = open(os.path.join(out, "eval", "model", "gpqa_diamond.jsonl")).read()
    refused_private = refused_public = False
    try:
        package.check_push(out, "", public=False)
    except PermissionError:
        refused_private = True
    ready = b["gate"]["ready"]
    cfgp = os.path.join(out, "diraclm_config.json")
    cfg = json.load(open(cfgp))
    allowed_public = package.check_push(out, "Tony", public=True)["name"] == "DiracLM-4B"
    cfg["release_gate"] = {"ready": False, "reasons": ["test"]}
    json.dump(cfg, open(cfgp, "w"))
    try:
        package.check_push(out, "Tony", public=True)
    except PermissionError:
        refused_public = True
    private_ok = package.check_push(out, "Tony", public=False)["name"] == "DiracLM-4B"
    # not-ready variant: drop QSA -> nothing included
    res0 = R.Resolution({}, res.excluded)
    gate0 = R.release_gate(res0, package.reasoning_eval.compare_dirs(eb, em))
    card0 = model_card.render(rec, res0, gate0)
    check("10 recipe -> gate -> card -> package",
          ok_resolve and ready and "bbh" in b["gate"]["claims"]
          and b["gate"]["claims"]["bbh"].startswith("better than base")
          and b["gate"]["claims"]["math500"].startswith("within")
          and "optimizer.pt" not in man and "model.safetensors" in man and "LICENSE" in man
          and "SECRET" not in gp and "license: apache-2.0" in card
          and "No quantum hardware" in card and "FermiQLLM" in card
          and "Not a release candidate" not in card and "Not a release candidate" in card0
          and not gate0["ready"] and refused_private and refused_public
          and allowed_public and private_ok,
          f"gate={b['gate']} excluded={list(b['excluded'])}")

# [11]
try:
    import torch
    import torch.nn.functional as F
    from phase4_llm import qsa_attention as Q
except ImportError:
    torch = None
if torch is None:
    print("[SKIP] 11 QSA attention (torch not installed)")
else:
    torch.manual_seed(0)
    B, H, Hkv, D, L = 1, 4, 2, 16, 40
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, Hkv, L, D)
    v = torch.randn(B, Hkv, L, D)
    full = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    o_pre, _ = Q.qsa_hf_attention(None, q, k, v, None)
    o_dec, _ = Q.qsa_hf_attention(None, q[:, :, -1:], k, v, None)     # 1-token decode
    o_chk, _ = Q.qsa_hf_attention(None, q[:, :, -5:], k, v, None)     # chunked prefill
    pad = torch.ones(B, 1, L, L, dtype=torch.bool).tril()
    pad[..., :3] = False                                               # 3 left-pad keys
    pad[..., :3, :3] = torch.eye(3, dtype=torch.bool)                  # keep rows finite
    o_pad, _ = Q.qsa_hf_attention(None, q, k, v, pad)
    ref_pad = F.scaled_dot_product_attention(q, k, v, attn_mask=pad, enable_gqa=True)
    check("11 QSA HF attention: decode / chunked / padded paths",
          torch.allclose(o_pre.transpose(1, 2), full, atol=1e-5)
          and torch.allclose(o_dec.transpose(1, 2), full[:, :, -1:], atol=1e-5)
          and torch.allclose(o_chk.transpose(1, 2), full[:, :, -5:], atol=1e-5)
          and torch.allclose(o_pad.transpose(1, 2), ref_pad, atol=1e-5))

print(f"QL8 validation: {sum(PASS)}/{len(PASS)} passed")
sys.exit(0 if all(PASS) else 1)
