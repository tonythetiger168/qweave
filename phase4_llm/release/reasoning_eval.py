"""
DiracLM reasoning benchmarks: MATH-500, GPQA-Diamond, BBH
=========================================================
FermiQLLM 1.0 claims +10-20 % on MATH-500, GPQA-Diamond and BBH with no
baseline, no protocol and no per-question output. This harness fixes all
three: the same prompt, sampling and grader for every model, every item's
result kept on disk, and a paired bootstrap over items for every
"model vs plain base" difference.

  # score one model (vLLM backend, Qwen3 thinking-mode sampling)
  python -m phase4_llm.release.reasoning_eval run --model Qwen/Qwen3-4B \
      --tasks math500,gpqa_diamond,bbh --out /workspace/ql8/eval/qwen3-4b

  # compare two scored models (paired over items)
  python -m phase4_llm.release.reasoning_eval compare \
      /workspace/ql8/eval/qwen3-4b /workspace/ql8/eval/diraclm-4b

Protocol (frozen before any GPU run, see docs/QL8_RELEASE_PLAN.md):
  * MATH-500   HuggingFaceH4/MATH-500 test, 500 items, "\\boxed{}" answer,
               Hendrycks/PRM800K-style normaliser (+ optional math_verify)
  * GPQA-D     Idavidrein/gpqa gpqa_diamond, 198 items, choices shuffled with
               a fixed per-item seed, "Answer: X"
  * BBH        lukaemon/bbh, 27 subtasks, 6,511 items, zero-shot CoT,
               "So the answer is ..."
  * k samples per item (default 4; 8 for GPQA), score = mean over samples,
    so a per-item accuracy in [0, 1]; headline = mean over items (avg@k)
  * <think>...</think> is stripped before grading; a reply that never
    closes its think block or never states an answer scores 0 and is
    counted in `no_answer`

GPQA's authors ask that questions are not published in plain text, so the
per-item file for GPQA stores ids, extracted answers and scores only.

Graders and the bootstrap are pure Python + numpy and run in CI
(validate_ql8.py); generation needs vLLM or transformers on a GPU.
"""

import argparse
import json
import math
import os
import random
import re
import time
from typing import Callable, Dict, List, Optional, Sequence

# ----------------------------------------------------------------------
# answer extraction and grading
# ----------------------------------------------------------------------
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def strip_think(text: str) -> Optional[str]:
    """Drop the reasoning block. None if a think block was opened and never
    closed (truncated generation): such replies have no final answer."""
    if "<think>" in text and "</think>" not in text:
        return None
    if "</think>" in text and "<think>" not in text:     # template put <think> in prompt
        return text.split("</think>", 1)[1]
    return THINK_RE.sub("", text)


def last_boxed(text: str) -> Optional[str]:
    """Content of the last \\boxed{...} / \\fbox{...}, with nested braces."""
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        # "\boxed 5" form
        m = re.match(r"\\boxed\s+([^\s$]+)", text[idx:])
        return m.group(1) if m else None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def _fix_fracs(s: str) -> str:
    parts = s.split("\\frac")
    out = parts[0]
    for p in parts[1:]:
        out += "\\frac"
        if p.startswith("{") or len(p) < 2:
            out += p
            continue
        a, b, rest = p[0], p[1], p[2:]
        if b == "{":                       # \frac1{2} -> \frac{1}{2}
            out += "{" + a + "}" + p[1:]
        else:                              # \frac12 -> \frac{1}{2}
            out += "{" + a + "}{" + b + "}" + rest
    return out


def _fix_sqrt(s: str) -> str:
    return re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)


def _slash_to_frac(s: str) -> str:
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    return f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}" if m else s


def normalize_math(s: str) -> str:
    """Hendrycks MATH / PRM800K-style string normaliser."""
    s = s.strip().replace("\n", "").replace("\\!", "")
    s = s.replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("°", "")
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\%", "").replace("%", "")
    s = re.sub(r"\\text\{\s*([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{\s*([^{}]*)\}", r"\1", s)
    s = s.replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    # trailing unit words after a number: "5 cm", "10 inches"
    s = re.sub(r"^(-?[\d.,]+)\s+[a-zA-Z ]+$", r"\1", s)
    s = s.replace(" ", "")
    if s.endswith("."):
        s = s[:-1]
    s = s.replace("{.", "{0.")
    if s.startswith("."):
        s = "0" + s
    # "x=5" -> "5" (short left-hand side only)
    if s.count("=") == 1 and len(s.split("=")[0]) <= 2:
        s = s.split("=")[1]
    s = _fix_sqrt(s)
    s = _fix_fracs(s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    s = _slash_to_frac(s)
    # 1,000 -> 1000 (only digit groups)
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    return s


def _as_number(s: str) -> Optional[float]:
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(\d+)\}", s)
    if m:
        den = int(m.group(3))
        if den == 0:
            return None
        v = int(m.group(2)) / den
        return -v if m.group(1) else v
    try:
        return float(s)
    except ValueError:
        return None


def math_equal(pred: Optional[str], gold: str, use_math_verify: bool = False) -> bool:
    if pred is None:
        return False
    if use_math_verify:
        try:
            from math_verify import parse, verify  # optional, GPU pods only
            return bool(verify(parse("$" + gold + "$"), parse("$" + pred + "$")))
        except ImportError:
            pass
    a, b = normalize_math(pred), normalize_math(gold)
    if a == b:
        return True
    x, y = _as_number(a), _as_number(b)
    return x is not None and y is not None and math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-9)


LETTER_PATTERNS = [
    r"[Aa]nswer\s*(?:is)?\s*[:：]?\s*\(?\**\s*([A-D])\b",
    r"\\boxed\{\s*\(?([A-D])\)?\s*\}",
    r"\(([A-D])\)",
]


def extract_choice(text: str, letters: str = "ABCD") -> Optional[str]:
    """Last stated option letter: 'Answer: X', then \\boxed{X}, then '(X)'."""
    for pat in LETTER_PATTERNS:
        hits = [h for h in re.findall(pat, text) if h in letters]
        if hits:
            return hits[-1]
    return None


def _norm_bbh(s: str) -> str:
    s = s.strip().strip("*").strip()
    s = re.sub(r"\s+", " ", s)
    if s.endswith(".") and not re.fullmatch(r"-?\d+\.", s):
        s = s[:-1]
    return s.strip().lower()


def extract_bbh(text: str) -> Optional[str]:
    hits = re.findall(r"[Tt]he answer is[:\s]*(.+?)(?:\n|$)", text)
    if hits:
        return hits[-1].strip()
    b = last_boxed(text)
    return b.strip() if b is not None else None


def bbh_equal(pred: Optional[str], target: str) -> bool:
    if pred is None:
        return False
    t = target.strip()
    if re.fullmatch(r"\([A-R]\)", t):                      # multiple choice
        m = re.findall(r"\(([A-R])\)", pred) or re.findall(r"^\s*([A-R])\b", pred)
        return bool(m) and m[-1] == t[1]
    p, g = _norm_bbh(pred), _norm_bbh(t)
    if p == g:
        return True
    # numbers: "24" vs "24.0"
    try:
        return math.isclose(float(p.replace(",", "")), float(g), abs_tol=1e-9)
    except ValueError:
        return False


# ----------------------------------------------------------------------
# tasks
# ----------------------------------------------------------------------
BBH_SUBTASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa",
    "dyck_languages", "formal_fallacies", "geometric_shapes", "hyperbaton",
    "logical_deduction_five_objects", "logical_deduction_seven_objects",
    "logical_deduction_three_objects", "movie_recommendation", "multistep_arithmetic_two",
    "navigate", "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection", "snarks", "sports_understanding",
    "temporal_sequences", "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects",
    "web_of_lies", "word_sorting",
]

MATH_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
GPQA_SUFFIX = ("\n\nThink step by step, then give the letter of the correct option "
               "on the last line in the form 'Answer: X'.")
BBH_SUFFIX = ("\n\nThink step by step, then finish your reply with "
              "\"So the answer is <answer>.\"")


def gpqa_item(row: Dict, idx: int) -> Dict:
    """Shuffle the four options with a seed fixed by the item index."""
    opts = [row["Correct Answer"], row["Incorrect Answer 1"],
            row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    order = list(range(4))
    random.Random(f"gpqa-diamond-{idx}").shuffle(order)
    letters = "ABCD"
    body = row["Question"].strip() + "\n\n" + "\n".join(
        f"({letters[k]}) {opts[o].strip()}" for k, o in enumerate(order))
    return {"id": f"gpqa_diamond/{idx}", "prompt": body + GPQA_SUFFIX,
            "gold": letters[order.index(0)], "public": False}


def load_items(task: str, limit: Optional[int] = None) -> List[Dict]:
    from datasets import load_dataset
    items: List[Dict] = []
    if task == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        for r in ds:
            items.append({"id": f"math500/{r['unique_id']}", "prompt": r["problem"] + MATH_SUFFIX,
                          "gold": r["answer"], "public": True,
                          "meta": {"subject": r.get("subject"), "level": r.get("level")}})
    elif task == "gpqa_diamond":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        items = [gpqa_item(r, i) for i, r in enumerate(ds)]
    elif task == "bbh":
        for sub in BBH_SUBTASKS:
            ds = load_dataset("lukaemon/bbh", sub, split="test")
            for i, r in enumerate(ds):
                items.append({"id": f"bbh/{sub}/{i}", "prompt": r["input"] + BBH_SUFFIX,
                              "gold": r["target"], "public": True, "meta": {"subtask": sub}})
    else:
        raise ValueError(f"unknown task {task}")
    return items[:limit] if limit else items


def grade(task: str, reply: str, gold: str, use_math_verify: bool = False) -> Dict:
    final = strip_think(reply)
    if final is None:
        return {"pred": None, "correct": False, "truncated": True}
    if task == "math500":
        pred = last_boxed(final)
        ok = math_equal(pred, gold, use_math_verify)
    elif task == "gpqa_diamond":
        pred = extract_choice(final)
        ok = pred == gold
    elif task == "bbh":
        pred = extract_bbh(final)
        ok = bbh_equal(pred, gold)
    else:
        raise ValueError(task)
    return {"pred": pred, "correct": bool(ok), "truncated": False}


def score_task(task: str, items: Sequence[Dict], replies: Sequence[Sequence[str]],
               use_math_verify: bool = False) -> List[Dict]:
    """replies[i] = k sampled completions for items[i] -> per-item records."""
    recs = []
    for it, outs in zip(items, replies):
        g = [grade(task, o, it["gold"], use_math_verify) for o in outs]
        rec = {"id": it["id"], "gold": it["gold"], "k": len(g),
               "score": sum(x["correct"] for x in g) / max(len(g), 1),
               "preds": [x["pred"] for x in g],
               "no_answer": sum(x["pred"] is None for x in g),
               "truncated": sum(x["truncated"] for x in g)}
        if it.get("meta"):
            rec["meta"] = it["meta"]
        if it.get("public", True):
            rec["prompt"] = it["prompt"]
            rec["replies"] = list(outs)
        recs.append(rec)
    return recs


def summarize(recs: Sequence[Dict]) -> Dict:
    n = len(recs)
    k = sum(r["k"] for r in recs)
    out = {"n_items": n, "avg_at_k": round(100 * sum(r["score"] for r in recs) / max(n, 1), 2),
           "no_answer_rate": round(100 * sum(r["no_answer"] for r in recs) / max(k, 1), 2),
           "truncated_rate": round(100 * sum(r["truncated"] for r in recs) / max(k, 1), 2)}
    subs: Dict[str, List[float]] = {}
    for r in recs:
        st = (r.get("meta") or {}).get("subtask")
        if st:
            subs.setdefault(st, []).append(r["score"])
    if subs:
        out["subtasks"] = {s: round(100 * sum(v) / len(v), 2) for s, v in sorted(subs.items())}
        out["macro_avg"] = round(sum(out["subtasks"].values()) / len(subs), 2)
    return out


# ----------------------------------------------------------------------
# paired bootstrap over items
# ----------------------------------------------------------------------
def paired_bootstrap(a: Dict[str, float], b: Dict[str, float], n_boot: int = 10000,
                     seed: int = 0, alpha: float = 0.05) -> Dict:
    """Difference b - a in percentage points over the shared items, with a
    percentile bootstrap CI (items resampled with replacement)."""
    import numpy as np
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ValueError("no shared items")
    d = np.array([b[i] - a[i] for i in ids], dtype=np.float64) * 100.0
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    chunk = max(1, min(n_boot, 20_000_000 // max(len(d), 1)))
    for s in range(0, n_boot, chunk):
        e = min(n_boot, s + chunk)
        idx = rng.integers(0, len(d), size=(e - s, len(d)))
        boots[s:e] = d[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"n_items": len(ids), "diff_pt": round(float(d.mean()), 2),
            "ci_lo": round(float(lo), 2), "ci_hi": round(float(hi), 2),
            "a_pct": round(100 * sum(a[i] for i in ids) / len(ids), 2),
            "b_pct": round(100 * sum(b[i] for i in ids) / len(ids), 2)}


def read_scores(path: str) -> Dict[str, float]:
    with open(path) as f:
        return {r["id"]: r["score"] for r in map(json.loads, f)}


def compare_dirs(dir_a: str, dir_b: str, tasks: Sequence[str] = ("math500", "gpqa_diamond", "bbh"),
                 n_boot: int = 10000) -> Dict:
    out = {"a": dir_a, "b": dir_b, "tasks": {}}
    for t in tasks:
        pa, pb = os.path.join(dir_a, f"{t}.jsonl"), os.path.join(dir_b, f"{t}.jsonl")
        if os.path.exists(pa) and os.path.exists(pb):
            out["tasks"][t] = paired_bootstrap(read_scores(pa), read_scores(pb), n_boot)
    return out


# ----------------------------------------------------------------------
# generation backends (GPU)
# ----------------------------------------------------------------------
QWEN3_THINKING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
QWEN3_NO_THINK = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}


def _chat(tok, prompt: str, thinking: bool) -> str:
    if getattr(tok, "chat_template", None) is None:      # plain base checkpoints
        return prompt + "\n\n"
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=thinking)
    except TypeError:                       # templates without the thinking switch
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def vllm_generator(model: str, max_tokens: int, thinking: bool, max_model_len: int,
                   seed: int = 0, tp: int = 1) -> Callable[[List[str], int], List[List[str]]]:
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, dtype="bfloat16", max_model_len=max_model_len, seed=seed,
              tensor_parallel_size=tp, trust_remote_code=False)
    tok = llm.get_tokenizer()
    samp = QWEN3_THINKING if thinking else QWEN3_NO_THINK

    def gen(prompts: List[str], k: int) -> List[List[str]]:
        sp = SamplingParams(n=k, max_tokens=max_tokens, seed=seed, **samp)
        outs = llm.generate([_chat(tok, p, thinking) for p in prompts], sp)
        return [[c.text for c in o.outputs] for o in outs]
    return gen


def hf_generator(model: str, max_tokens: int, thinking: bool, seed: int = 0,
                 batch: int = 8) -> Callable[[List[str], int], List[List[str]]]:
    """transformers backend for models vLLM cannot load (QSA attention,
    QL5 low-rank layers). Uses phase4_llm.release.loader."""
    import torch
    from phase4_llm.release.loader import load_diraclm
    m, tok = load_diraclm(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    samp = QWEN3_THINKING if thinking else QWEN3_NO_THINK

    @torch.no_grad()
    def gen(prompts: List[str], k: int) -> List[List[str]]:
        torch.manual_seed(seed)
        res: List[List[str]] = []
        for i in range(0, len(prompts), batch):
            chunk = [_chat(tok, p, thinking) for p in prompts[i:i + batch]]
            enc = tok(chunk, return_tensors="pt", padding=True).to(m.device)
            out = m.generate(**enc, do_sample=True, max_new_tokens=max_tokens,
                             num_return_sequences=k, **samp)
            txt = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            res += [txt[j * k:(j + 1) * k] for j in range(len(chunk))]
        return res
    return gen


DEFAULT_K = {"math500": 4, "gpqa_diamond": 8, "bbh": 1}


def run(args) -> Dict:
    os.makedirs(args.out, exist_ok=True)
    if args.backend == "vllm":
        gen = vllm_generator(args.model, args.max_tokens, args.thinking, args.max_model_len,
                             args.seed, args.tp)
    else:
        gen = hf_generator(args.model, args.max_tokens, args.thinking, args.seed)
    summary = {"model": args.model, "backend": args.backend, "thinking": args.thinking,
               "max_tokens": args.max_tokens, "seed": args.seed, "tasks": {}}
    for task in args.tasks.split(","):
        items = load_items(task, args.limit)
        k = args.k or DEFAULT_K[task]
        t0 = time.time()
        replies = gen([it["prompt"] for it in items], k)
        recs = score_task(task, items, replies, args.math_verify)
        with open(os.path.join(args.out, f"{task}.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        s = summarize(recs)
        s.update(k=k, wall_s=round(time.time() - t0, 1))
        summary["tasks"][task] = s
        print(f"[ql8-eval] {task}: {json.dumps(s)[:300]}", flush=True)
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--tasks", default="math500,gpqa_diamond,bbh")
    r.add_argument("--out", required=True)
    r.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    r.add_argument("--k", type=int, default=0, help="samples per item (0 = task default)")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--max-tokens", type=int, default=32768)
    r.add_argument("--max-model-len", type=int, default=40960)
    r.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    r.add_argument("--math-verify", action="store_true")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--tp", type=int, default=1)
    c = sub.add_parser("compare")
    c.add_argument("base_dir")
    c.add_argument("model_dir")
    c.add_argument("--out", default="")
    c.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a)
    else:
        res = compare_dirs(a.base_dir, a.model_dir, n_boot=a.n_boot)
        print(json.dumps(res, indent=2))
        if a.out:
            with open(a.out, "w") as f:
                json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
