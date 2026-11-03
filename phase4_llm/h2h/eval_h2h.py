"""
Head-to-head with FermiQLLM 1.0's headline claim
=================================================
FermiQLLM 1.0 says it beats same-size open-source models by 10-20 % on
MATH-500, GPQA-Diamond and BBH (company-internal numbers, no weights).
This script scores any HF model on those three benchmarks with vLLM, the
same prompts and graders for every model, and saves per-question
correctness so `report_h2h.py` can give paired bootstrap CIs.

  python phase4_llm/h2h/eval_h2h.py --model Qwen/Qwen3-0.6B --mode nothink \
      --name qwen3-0.6b --out /workspace/h2h

Modes
  plain    raw text prompt (Base models)
  nothink  chat template, enable_thinking=False, greedy
  think    chat template, enable_thinking=True, T=0.6 top-p 0.95 (Qwen's setting)

GPQA's authors ask that its questions are never published, so outputs
store only per-question correctness, never question text or completions.
"""

import argparse, json, os, random, re, time

MATH_INSTR = "Please reason step by step, and put your final answer within \\boxed{}."
MC_INSTR = ("Think step by step, then finish with a line of the form "
            "'Answer: X' where X is one of A, B, C, D.")
BBH_TASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "dyck_languages", "formal_fallacies",
    "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate",
    "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection", "snarks",
    "sports_understanding", "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting"]


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def load_math500():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [{"id": r["unique_id"], "q": r["problem"], "gold": r["answer"]} for r in ds]


def load_gpqa(seed=0):
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    items = []
    for i, r in enumerate(ds):
        opts = [r["Correct Answer"], r["Incorrect Answer 1"],
                r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        perm = list(range(4))
        random.Random(seed * 100003 + i).shuffle(perm)      # fixed per question
        letters = "ABCD"
        body = "\n".join(f"{letters[k]}) {opts[perm[k]].strip()}" for k in range(4))
        gold = letters[perm.index(0)]
        items.append({"id": f"gpqa{i}", "q": f"{r['Question'].strip()}\n\n{body}",
                      "gold": gold})
    return items


BBH_COT_URL = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/cot-prompts/{}.txt"


def bbh_fewshot(task):
    """Official 3-shot chain-of-thought exemplars (Suzgun et al. 2022), header stripped."""
    import urllib.request
    txt = urllib.request.urlopen(BBH_COT_URL.format(task), timeout=60).read().decode()
    return txt.split("-----", 1)[-1].strip()


def load_bbh():
    from datasets import load_dataset
    items = []
    for t in BBH_TASKS:
        few = bbh_fewshot(t)
        ds = load_dataset("lukaemon/bbh", t, split="test")
        for i, r in enumerate(ds):
            items.append({"id": f"{t}/{i}", "task": t, "q": r["input"], "gold": r["target"],
                          "fewshot": few})
    return items


# ----------------------------------------------------------------------
# prompts
# ----------------------------------------------------------------------
def build_prompt(tok, bench, q, mode, fewshot=None):
    if bench == "bbh":            # standard BBH: 3-shot CoT, "So the answer is X."
        text = f"{fewshot}\n\nQ: {q}\nA: Let's think step by step."
        if mode == "plain":
            return text
        msgs = [{"role": "user", "content": text + "\n\n(Answer the last question in the "
                 "same format as the examples, ending with 'So the answer is ...'.)"}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=(mode == "think"))
    instr = {"math500": MATH_INSTR, "gpqa": MC_INSTR}[bench]
    if mode == "plain":
        return f"Question: {q}\n{instr}\nAnswer: Let's think step by step."
    msgs = [{"role": "user", "content": f"{q}\n\n{instr}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=(mode == "think"))


def strip_think(text):
    text = text.split("</think>")[-1]
    # plain (Base) mode: the model keeps writing new few-shot items; keep the first answer
    return re.split(r"\n\s*(?:Q|Question):", text, maxsplit=1)[0]


# ----------------------------------------------------------------------
# graders
# ----------------------------------------------------------------------
def grade_math(pred, gold):
    from math_verify import parse, verify
    try:
        g = parse("\\boxed{" + gold + "}")
        p = parse(pred)
        return bool(verify(g, p))
    except Exception:
        return False


def grade_mc(pred, gold):
    m = re.findall(r"Answer\s*[:：]\s*\(?\s*([ABCD])\b", pred)
    if not m:
        m = re.findall(r"\b([ABCD])\)?\s*$", pred.strip())
    return bool(m) and m[-1] == gold


def _norm(s):
    s = re.sub(r"</?answer>?|<answer|>$", "", s, flags=re.IGNORECASE)
    s = s.replace("**", "").strip().strip(".").strip()
    s = s.strip("\"'`“”‘’").strip().strip(".").strip()
    s = re.sub(r"^\((\w)\)$", r"\1", s)                     # "(A)" -> "A"
    return re.sub(r"\s+", " ", s).lower()


def grade_bbh(pred, gold):
    m = re.findall(r"answer is\s*:?\s*(.+?)\s*(?:\.\s*)?$", pred.strip(),
                   flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return False
    last = m[-1]
    while re.search(r"answer is", last, flags=re.IGNORECASE):   # "answer is: **So the answer is (D)**"
        last = re.split(r"answer is\s*:?", last, flags=re.IGNORECASE)[-1]
    ans = _norm(last.replace("**", ""))
    g = _norm(gold)
    if ans == g:
        return True
    mc = re.match(r"^\(?([a-r])\)", ans)                     # "(a) option text"
    return bool(mc) and len(g) == 1 and mc.group(1) == g


GRADERS = {"math500": grade_math, "gpqa": grade_mc, "bbh": grade_bbh}
LOADERS = {"math500": load_math500, "gpqa": load_gpqa, "bbh": load_bbh}


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["plain", "nothink", "think"], default="nothink")
    ap.add_argument("--benches", default="math500,gpqa,bbh")
    ap.add_argument("--limit", type=int, default=0, help="per benchmark (0 = all)")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="default: 4096 (plain/nothink), 16384 (think)")
    ap.add_argument("--max-model-len", type=int, default=20480)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--out", default="/workspace/h2h")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_mem, seed=0)
    max_tok = a.max_tokens or (16384 if a.mode == "think" else 4096)
    if a.mode == "think":
        sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=max_tok, seed=0)
    else:
        # no stop strings: with vLLM they emptied Base-model BBH answers whose few-shot
        # reasoning starts on a new line; plain-mode output is cut afterwards instead
        sp = SamplingParams(temperature=0.0, max_tokens=max_tok)

    os.makedirs(a.out, exist_ok=True)
    report = {"model": a.model, "name": a.name, "mode": a.mode, "max_tokens": max_tok,
              "limit": a.limit, "benches": {}}
    for bench in a.benches.split(","):
        t0 = time.time()
        items = LOADERS[bench]()
        if a.limit:
            items = items[:a.limit] if bench != "bbh" else \
                [it for k, it in enumerate(items) if k % max(1, len(items) // a.limit) == 0][:a.limit]
        prompts = [build_prompt(tok, bench, it["q"], a.mode, it.get("fewshot")) for it in items]
        outs = llm.generate(prompts, sp)
        correct, ntoks, trunc = [], [], 0
        for it, o in zip(items, outs):
            txt = strip_think(o.outputs[0].text)
            ntoks.append(len(o.outputs[0].token_ids))
            trunc += o.outputs[0].finish_reason == "length"
            correct.append(int(GRADERS[bench](txt, it["gold"])))
        res = {"n": len(items), "acc": round(100 * sum(correct) / max(len(items), 1), 2),
               "ids": [it["id"] for it in items], "correct": correct,
               "mean_gen_tokens": round(sum(ntoks) / max(len(ntoks), 1), 1),
               "truncated": trunc, "seconds": round(time.time() - t0, 1)}
        if bench == "bbh":                                   # macro average over tasks
            per = {}
            for it, c in zip(items, correct):
                per.setdefault(it["task"], []).append(c)
            res["per_task"] = {t: round(100 * sum(v) / len(v), 1) for t, v in per.items()}
            res["acc_macro"] = round(sum(res["per_task"].values()) / len(per), 2)
        # keep enough to re-grade offline; GPQA: only the extracted letter, never text
        if bench == "gpqa":
            res["extracted"] = [(re.findall(r"Answer\s*[:：]\s*\(?\s*([ABCD])\b",
                                            strip_think(o.outputs[0].text)) or [""])[-1]
                                 for o in outs]
        else:
            res["samples"] = [{"id": it["id"], "gold": it["gold"],
                               "pred_tail": strip_think(o.outputs[0].text)[-300:]}
                              for it, o in zip(items, outs)]
        report["benches"][bench] = res
        print(f"[h2h] {a.name} {bench}: {res['acc']}% (n={res['n']}, "
              f"{res['mean_gen_tokens']} tok, {trunc} truncated, {res['seconds']} s)", flush=True)
        json.dump(report, open(f"{a.out}/{a.name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
