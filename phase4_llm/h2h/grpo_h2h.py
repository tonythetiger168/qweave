"""
Open recipe: GRPO on math with a verifiable reward (TRL + colocated vLLM)
========================================================================
The honest question behind FermiQLLM's "+10-20 % on MATH-500 / GPQA /
BBH": how much does a plain, published RL recipe on the same base give,
with no "quantum" component at all? This trains Qwen3-0.6B (non-thinking
chat format) with GRPO on GSM8K-train + MATH-train (disjoint from
MATH-500, which is drawn from MATH-test), reward = math_verify
correctness of the \\boxed{} answer.

  python phase4_llm/h2h/grpo_h2h.py --model Qwen/Qwen3-0.6B \
      --steps 400 --out /workspace/h2h/grpo-0.6b
"""

import argparse, dataclasses, json, os, random, time

INSTR = "Please reason step by step, and put your final answer within \\boxed{}."


def build_dataset(tok, n_gsm, n_math, seed=0):
    from datasets import load_dataset, Dataset
    rows = []
    g = load_dataset("openai/gsm8k", "main", split="train")
    for r in g.select(range(min(n_gsm, len(g)))):
        rows.append({"q": r["question"], "gold": r["answer"].split("####")[-1].strip()})
    m = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    for r in m.shuffle(seed=seed).select(range(min(n_math, len(m)))):
        if r.get("level") in ("Level 1",):                  # too easy to carry signal
            continue
        rows.append({"q": r["problem"], "gold_solution": r["solution"], "gold": ""})
    random.Random(seed).shuffle(rows)

    def render(q):
        return tok.apply_chat_template([{"role": "user", "content": f"{q}\n\n{INSTR}"}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    return Dataset.from_list([{"prompt": render(r["q"]), "gold": r["gold"],
                               "gold_solution": r.get("gold_solution", "")} for r in rows])


def correctness_reward(prompts, completions, gold, gold_solution, **kw):
    from math_verify import parse, verify
    out = []
    for c, g, gs in zip(completions, gold, gold_solution):
        text = c if isinstance(c, str) else c[0]["content"]
        try:
            gt = parse("\\boxed{" + g + "}") if g else parse(gs)
            out.append(1.0 if verify(gt, parse(text)) else 0.0)
        except Exception:
            out.append(0.0)
    return out


def boxed_format_reward(prompts, completions, **kw):
    """Small shaping term: 0.1 for producing exactly one \\boxed{} answer."""
    res = []
    for c in completions:
        text = c if isinstance(c, str) else c[0]["content"]
        res.append(0.1 if text.count("\\boxed{") == 1 else 0.0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--prompts-per-step", type=int, default=32)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=1536)
    ap.add_argument("--n-gsm", type=int, default=7473)
    ap.add_argument("--n-math", type=int, default=7500)
    ap.add_argument("--vllm-mem", type=float, default=0.35)
    ap.add_argument("--out", default="/workspace/h2h/grpo-0.6b")
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    tok = AutoTokenizer.from_pretrained(a.model)
    ds = build_dataset(tok, a.n_gsm, a.n_math)
    print(f"[grpo] {len(ds)} training prompts", flush=True)

    completions_per_step = a.prompts_per_step * a.num_generations
    micro = min(64, completions_per_step)
    want = dict(
        output_dir=a.out, learning_rate=a.lr, lr_scheduler_type="constant_with_warmup",
        warmup_steps=10, max_steps=a.steps, per_device_train_batch_size=micro,
        gradient_accumulation_steps=max(1, completions_per_step // micro),
        num_generations=a.num_generations, max_prompt_length=768,
        max_completion_length=a.max_completion, temperature=1.0, top_p=1.0,
        beta=0.0, loss_type="dr_grpo", scale_rewards=False,
        bf16=True, gradient_checkpointing=True, logging_steps=5,
        save_strategy="no", report_to=[], seed=0,
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_mem,
        reward_weights=[1.0, 1.0], log_completions=False)
    fields = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = sorted(k for k in want if k not in fields)
    cfg = GRPOConfig(**{k: v for k, v in want.items() if k in fields})
    print(f"[grpo] GRPOConfig args not in this TRL version (dropped): {dropped}", flush=True)

    trainer = GRPOTrainer(model=a.model, args=cfg, train_dataset=ds,
                          reward_funcs=[correctness_reward, boxed_format_reward],
                          processing_class=tok)
    t0 = time.time()
    trainer.train()
    hours = (time.time() - t0) / 3600
    os.makedirs(a.out, exist_ok=True)
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    hist = [h for h in trainer.state.log_history if "reward" in h or "rewards/correctness_reward/mean" in h]
    json.dump({"model": a.model, "steps": a.steps, "lr": a.lr,
               "prompts_per_step": a.prompts_per_step, "num_generations": a.num_generations,
               "train_hours": round(hours, 3),
               "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
               "dropped_config_args": dropped, "log": hist},
              open(os.path.join(a.out, "grpo_report.json"), "w"), indent=1)
    print(f"[grpo] done in {hours:.2f} h -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
