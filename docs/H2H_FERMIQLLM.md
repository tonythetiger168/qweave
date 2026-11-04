# Head-to-head with FermiQLLM 1.0 on its own benchmarks

> 2026-11-03 · 1× NVIDIA H100 80GB (RunPod) · vLLM 0.33, TRL 1.17.1, transformers 5.20, torch 2.14.
> Raw per-question results: [`results/h2h_lmer5m10omil58/`](../results/h2h_lmer5m10omil58/) (final table, all five models,
> GRPO training log); HF dataset `chienhsinlin/qweave-runs`, runs `zy3sq0lkjt0ww7_h2h`,
> `8fpwl0icp2kq4k_h2h`, `lmer5m10omil58_h2h`.
> Code: `phase4_llm/h2h/` · runner: `scripts/runpod_h2h.sh`.

## The claim being tested

FermiQLLM 1.0 (費米宇宙, 2026-09-28) says it beats **same-size open-source models
by 10–20 %** on MATH-500, GPQA-Diamond and BBH, via "full-chain quantum enhancement"
of a Qwen base. These are company-internal numbers: no weights, no paper, no
third-party reproduction ([investigation](FERMIQLLM_GLOBAL_INVESTIGATION.md)).

We ask the falsifiable version of that claim: **how much does a plain, published,
non-quantum recipe on the same Qwen base get?** If a few GPU-hours of standard RL reach
"+10–20 %", the claim says nothing about quantum methods.

## Setup

- **Baseline (same-size open-source model):** official `Qwen/Qwen3-0.6B`, non-thinking chat mode, greedy.
- **Ours:** `Qwen/Qwen3-0.6B` + GRPO, 300 steps, verifiable math reward (`math_verify`),
  trained on GSM8K-train + MATH-train (MATH-500 is drawn from MATH-test, so it is held out;
  GPQA and BBH are not trained on at all). 32 prompts × 8 samples per step, lr 1e-6,
  Dr. GRPO loss, no KL. **3.24 H100-hours**, no quantum component.
- **References:** Qwen3-0.6B thinking mode (T 0.6), Qwen3-1.7B non-thinking (3× the size),
  Qwen3-0.6B-Base (plain prompts).
- **Benchmarks (full test sets):** MATH-500 (500), GPQA-Diamond (198, options shuffled
  with a fixed seed), BBH (6,511 over 27 tasks, the official 3-shot CoT prompts).
  Same prompts and graders for every model; paired bootstrap 95 % CIs over questions.

## Results

| Model | MATH-500 | GPQA-Diamond | BBH (3-shot CoT) |
|---|---|---|---|
| Qwen3-0.6B non-thinking — **baseline** | 55.8 | 22.2 | 43.6 |
| **qweave GRPO 0.6B (ours, non-thinking)** | **64.4** (+8.6 pt [+4.4, +13.2], **+15.4 %**) | 21.7 (−0.5 [−8.1, +6.6]) | **46.8** (+3.2 pt [+2.2, +4.3], **+7.4 %**) |
| Qwen3-0.6B thinking | 74.0 (+32.6 %) | 20.2 (−2.0) | 52.0 (+19.2 %) |
| Qwen3-1.7B non-thinking | 73.6 (+31.9 %) | 26.8 (+4.5, n.s.) | 62.6 (+43.4 %) |
| Qwen3-0.6B-Base (plain) | 39.6 | 10.1 | 41.2 |

Relative gain = (model − baseline) / baseline, the form of FermiQLLM's "+10–20 %".
A CI that excludes 0 is a significant difference.

GRPO training: correctness reward 0.55 → 0.67 (first vs last fifth of training); mean
answer length 472 → 740 tokens, so the trained model is slower per answer.

## What this shows

1. **"+10–20 % on MATH-500" needs nothing quantum.** A standard open RL recipe gives
   **+15.4 %** (significant) in 3.2 GPU-hours. Simply switching Qwen3-0.6B to its
   built-in thinking mode gives **+32.6 %**. A headline gain in this range is not evidence
   for any "quantum enhancement" without a same-recipe control.
2. **Transfer is partial.** Math RL also lifts BBH by +7.4 % (significant, never trained on),
   below the claimed band, and does **not** move GPQA-Diamond (−0.5, CI spans 0). GPQA at
   0.6B sits near chance (25 %) for every model, so "+10–20 %" there would be within noise
   on 198 questions (CI half-width ≈ ±7 pt).
3. **The baseline choice decides the headline.** Against non-thinking Qwen3-0.6B, our model
   is +15 % on MATH-500; against thinking-mode Qwen3-0.6B it is −13 %. FermiQLLM does not
   say which "same-size open-source model" or mode it compared against.
4. Where qweave beats FermiQLLM outright is **verifiability**: every number above has a
   command, per-question correctness on disk and a stated baseline.

## Honest limits

- One training seed, greedy evaluation (thinking mode: one sample). No hyper-parameter search.
- GPQA questions and completions are never stored (the dataset's terms); only per-question
  correctness and the extracted letter.
- Qwen3-0.6B thinking scored 20.2 on GPQA vs Qwen's reported ~28: single-sample variance on
  198 questions plus our shuffled-option prompt; not tuned.
- The BBH prompt went through two fixes during this work (a zero-shot instruction that made
  the 0.6B model answer "(A)" on non-multiple-choice tasks; a Base-model prompt that ended
  at "step by step." where the exemplars continue on a new line, which left 61 % of Base
  answers empty). All reported numbers use the final official 3-shot prompts. Sanity check
  against Qwen's own published numbers: BBH 41.2 for Qwen3-0.6B-Base (Qwen: 41.5), GPQA 22.2
  for Qwen3-0.6B non-thinking (Qwen: ~23).

## Reproduce

```bash
curl -sL https://raw.githubusercontent.com/tonythetiger168/qweave/claude/h2h-fermiqllm/scripts/runpod_h2h.sh | bash
```

Needs `HF_TOKEN` with access to `Idavidrein/gpqa`; ~5 h on one H100 (≈ US$20). `QUICK=1` checks a
pod in ~15 minutes.
