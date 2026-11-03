# QL7 stage A1 results (1× H100, 2026-11-03)

Pod `nj4lu0outmhol3` (1× H100 80GB SXM, Secure), commit `5122b16` (branch `claude/ql7-quick-fixes`),
torch 2.14.0+cu130, transformers 5.20.2, vLLM 0.33.0, TRL 1.17.1. Raw data: HF dataset
`chienhsinlin/qweave-runs`, `runs/nj4lu0outmhol3_ql7/stageA1` (QUICK checks: `runs/i30fv79wcbur4u_ql7`,
`runs/m2qiqpg1i8d3u4_ql7`). One seed each, so treat every number below as a single run.

## Baseline GRPO (Qwen3-0.6B-Base, GSM8K, 200 steps, B = 64 × G = 8)

| step | 0 | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GSM8K test (1,319, greedy) | 45.7 | 54.6 | 60.4 | 63.5 | 65.0 | 67.9 | 68.2 | 67.9 | 70.9 | 70.8 | 68.8 |
| GPU-hours | 0 | 0.048 | 0.095 | 0.138 | 0.181 | 0.215 | 0.251 | 0.284 | 0.317 | 0.353 | 0.387 |

* Time split (eval excluded): generation 33%, training 66%, everything else 1%.
  Generation is short because rollouts stop at the base model's next "Question:".
* Zero-signal groups (all 8 rewards equal) grow as the policy improves: 26%, 46%, 56%,
  58% of groups in steps 1–50, 51–100, 101–150 and 151–200. Training already skips
  zero-advantage rows (53% of completions were trained on), so those groups cost
  generation time only.

## TRL reference (same recipe in `GRPOTrainer`, vLLM colocate)

| | our loop (`base`) | TRL 1.17.1 |
|---|---|---|
| GPU-hours, 200 steps | **0.387** | 1.178 |
| GPU-hours per step | 0.00193 | 0.00589 |
| final GSM8K | 68.8 | 67.9 |

Our baseline loop is not a slow strawman: it is about **3× faster per step** than the
standard tool on the same recipe and ends at the same score. (TRL's wall time is
`trainer.train()` only; its eval is excluded like ours.)

## What A1 says about the QUBO levers

**Prompt selection (persistence bound, from the baseline logs only).** 41.6% of prompt
visits in 200 steps were revisits. On a revisit, a prompt that was zero-signal last
time was zero-signal again 74% of the time (vs 48% for a prompt that had signal).
Skipping every last-time-zero prompt would skip 38% of revisit rollouts and wrongly
skip 9.9% of revisits. That passes the plan's literal A1 rule (> 25% rollouts,
< 10% false skips), **but in GPU-hours the ceiling is small**: only generation is
saved (training already ignores zero-signal groups), so the saving is at most
0.33 × 0.38 ≈ 13% of the step time once every visit is a revisit, and ≈ 5% in a
200-step run. Reaching 0.75× would need selection to also make each GPU-hour learn
more (better sample efficiency), which no measurement here supports.

**Layer selection.** In the QUICK rerun, training 14 of 28 blocks per step took
50.7 s of training time over 10 steps vs 52.0 s for all blocks (−2.5%): the tied
embedding/LM head is always trained and autograd still runs through every block
above the lowest selected one. With training at 66% of the step, even a perfect
halving of block cost would give ≈ 0.83×.

## Verdict

* The interacting-QUBO levers (SQA or classical) **cannot reach 0.75×** against our
  own baseline on this setup: measured ceilings ≈ 0.87× (prompts) and ≈ 0.97× (layers
  as implemented). QL7's QUBO route is a negative result, as the plan allows.
* Against the standard tool, a plain, well-engineered GRPO loop reaches the same
  score at **≈ 0.33× TRL's GPU-hours** (one seed each). This is the defensible
  "RL cost" claim, and it has nothing to do with annealing.
* Stage A2 (4× H100, ~US$65–80) as planned would mostly confirm the ceilings above;
  not recommended. A cheaper follow-up that would make the TRL comparison citable:
  2 more seeds of `base` and of `trl_ref` (~US$12).

Cost of QL7 GPU work so far: ≈ US$20 over six pods (three failed in setup; see
`qweave-ql7` notes in the PR thread).
