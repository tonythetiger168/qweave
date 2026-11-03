# QL7 — RL at ≤0.75× the GPU-hours (plan and run plan)

> 2026-11-01 · Code: `phase4_llm/ql7_qubo.py`, `ql7_rl.py`, `ql7_grpo_train.py`,
> `ql7_trl_ref.py`, `ql7_sync_check.py`, `ql7_gate.py`, `validate_ql7.py` (CPU, in CI),
> `scripts/runpod_ql7.sh`. No GPU run has been done yet.

## 1. Target and gate

FermiQLLM claims "RL cost −25%" with no baseline. Our version is checkable:

* **Task:** GRPO on GSM8K train, Qwen3-0.6B-Base (then Qwen3-4B-Base), graded on the
  full GSM8K test set (1,319 items, greedy, same template and grader for every arm).
* **Named baseline (`base`):** the same GRPO loop with uniform prompt sampling and all
  blocks trained (B = 64 prompts × G = 8 completions, DAPO token-level loss, clip
  0.2/0.28, no KL, lr 1e-6, fp32 master weights + bf16 autocast, vLLM rollouts
  colocated). A TRL `GRPOTrainer` run with the same recipe (`trl_ref`) checks that our
  baseline loop is not slower per step than the standard tool; if it is, the gate is
  re-run against TRL.
* **Gate (`ql7_gate.py`):** target S\* = baseline mean final score − 1 pt. For each arm,
  GPU-hours to first reach S\* (centred 3-point smoothing, interpolated) divided by the
  baseline's. **PASS** = ratio ≤ 0.75 with the seed-bootstrap 95% CI upper bound ≤ 1.0,
  **and** final score non-inferior (paired per-question bootstrap, 95% CI lower bound
  ≥ −2 pt). GPU-hours count rollouts, scoring, training, weight sync, probe steps and
  the QUBO solve time; eval time is excluded (identical for all arms).
* Anything else is published as a negative result.

## 2. Where the savings can come from (Amdahl)

Per GRPO step the time goes to generation, policy forward/backward, and small
overheads. `amdahl_ratio` in `ql7_rl.py`:

    ratio = f_gen·(1 − s_gen) + f_train·(1 − s_train) + f_other

With a typical colocated split (f_gen ≈ 0.6, f_train ≈ 0.35), cutting 30% of both
phases gives 0.715× (passes), while cutting 40% of training alone gives only 0.86×
(fails). So **a lever that only touches the backward pass cannot reach 0.75× on its
own**; the lever has to remove rollouts. Stage A1 measures the real fractions.

## 3. The two levers, each as an interacting QUBO, each head to head

QL2 (F5) found annealing is the wrong tool for a separable objective. QL7 only uses
it where the objective has pairwise terms, and always against greedy, greedy + 1-swap
local search (the strong classical baseline), SA and random controls.
`ql7_qubo.py` solves "pick k of n" exactly with swap moves (no penalty terms).

**Lever 1, prompt selection (cuts generation and training).** A GRPO group whose G
rewards are all equal has advantage 0 and gives no gradient, yet paid for G rollouts.
Each prompt keeps a Beta posterior on its pass rate (U-shaped prior, decays back to
the prior over time so skipped prompts are retried). The chance its next group has
signal is E[1 − p^G − (1 − p)^G], in closed form. From a pool of 4B candidates we
choose B prompts by

    min  −Σ s_i x_i + λ Σ_{i<j} sim_ij x_i x_j,   Σ x = B

with sim = cosine of hashed n-gram features (near-duplicate prompts give correlated
gradients). The solve runs on a CPU thread while the GPU trains, so it costs no wall
time unless it is slower than a training phase (any wait is charged). Controls:
uniform (`base`), random subset of the pool (`qp_random`), and DAPO dynamic sampling
(`dyn`: oversample 2×, drop zero-signal groups after generating them).

**Lever 2, layer-block selection (cuts training only).** Each step trains k of L
blocks. On probe steps we measure per-block |g_i|² and the Hessian cross terms
M_ij = g_iᵀ H_ij g_j by finite differences (one extra backward pass per block). The
second-order loss change of updating set S is a QUBO,
dL(S) = −η Σ_S |g_i|² + η²/2 Σ_{S×S} M_ij. Frozen blocks skip their weight-gradient
GEMMs and the optimizer update; autograd stops below the lowest trained block.
Controls: `topk` (separable: ignores M), `random`. By §2 this lever alone is not
expected to pass; it is in the plan because the surpass plan names it, and the
measured interaction strength tells us whether SQA even has room to differ from greedy.

## 4. What the CPU checks already show (`validate_ql7.py`, 12/12)

* On random interacting QUBOs (n = 14, k = 5), SA and SQA hit the exact optimum in
  8/8 instances, greedy + swap in 6/8, plain greedy in 3/8. On separable instances
  greedy is exactly optimal. So the head to head is meaningful, and only when M ≠ 0.
* Finite-difference cross terms match the exact Hessian on a quadratic; frozen blocks
  get no gradients; GRPO loss gradients and clipping match the formula.
* **Warning from the toy task** (residual-MLP policy, 64 prompts, no transfer between
  prompts; 4 seeds, 60 steps, same rollouts):

  | arm | final expected reward | zero-signal groups |
  |---|---|---|
  | uniform (base) | 0.82 | 0.60 |
  | qp random pool | 0.73 | 0.59 |
  | qp greedy+swap | 0.49 | 0.76 |
  | qp SQA | 0.52 | 0.73 |
  | ql top-k (2 of 4 blocks) | 0.95 | 0.53 |
  | ql greedy+swap | 0.96 | 0.52 |
  | ql SQA | 0.96 | 0.52 |
  | ql random | 0.94 | 0.49 |

  On a *stationary* bandit the prompt selector cuts zero-signal groups from 22% to 5%,
  but on the toy policy it does worse than uniform: each update moves a trained prompt
  from mixed to all-correct before it is seen again, so the posterior is stale, and
  prompts that are skipped never improve because nothing transfers between them. Real
  GSM8K RL at lr 1e-6 moves pass rates much more slowly and transfers across
  problems, but this is exactly the failure to look for. Hence stage A1 below measures
  **persistence** (how often a zero-signal prompt is zero-signal again on revisit)
  from baseline logs alone, before any selection arm is paid for.
  The layer arms beating uniform here is a toy effect (half the blocks frozen acts
  like a smaller step size), not evidence for GPU savings.

## 5. Run plan and cost (1× H100 SXM Secure ≈ US$3.99/h, 4× ≈ US$15.96/h)

Per 0.6B run (200 steps): rollouts ~6–8 s, training ~5–8 s, eval of the full test set
every 20 steps (~11 × 30 s): about 1 h. Estimates, not measurements.

| stage | pods | what | wall | cost |
|---|---|---|---|---|
| QUICK | 1× H100 | CPU checks, smoke, vLLM weight-sync check, 10-step base/qp/ql/TRL runs | 20–30 min | ~US$2 |
| A1 | same pod | `base_s0` + `trl_ref_s0`, 200 steps each: phase fractions, persistence bound, TRL speed check | 2–2.5 h | ~US$8–10 |
| A2 | 4× H100 | `base_s1,s2`, `dyn`, `qp_random`, `qp_greedy_swap` ×3, `qp_sqa` ×3, `ql_{topk,greedy_swap,sqa,random}`, then the gate | 4–5 h | ~US$65–80 (cap 330 min ⇒ ≤ US$88) |
| B | later | Qwen3-4B-Base, winning arm vs base, 3 seeds each | — | costed only after A2 passes |

**Decision points.** After A1: if skipping last-time-zero-signal prompts would save
< 25% of rollouts, or wrongly skips > 10%, the prompt lever cannot pass → stop and
publish (total ~US$12). After A2: PASS → cost stage B; FAIL → negative result.
This is far below the surpass plan's original QL7 estimate (~US$500).

Launch (per the standing rule, results go to `chienhsinlin/qweave-runs`, then the
pod is deleted): `QUICK=1` then `STAGE=A1` on one 1× H100 pod via the usual wrapper;
`STAGE=A2` on a 4× H100 pod; wrap each in `timeout <minutes>m`.
