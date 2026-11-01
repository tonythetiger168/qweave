# Surpassing FermiQLLM 1.0 — Improvement Plan

> 2026-10-28 · Based on: [QL_RESULTS.md](QL_RESULTS.md) (measured on H100),
> [FERMIQLLM_GLOBAL_INVESTIGATION.md](FERMIQLLM_GLOBAL_INVESTIGATION.md),
> [QUANTUM_LLM_ROADMAP.md](QUANTUM_LLM_ROADMAP.md).

## 0. In one sentence

FermiQLLM 1.0's five "quantum-enhanced" stages are backed only by the company's
internal numbers. qweave has taken the same techniques (tensor-network compression,
simulated quantum annealing, entanglement-inspired structure) and **measured them on
the same Qwen3 base family, in public, negative results included**. The next step is
to use those lessons to beat it **on public leaderboards** with equal compute.

## 1. What the measurements taught us (2026-10-28, 1× H100, ~US$25)

| # | Finding | Data | What it means for FermiQLLM's claims |
|---|---|---|---|
| F1 | Raw-norm TN/MPO compression does not work on trained weights | Legacy MPO: PPL 21.7 → 755,084; TT-matrix: → 747,971, and training is 34× slower | "Tensor networks embedded in the model structure" without activation awareness is PR, not engineering |
| F2 | Only activation-whitened 2-site MPO (SVD-LLM style) is viable | −30% of candidate params: PPL 199 → heal recovers 97–99% of the gap | The real technical core is **activation-aware low rank + distillation heal** |
| F3 | In-domain heal makes a compressed model "look better than the original" | WikiText heal: 17.0 < dense 21.7; but dense fine-tuned on the same data for 150 steps reaches 13.7 | "+10–20% over same-size models" may just be domain adaptation unless compared with **a dense model fine-tuned with the same recipe** |
| F4 | PPL hides the collapse of reasoning | −11% params: PPL +16%, commonsense flat, **MMLU −13 pt, GSM8K −34 pt** | MATH-500 / GPQA claims must publish per-question results; compressing a small model hurts math and knowledge first |
| F5 | "Quantum annealing" is the wrong tool for separable rank allocation | greedy 0.6–0.9% from the lower bound (0.02 s); SA 6–8%; SQA 19–32% (45 s) | Annealing can only add value on objectives **with interactions between layers**, and that must be shown |
| F6 | Low-rank layers are faster only when the rank is aligned to tensor cores | r=384: 1.14× dense; r=477: 0.39× | "+15% inference" depends on kernel details, not physics |
| F7 | Quantum-inspired sparse attention really works at long context | 16K: 1.49×, PPL +0.96%; 32K: 3.47× | A direction FermiQLLM **does not claim**, and one where we already pass the gate |
| F8 | "Entanglement entropy sets the sparsity budget" is a negative result | adaptive +6.9% vs uniform +3.8% (same mean k) | Physics metaphors must be measured, not used directly as design rules |

## 2. Stage by stage: FermiQLLM claim → qweave today → target to beat it

| Stage | FermiQLLM 1.0 | qweave today (measured) | Target (verifiable) |
|---|---|---|---|
| Model structure | TN embedded, inference +15% | wsvd 2-site MPO with aligned ranks; TT/MPO negative results published | QL5/QL6: −25% params, MMLU −≤1 pt, decode ≥1.3× |
| Quantum annealing | Used for optimization | SQA / SA / greedy / DP fully compared; conclusion: use DP/greedy for separable problems | QL7: use it only on objectives with interaction terms, head-to-head with greedy |
| Training / RL | RL cost −25% | Distillation heal; TN-only heal uses 29% less VRAM | QL7: GRPO at the same score with ≤0.75× GPU-hours |
| Attention | Not mentioned | QSA 1.49× at 16K with +0.96% PPL (gate passed) | QL6: integrate into vLLM prefill, ≥2× at 32K+ |
| Evaluation | Internal +10–20%, no weights | lm-eval + robustness suite, all JSON public | QL8: public leaderboards + weights + one-command reproduction |

## 3. Improvement plan (QL5–QL8), each item derived from a measured lesson

### QL5 — Keep knowledge and reasoning (fixes F3/F4)
- **Base model:** use `Qwen3-0.6B-Base` / `Qwen3-4B-Base` as the PPL reference (fine-tuning the chat model lowers its PPL by itself, see F3).
- **Knowledge-aware sensitivity:** mix MMLU-style knowledge text and GSM8K-style reasoning chains into calibration (no overlap with the eval sets),
  and change the rank-allocation objective from "activation error" to "activation error × the layer's Fisher importance".
- **Longer heal:** ≥100M tokens (currently 6M), sweep KD temperature and mix ratio, upweight reasoning data.
- **Combine with quantization:** low rank + INT8/FP8 (for small models the Pareto front is usually set by quantization, so low rank only matters stacked on top of it).
- **Acceptance** (out-of-domain): at −20% params, MMLU −≤2 pt and GSM8K −≤5 pt, with a dense model fine-tuned with the same recipe as the control.
- Cost: 4× H100, about US$300–600.

### QL6 — Actually get faster (fixes F6, builds on F7)
- Always align ranks to 64/128 (implemented as `--rank-align`), and fuse `x·Bᵀ·Aᵀ` into one Triton kernel.
- Measure tokens/s for batch-1 decode (memory-bandwidth bound): the only regime where low rank / TN has a natural advantage.
- Wire QSA into vLLM long-context prefill, with top-16 as the default (16K: +0.96% PPL).
- **Acceptance:** decode ≥1.3× dense (vs FermiQLLM's "inference +15%"), 32K prefill ≥2×.

### QL7 — Put "quantum annealing" on the right problem (fixes F5/F8)
- First measure interactions between layers: Hessian cross terms / non-additive PPL changes from compressing pairs of layers; use SQA only where interactions are significant.
- RL: in GRPO, write "which layers / rank directions to update this step" as a QUBO with interaction terms, and run SQA vs greedy vs full AdamW head-to-head.
- **Acceptance:** ≤0.75× GPU-hours at the same final score; otherwise publish it as a negative result.

### QL8 — Public, reproducible, on the leaderboards
- Weights, configs and logs all on HF; `scripts/runpod_all_phases.sh` reruns everything with one command.
- Submit to public evaluations such as the Open LLM Leaderboard, with an item-by-item comparison against FermiQLLM's public material.
- Quantum-hardware interface: keep the Phase II DEM + PyMatching pipeline (now fixed) for future QPU subroutine experiments;
  until there is provable quantum advantage, always say *quantum-inspired*.

### Additional items (2026-10-28, gaps found by checking FermiQLLM's five stages)

FermiQLLM's "gauge degrees of freedom" and "data representation" stages are not yet
covered by QL5–QL8, and the evaluation must avoid being pulled into an "accuracy race"
we are not competing in.

- **G1 — Gauge fixing (part of QL5):** an MPO / low-rank factorization has a gauge
  freedom on every bond (an `A·A⁻¹` can be inserted). Fix it to canonical form with QR
  sweeps (porting `MPSCanonical`) and balance the core scales in whitened coordinates.
  Acceptance: heal needs ≤0.8× the steps to reach the same PPL as the free gauge, *or*
  the PPL gap after stacking INT8/FP8 shrinks by ≥20%; if neither holds, report it as a
  negative result. This turns the other side's unexplained term into a testable claim.
- **D1 — Data representation: embedding / LM-head compression (part of QL5):** Qwen3-0.6B's
  151,936 × 1,024 embedding is about 26% of its parameters. Following F1/F2, do not use
  raw-norm TT; use **token-frequency-weighted** low rank (the embedding-layer equivalent of
  activation whitening) with one shared factorization for the tied head. Acceptance:
  ≥2× compression of that matrix with out-of-domain MMLU −≤1 pt.
- **E2 — Quality–compression Pareto (part of QL8):** FermiQLLM claims an **accuracy gain**,
  while TN compression inherently trades accuracy for efficiency. Present the comparison as
  Pareto curves (params / tok/s / VRAM vs score) rather than one headline number; any claimed
  "gain" needs ≥3 seeds and a paired-bootstrap confidence interval.
- **Named baselines (required in the QL8 comparison table):** uncompressed Qwen3-0.6B/4B-Base,
  a dense model fine-tuned with the same recipe (F3), a plain truncated-SVD control with the
  same parameter count, and published CompactifAI / Minima numbers (where comparable).

## 4. Why this can "surpass" it

1. **Verifiable:** FermiQLLM has 0 papers, 0 weights, 0 third-party reproductions. Every number of ours comes with a command and JSON.
2. **Same base, same-recipe controls:** F3 shows that "stronger than same-size models" is easily an illusion caused by domain adaptation; we get the control right.
3. **Negative results published:** F1, F5 and F8 are exactly where "quantum" narratives exaggerate most; publishing them is our credibility moat.
4. **A new direction that already passes its gate** (F7, QSA), which FermiQLLM does not claim.

## 5. Timeline and cost

| Stage | Hardware | Estimated cost | Duration |
|---|---|---|---|
| QL1.5–QL4 (this PR, 0.6B, done) | 1× H100 | ~US$25 | Done |
| QL5 knowledge preservation + 4B | 4× H100 | US$300–600 | 2–3 weeks |
| QL6 kernels + vLLM | 1× H100 | ~US$50 | 2 weeks |
| QL7 RL + interacting QUBO | 4× H100 | ~US$500 | 3–4 weeks |
| QL8 public leaderboards | — | Small | Ongoing |
