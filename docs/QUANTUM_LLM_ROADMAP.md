# Quantum-Enhanced LLM: Technical Analysis & CUDA Roadmap

> 中文摘要：本文分析費米宇宙 FermiQLLM 1.0（2026-09-28 發布）的技術主張，對照已發表的
> 量子啟發（quantum-inspired）文獻逐項評估，並提出以本套件（qweave）為驗證骨幹的
> 開源可復現路線圖（QL1–QL4），核心為 CUDA 訓練整合。

## 1. What FermiQLLM 1.0 Claims

Source: 36氪 / 北京商報 / 騰訊新聞 (2026-09-28). Tsinghua-affiliated startup 費米宇宙
(seed round ~¥100M, valuation ~¥1B), base = Qwen open-source model.

| # | Claim | Detail |
|---|-------|--------|
| C1 | "全鏈路量子增強" across 5 stages | data representation, model structure, training, RL, evaluation |
| C2 | Tensor networks + simulated annealing + gauge DOF | embedded into *classical* GPU pipeline; no quantum hardware |
| C3 | Inference +15%, RL training cost −25% | vs same-parameter traditional model |
| C4 | MATH-500 / GPQA-Diamond / BBH +10–20% | vs same-size open-source base |
| C5 | "全球首款" full-chain quantum-enhanced LLM | marketing superlative |

## 2. Critical Assessment

| Claim | Published prior art | Verdict |
|-------|--------------------|---------|
| C2 tensor-network compression | **CompactifAI** (Multiverse/Orús): LLaMA-2 7B → MPO, −70% params, −50% train time, −25% inference, 2–3% acc drop after healing. **Minima** (arXiv 2602.01613, 2026): Qwen3-32B mixed Tucker/TT/TR + sensitivity model + Triton kernels + spec decoding: −37% VRAM, ~1.9× TPS | **Real technique, rebranded.** Not novel physics — applied math |
| C2 quantum simulated annealing | **GTQA** (arXiv 2409.12240): tensor-network simulation of quantum annealing on classical hardware for QUBO; SA-style global optimizers for RL are well known | **Plausible**, but "quantum annealing" here = classical simulation of it. No quantum speedup |
| C1/C5 "全鏈路" / "全球首款" | Multiverse, TensorLLM, LightFormer, MLoRQ, quantum LLM hybrids all predate 2026 | **Marketing.** Quantum-*inspired* classical computing, not quantum enhancement in the physics sense |
| C3/C4 numbers | Self-reported ("據內人士透露"), no public weights, no tech report, no peer review. Some secondary articles contain fabricated embellishments (e.g. "GTC added a forum", "MIT journal special issue", MATH-500 78.4% beating GPT-5 Turbo) | **Unverified** — treat as press-release figures |

**Bottom line**: the honest core is "quantum-inspired tensor-network compression + annealing-style
optimization on GPU", which is legitimate and publishable — but the "quantum enhancement" framing
is branding. There is no quantum hardware, no quantum advantage, and no disclosed mechanism that
wouldn't be described in the TN-compression literature above.

### What they got directionally right
- Betting on TN compression of attention/MLP weights is sound (Minima shows 2× VRAM + 2× TPS on 32B).
- "No quantum hardware needed" is the correct commercial call for 2026.
- Entanglement-structured sparsity in attention is an open research direction worth pursuing.

### Where an open project can beat them
- **Reproducibility**: open weights, open harness, third-party verification. Their numbers cannot be checked.
- **Honest scope**: claim "quantum-inspired TN compression" (defensible) instead of "quantum enhancement".
- **Validation backbone**: use qweave's simulators to ground-truth every quantum-mechanical
  metaphor (entanglement spectra of attention maps, annealing energy landscapes) instead of using
  them purely as inspiration.

## 3. Roadmap QL1–QL4 (CUDA training integrated)

Leverages existing suite modules. All targets reproducible on 1–8 GPUs.

### QL1 — TN Compression Engine (Month 1) ✅ DONE (2026-10-28)
**Delivered**: `phase4_llm/tn_layers.py` + `validate_ql1.py` — 14/14 checks passing.
- `mpo_decompose`: TT-SVD of (out,in) weights into (chi,2,chi) cores with adaptive eps
  truncation ported from `MPSCanonical`; `MPOLinear` drop-in nn.Linear replacement
  (exact forward to 1e-7 incl. non-power-of-2 dims); `sensitivity_profile` for
  per-layer chi budgeting.
- Measured: 70.6% parameter cut on structured weights (chi<=64, eps=1e-3);
  error decreases monotonically with chi; eps budget sweep 26→61 max bond.
- **Physics lesson (recorded)**: TT/MPO rank across bit-cuts is NOT matrix rank —
  a rank-16 matrix can have large TT-rank. Compression works when weights have low
  entanglement across bit cuts (trained weights do — cf. CompactifAI LLaMA results).
- **Still open (moved to QL1.5)**: LoRA-style heal fine-tune on a real 0.6B model —
  needs GPU env (single 24GB card); see QL2.

### QL1.5 — Heal Fine-tune ✅ READY (2026-10-28, awaiting GPU run)
Delivered: `phase4_llm/heal_finetune.py` (sensitivity-ranked compression, PPL eval,
AdamW heal, JSON report + acceptance gate) + `scripts/runpod_setup.sh` +
`docs/QL1.5_RUNPOD.md` (RTX 4090, est. cost < $1). Smoke mode passes on CPU
(heal improves PPL −2.4% on the tiny test model). Next: run on Qwen3-0.6B.

### QL2 — CUDA Training at Scale + Annealing Optimizer (Month 2)
**Goal**: make TN training fast and add the "quantum annealing" component — classically.
- **Triton TN matmul kernel**: contracted MPO-core × activation without materializing the weight
  (the key to beating dense speed; Minima's stage-4 equivalent).
- **Distributed**: FSDP over 4–8 GPUs; measure GPU-hours vs uncompressed baseline on 1–4B model,
  target ≥ 1.3× training throughput from reduced comm/memory traffic (CompactifAI-style effect).
- `phase4_llm/anneal_optimizer.py`: GPU quantum-simulated-annealing fine-tune for RL — QUBO
  relaxation of low-rank update selection, annealing schedule executed on GPU (tensorized state,
  our `trajectory.py` channels as the "thermal bath" sampler; see arXiv 2409.12240 for the
  TN-annealing formalism). Acceptance: matches/beats AdamW on a small RLHF/GRPO run at ≤ 0.7× GPU-hours.
- **CUDA specifics**: bf16 params + fp32 MPO cores, CUDA graphs for the einsum chain, pin memory,
  overlap comm; benchmark matrix recorded in `docs/QL2_BENCHMARKS.md`.
- **Gauge fixing (G1, see §3.1)**: canonicalize MPO cores (left/right/mixed gauge via QR,
  ported from `MPSCanonical`) before heal and before low-bit quantization; compare free-gauge
  vs fixed-gauge heal on loss curve and post-quant PPL.
- **4B checkpoint (§3.1 S1)**: run the full compress→heal→eval chain on Qwen3-4B, the same
  scale as FermiQLLM's March-2026 pilot.

### QL3 — Entanglement Diagnostics & QSA Attention (Month 3)
**Goal**: turn the physics metaphor into a measurable, useful mechanism.
- Entanglement spectrum of attention maps: run attention matrices through `MPSCanonical`;
  rank-χ MPS fit of head activations; show which heads are low-entanglement (compressible).
- **QSA (quantum-inspired sparse attention)**: prune/gate attention edges by measured entanglement;
  implement as CUDA kernel (block-sparse flash attention variant).
- Acceptance: ≥ 1.2× long-context (16K+) throughput at ≤ 1pt quality drop vs dense FA2 baseline.
- **Data-representation stage (§3.1 D1)**: TT-embedding for the token embedding table and
  tied LM head (Qwen3 vocab 151,936 × d), the largest single matrices in small models.

### QL4 — Open Benchmark & Reproducibility Harness (Month 4)
**Goal**: publish what FermiQLLM didn't.
- `phase4_llm/eval/`: LLM-eval-harness compatible runner (MMLU, GSM8K, MATH-500, BBH, GPQA-Diamond)
  + our own **QRB-style robustness suite** (input noise, prompt perturbation, long-context drift).
- Head-to-head table vs uncompressed baseline and (where legal) reproduced CompactifAI/Minima
  configurations — every number reproducible from repo + config.
- **FermiQLLM head-to-head (§3.1 E1/E2)**: MATH-500 / GPQA-Diamond / BBH with named baselines,
  pinned harness version, seeds and configs; quality-vs-compression Pareto plot.
- Tie-in: Clifford sampler generates device-level error models for *future* quantum-hardware
  offloading experiments (keeps the suite's original mission).

## 3.1 Beat-FermiQLLM Plan (added 2026-10-28)

**Goal**: match or beat every FermiQLLM 1.0 claim *with numbers anyone can re-run*. Their
claims are internal, baseline-free and unpublished (see
[FERMIQLLM_GLOBAL_INVESTIGATION.md](FERMIQLLM_GLOBAL_INVESTIGATION.md)); ours ship with
weights, configs and logs. We win on verifiability first and on numbers second.

### Claim-by-claim scoreboard

| FermiQLLM stage / claim | qweave answer | Phase | Status |
|---|---|---|---|
| Model structure: tensor networks | `MPOLinear`, adaptive-χ TT-SVD | QL1 | ✅ done |
| Training: heal after compression | `heal_finetune.py` | QL1.5 | ⏳ GPU run |
| RL: "quantum simulated annealing", −25% RL cost | GPU annealing optimizer, target ≤ 0.7× GPU-hours (−30%) | QL2 | planned |
| "Gauge degrees of freedom" (never explained) | **G1** gauge-fixed MPO heal + quant | QL2 | **new** |
| Data representation | **D1** TT-embedding + tied LM head | QL3 | **new** |
| Inference +15% (speed or quality unstated) | Triton MPO matmul (speed) + QSA attention; report *both* tok/s and quality | QL2–QL3 | planned |
| Evaluation: MATH-500/GPQA-D/BBH +10–20% | **E1** head-to-head with named baselines | QL4 | **new** |
| 4B pilot scale | **S1** Qwen3-4B checkpoint | QL2 | **new** |

### New work items

- **G1 — Gauge fixing.** An MPO has a gauge freedom on every bond (insert `A·A⁻¹`). Fixing it
  (canonical form via QR sweeps, as in `MPSCanonical`) makes heal gradients well-conditioned and
  bounds per-core scale for quantization. Acceptance: fixed-gauge heal reaches the free-gauge
  PPL in ≤ 0.8× steps, *or* lowers int4 PPL gap by ≥ 20%. Publish the negative if neither holds.
- **D1 — TT embedding / LM head.** In Qwen3-0.6B the 151,936 × 1,024 embedding is ~26% of
  params. Acceptance: ≥ 4× compression of that matrix at ≤ 0.5 PPL increase after heal.
- **S1 — 4B checkpoint.** Qwen3-4B through compress → heal → eval on 1× H100 or 2× A100.
  Gives a like-for-like size point against FermiQLLM's pilot.
- **E1 — Head-to-head eval.** MATH-500, GPQA-Diamond, BBH (plus MMLU, GSM8K) via
  lm-evaluation-harness, pinned version, fixed seeds, greedy decoding, configs in repo.
  Baselines, all named:
  1. Uncompressed Qwen3-0.6B / Qwen3-4B (same base family FermiQLLM uses).
  2. Same models at equal parameter count by plain truncated-SVD (non-TN control).
  3. Published CompactifAI / Minima numbers where comparable.
- **E2 — Quality-vs-compression line.** FermiQLLM claims *accuracy gains*; TN compression
  trades accuracy for efficiency. Report a Pareto curve (params, tok/s, VRAM vs score) instead
  of a single headline number, so we are judged on the metric we actually target. Any
  claimed accuracy *gain* must survive ≥ 3 seeds and a paired-bootstrap CI.

### Win conditions

1. **Verifiable**: every number reproducible from one command + pinned config (they have none).
2. **RL cost**: annealing optimizer ≤ 0.7× GPU-hours vs AdamW at equal reward (beats −25%).
3. **Inference**: ≥ 1.15× tok/s *and* stated quality delta at equal params (their +15%, made precise).
4. **Benchmarks**: within 1 pt of uncompressed base at ≥ 40% fewer params, with CIs.
5. **Disclosure**: tech report on arXiv, including negative results.

## 4. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| TN matmul slower than cuBLAS at small χ | Keep dense fallback per layer (sensitivity router) |
| Accuracy drop beyond budget | Heal longer + mixed dense/TN layers (Minima finding) |
| GPU scarcity in dev env | Phase-gated: QL1 on single 24GB card; QL2 needs 4×A100 (cloud) |
| Head-to-head framed as accuracy race we can't win | Lead with E2 Pareto + verifiability; never claim gains without CIs |
| "Quantum-washing" criticism | Scope claims to "quantum-inspired", cite prior art, publish negatives |

## 5. References

- FermiQLLM 1.0 coverage: 36氪 2026-09-27; 北京商報/搜狐 2026-09-28; 百度百科 entry.
- CompactifAI (Multiverse Computing, ESANN 2025; Orús et al.) — MPO LLM compression.
- Minima: A Practical TN Compression Pipeline for Production-Scale LLMs (arXiv 2602.01613, 2026).
- GTQA: Large-scale quantum annealing simulation with tensor networks (arXiv 2409.12240).
- TensorLLM; LightFormer; MLoRQ — related TN-Transformer work.
