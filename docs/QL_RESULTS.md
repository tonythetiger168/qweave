# QL1.5–QL4 Measured Results (RunPod, 2026-10-28)

> All numbers from `scripts/runpod_all_phases.sh` on **1× NVIDIA H100 80GB HBM3**
> (RunPod, torch 2.8, transformers 5.21), model **Qwen/Qwen3-0.6B**.
> Raw JSON + logs: HF dataset `chienhsinlin/qweave-runs` (private), folders
> `runs/vbhwminemznaek` (full run) and `runs/0il1br2zf5k25e` (follow-up run).
> Every row below is reproducible with the command line stored in its report.
> Negative results are kept on purpose.

## 0. Phases I–III (quantum simulators) — all green on GPU

| Suite | Checks | Result |
|---|---|---|
| `phase1_basic/validate_p0.py` (vs naive + Stim) | 14 | PASS |
| `phase2_noise/validate_p0b.py` (IR/QASM, trajectories, DEM+PyMatching) | 22 | PASS (was 2 FAIL) |
| `phase3_scale/validate_p0c.py` (backends, MPS, IPEA, VQE, torch) | 27 | PASS (was crash on GPU host) |
| `phase4_llm/validate_ql1.py` (TT/MPO engine) | 14 | PASS |

**Phase II bug fixed** (`phase2_noise/detector_model.py`): PyMatching `fault_ids`
are *observable* indices, but every fault was registered as its own observable
and the 0/1 decode vector was read back as fault ids. Also no final data-qubit
detectors, and ancillas are never reset so `m_r = parity_r ⊕ m_{r-1}` was not
undone. After the fix all faults are graph-like (max 2 detectors), 33/33
single faults decode correctly, and d=3 repetition memory:

| p | undecoded | decoded (before) | decoded (now) |
|---|---|---|---|
| 0.001 | 0.65 % | 2.0 % (worse!) | **0.00 %** |
| 0.005 | 2.10 % | 10.3 % | **0.05 %** |
| 0.02 | 7.70 % | 29.9 % | **1.65 %** |

## 1. QL1.5 — compression + heal

Candidate layers: q/k/v/o/gate/up/down in all 28 blocks (196 Linear, 440 M of
596 M params; embeddings untouched). Heal: 1500 steps × 8 × 512 tokens,
AdamW lr 2e-5, only TN factors + norms trainable, KD = 0.5·CE + 0.5·KL(teacher).
PPL = WikiText-2 test, 1024-token windows.

### 1a. Format matters more than anything else (−30 % of candidate params)

| Format | PPL before heal | PPL after heal | Train tok/s | Notes |
|---|---|---|---|---|
| legacy row\|col MPO (`--format mpo`, χ=32, 50 % of q/o/gate/up layers → −49.5 % of those) | 755 084 | 1 387 (300 steps) | 6 532 | χ caps matrix rank at 32 |
| raw-norm TT-matrix (`--format tt`, d=4) | 747 971 | 461 (100 steps) | 505 | einsum chain, 34× slower |
| **whitened 2-site MPO (`--format wsvd`)** | **199** | **16.4–17.0** | **16 800–23 300** | activation-aware; two GEMMs |

Lesson: trained LLM weights are *not* low-entanglement in the raw Frobenius
norm. They are in the norm the network sees, ‖W X‖ — whitening by the
calibration Gram matrix (SVD-LLM) makes truncation error exact in that norm.

### 1b. Size/quality curve and the domain trap

| Run | heal data | cand. saved | total saved | PPL before | PPL healed | vs dense 21.74 | vs dense fine-tuned same recipe |
|---|---|---|---|---|---|---|---|
| dense fine-tune | WikiText-103 | 0 | 0 | 21.74 | 13.71 (150 steps) | −37 % | — |
| s15 KD | WikiText-103 | 15 % | 11.1 % | 51.1 | 15.06 | −31 % | (+10 % vs 13.71*) |
| s20 KD | WikiText-103 | 20 % | 14.8 % | 74.7 | 15.88 | −27 % | (+16 %*) |
| s30 KD | WikiText-103 | 30 % | 22.2 % | 199.5 | 17.00 | −22 % | (+24 %*) |
| s30 CE only | WikiText-103 | 30 % | 22.2 % | 199.5 | 16.39 | −25 % | (+20 %*) |
| **dense fine-tune** | **FineWeb-Edu** | 0 | 0 | 21.74 | **18.24** | −16 % | — |
| **s15 KD** | FineWeb-Edu | 15 % | 11.1 % | 119.7 | **26.58** | +22 % | **+46 %** |
| **s20 KD** | FineWeb-Edu | 20 % | 14.8 % | 315.9 | **30.15** | +39 % | **+65 %** |
| **s30 KD** | FineWeb-Edu | 30 % | 22.2 % | 1771.4 | **40.07** | +84 % | **+120 %** |

\* dense WikiText fine-tune ran 150 steps only; 1500 would be lower still.

**Honest reading.** Healing on WikiText-103 and testing on WikiText-2 makes
compressed models look *better than dense* (17.0 < 21.7) — that is domain
adaptation, not compression quality: the dense model fine-tuned 150 steps on
the same data reaches 13.7. With out-of-domain heal (FineWeb-Edu) the
compressed 0.6B model is 22–84 % worse than dense. `Qwen/Qwen3-0.6B` is the
post-trained chat model; even plain-text fine-tuning lowers its PPL
(21.7 → 18.2), so `Qwen3-0.6B-Base` is the right PPL reference (QL5).
The QL1.5 gate (PPL ≤ +3 % vs dense, out-of-domain) is **not met** at
≥ 11 % total saving on a 0.6B model.

Heal itself works: 97–99 % of the compression gap is recovered every time.
Memory: TN-only heal at −30 % peaks at 12.4 GB vs 17.5 GB for dense full
fine-tune (−29 %).

## 2. QL2 — kernels and the "quantum annealing" allocator

### 2a. Layer throughput (fwd+bwd, 8192 tokens, bf16 autocast, H100)

| Layer | dense | TT r32 eager / compiled / materialize | low-rank keep 0.5 | low-rank keep 0.3 |
|---|---|---|---|---|
| q_proj 1024→2048 | 18.9 M tok/s | 0.40 / 0.91 / 4.26 M | 9.9 M (r=341) | 16.6 M |
| up_proj 1024→3072 | 14.5 M | 0.25 / 0.58 / 2.56 M | **16.6 M (r=384, 1.14×)** | 12.9 M |
| down_proj 3072→1024 | 13.7 M | 0.47 / 0.47 / 3.47 M | **15.4 M (r=384, 1.12×)** | 12.4 M |

TT-matrix layers are 4–300× slower than cuBLAS at these sizes; `torch.compile`
(Inductor/Triton) helps 2× but cannot close the gap. Low-rank (2-site MPO)
beats dense only when the rank is tensor-core aligned (384 ✓, 341/477 ✗) →
ranks are now rounded to multiples of 64 (`--rank-align`).

### 2b. Rank allocation: greedy vs SA vs SQA vs exact (196 layers × 6 options)

Synthetic instances shaped like Qwen3-0.6B, budget 60 %, 5 seeds:

| Method | gap to DP lower bound | time |
|---|---|---|
| greedy (error-drop per parameter) | **0.6–0.9 %** | 0.02 s |
| exact DP (fine grid, costs rounded up) | 1.0–1.3 % | 0.8 s |
| simulated annealing (256 replicas, cold start) | 6.3–8.2 % | 10.7 s |
| path-integral SQA (16 Trotter × 32, cold start) | 19–32 % | 45 s |

On real Qwen3-0.6B allocations (greedy-seeded) SQA ties greedy at 30 % and
is 51–55 % worse at 15–20 %. **For this separable knapsack, "quantum
annealing" is the wrong tool** — greedy is near-optimal and 2000× faster.
Annealing only earns its place once the objective has pairwise layer
interactions (QL7).

## 3. QL3 — entanglement diagnostics + QSA attention

Per-head attention effective rank exp(S) over 28 × 16 heads (2K tokens):
min 1.0, median 2.8, max 2044 — extremely heavy-tailed.

| N | dense FA-SDPA | QSA top-8 | speedup | QSA top-16 | speedup |
|---|---|---|---|---|---|
| 8 192 | 0.96 ms | 1.55 ms | 0.62× | — | — |
| 16 384 | 3.36 ms | 1.94 ms | 1.74× | 2.24 ms | **1.49×** |
| 32 768 | 12.60 ms | 3.19 ms | 3.95× | 3.65 ms | 3.47× |

Long-context quality (8 × 16K windows, WikiText-103 test):

| Mode | PPL | Δ vs dense |
|---|---|---|
| dense | 16.697 | — |
| QSA uniform top-8 | 17.330 | +3.79 % |
| QSA entanglement-adaptive (same mean k) | 17.842 | +6.86 % |
| **QSA uniform top-16** | **16.857** | **+0.96 %** |

**QL3 gate (≥ 1.2× at 16K+, ≤ 1 % PPL): PASS** with uniform top-16
(1.49× at 16K, 3.47× at 32K, +0.96 %). Entanglement-adaptive budgets are a
negative result: low-rank heads are not the ones that tolerate sparsity.

## 4. QL4 — open benchmark (dense vs compressed)

`phase4_llm/eval/run_eval.py`, lm-evaluation-harness, 500 examples per task
(MMLU: 500 per subject cap), seed 0, bf16. ± = harness stderr. Compressed
models are the FineWeb-Edu-healed KD runs from §1b (out-of-domain heal).

| Metric | dense Qwen3-0.6B | −11 % params (s15) | −22 % params (s30) |
|---|---|---|---|
| params | 596.0 M | 529.9 M | 463.9 M |
| ARC-easy (acc_norm) | 55.4 ± 2.2 | **55.0** ± 2.2 | 48.4 ± 2.2 |
| ARC-challenge | 32.8 ± 2.1 | 29.0 ± 2.0 | 26.0 ± 2.0 |
| HellaSwag | 47.6 ± 2.2 | **49.4** ± 2.2 | 46.2 ± 2.2 |
| PIQA | 69.0 ± 2.1 | 65.8 ± 2.1 | 64.4 ± 2.1 |
| WinoGrande | 55.8 ± 2.2 | **55.8** ± 2.2 | 54.6 ± 2.2 |
| MMLU (57 subjects) | 41.3 ± 0.4 | 28.4 ± 0.4 | 22.9 ± 0.4 |
| GSM8K (5-shot, strict) | 43.8 ± 2.2 | 9.8 ± 1.3 | 0.8 ± 0.4 |
| WikiText-2 PPL, clean | 23.84 | 27.61 | 40.29 |
| PPL, 5 % char noise | 124.7 | 159.1 | 226.5 |
| PPL, 10 % char noise | 290.2 | 376.8 | 524.4 |
| NLL gap noise 10 % (robustness) | 2.50 | 2.61 | 2.57 |
| long-context PPL (8K windows) | 17.98 | 20.67 | 32.66 |
| NLL by position 0–512 / 4096–8191 | 3.04 / 2.79 | 3.18 / 2.96 | 3.56 / 3.46 |

**Reading.** At −11 % params, commonsense/reading tasks are within noise of
dense (ARC-e, HellaSwag, WinoGrande), but **knowledge (MMLU −13 pt) and
multi-step math (GSM8K −34 pt) collapse first** — PPL alone (+16 %) hides
this. Robustness to input noise is unchanged (NLL gap 2.50 → 2.57–2.61);
long-context drift is preserved at −11 % and flattens at −22 % (the
compressed model gains less from longer context).

This is the central evidence for the surpass plan: *any* "+10–20 % on
MATH-500/GPQA" claim for a compressed small model needs public task
numbers, because the cheap metric (PPL) and the expensive one (reasoning)
move very differently.
