# QL6: make the compressed model actually faster

> 2026-10-29 · Code: `phase4_llm/ql6_decode.py`, `validate_ql6.py` (9 groups, all
> PASS on CPU) · GPU runner: `scripts/runpod_ql6.sh` · Plan:
> [FERMIQLLM_SURPASS_PLAN.md](FERMIQLLM_SURPASS_PLAN.md) §3 QL6, F6, F7.
> **No GPU result yet.** The ceilings below are computed from model shapes, not measured.

## Gate

| claim | gate | baseline | how measured |
|---|---|---|---|
| batch-1 decode | **≥ 1.3×** tokens/s | dense model, **same engine** (and, for INT8 or fused variants, dense with the same INT8 / fusion) | 24 fixed prompts, 256 new tokens, decode time = t(256) − t(1) so prefill is excluded; 95 % paired-bootstrap CI over prompts |
| 32K-token prefill | **≥ 2×** | dense SDPA, whole model | end-to-end forward, median of 3 |

A lossy variant (route L1) only counts if its model also passed the QL5
quality gate. A lossless variant (route L2) must reproduce ≥ 99 % of the dense
model's greedy tokens. Shape-matched random copies (`synthetic_*`) are
speed-only and can never pass the gate. "PASS" is on the point estimate;
"PASS at CI bound" is the claim we would make in public.

FermiQLLM's "+15 % inference" names no baseline, engine or batch size; this
gate names all three.

## The ceiling, before any GPU money is spent

At batch 1 every generated token reads every weight once, so decode speed is
bounded by bytes read per token. `decode_bytes()` counts them per module
(tied embedding = one row, LM head = the full matrix, KV cache at the context
length). Ceiling = dense bytes / compressed bytes, at 2K context:

| model | layer keep | embedding keep | params removed | ceiling vs dense bf16 | with INT8, vs dense INT8 |
|---|---|---|---|---|---|
| Qwen3-0.6B | 0.9 | 0.5 | 19 % (**the QL5 plan**) | **1.19×** | 1.16× |
| Qwen3-0.6B | 0.8 | 0.5 | 26 % | 1.28× | 1.23× |
| Qwen3-0.6B | 0.6 | 0.5 | 44 % | 1.57× | 1.46× |
| Qwen3-4B | 0.9 | 0.5 | 15 % | 1.17× | 1.16× |
| Qwen3-4B | 0.8 | 0.5 | 23 % | 1.29× | 1.28× |
| Qwen3-4B | 0.6 | 0.5 | 41 % | 1.64× | 1.61× |

(`python phase4_llm/ql6_decode.py --roofline` prints the full table, including
0 and 8K context.)

**What this means.** At QL5's −20 %, low rank alone cannot reach 1.3× on
either model: the hard ceiling is ~1.2×, and real kernels usually reach
85–95 % of it. That is roughly FermiQLLM's "+15 %", not more. Getting past
1.3× needs one of:

1. **more compression** (about −30 % or more of all params), which QL5 has not
   shown can keep quality;
2. **INT8 on top**: compressed + INT8 vs dense bf16 is 2×, but against dense
   INT8 (the fair baseline, which the gate uses) the gain shrinks back to the
   low-rank ratio;
3. **route L2, lossless speculative decoding**: the dense model verifies
   several tokens drafted by a compressed model in one forward. Quality is the
   dense model's by construction, so it does not depend on QL5 passing.

So QL6 measures all three, and the most likely way to pass the gate honestly
is L2 on 4B.

## What the code does

| piece | function | why |
|---|---|---|
| R1 byte roofline | `decode_bytes`, `roofline_table`, `meta_qwen3` | ceilings above, from published Qwen3 configs on the meta device |
| F6 rank alignment | `align_ranks_`, `rank_audit` | measured: r=384 1.14× dense, r=477 0.39×. Truncation after re-gauging is the optimal rank-r cut (Eckart–Young); padding is exact. QL5 already allocates aligned ranks, this fixes models built elsewhere |
| K1 sibling fusion | `fuse_siblings_` | gate_proj and up_proj read the same x, so their B factors (or dense weights) become one GEMM. Exact. Applied to dense too |
| CUDA graphs | engine `compiled` (HF static cache + `torch.compile` reduce-overhead) | removes per-kernel launch overhead, which dominates small models in eager mode |
| L1 speed without QL5 | `synthetic_lowrank_` | speed depends only on shapes, so random factors of the right ranks measure L1's speed today |
| L2 lossless | `speculative_generate` (dynamic cache), `speculative_static` (static cache, two fixed shapes so both models replay CUDA graphs) | greedy output equals the target's greedy output; tested exactly in fp32 for perfect, noisy and random drafts |
| real INT8 | `as_linear_pairs_` + torchao | `fake_quant_` in QL5 gives no speed; torchao's weight-only INT8 kernels do |
| prefill | `time_prefill` with QSA top-16 via the HF attention interface | QL3 measured the attention kernel alone (3.47× at 32K); the gate is end to end |

**Why not a single Triton kernel for x·Bᵀ·Aᵀ** (the original plan): at decode
the r-vector t = x·Bᵀ is needed by every output tile, so one kernel must
either recompute it per tile or synchronise the whole grid. The launch overhead
a fused kernel would save is removed by CUDA graphs instead. If the measured
L1 speed falls well short of the ceiling with CUDA graphs on, a kernel is the
next thing to try.

**vLLM.** The plan also said "QSA in vLLM prefill". vLLM cannot load the
custom low-rank layers without a model plugin, so this PR measures prefill in
HF and runs dense vLLM only as a reference point (how far HF + CUDA graphs is
from a production engine). A vLLM plugin is follow-up work, worth doing only
if the HF numbers pass.

## Named baselines in every report

- dense, eager and CUDA graphs; dense + fusion; dense + torchao INT8
- off-the-shelf drafts for 4B: `Qwen3-0.6B-Base`, `Qwen3-1.7B-Base` (the standard speculative-decoding baseline)
- HF's own assisted generation (`assistant_model=`), to check our loop is not slower than the library's
- dense vLLM, batch 1, same prompts (different engine, reference only)

## Exact commands

On a RunPod pod with **1× H100 80GB**, image `runpod/pytorch` (torch ≥ 2.4),
secret `HF_TOKEN` set (reports go to the private dataset `<you>/qweave-runs`):

```bash
# 0. check the pod (~15 min, ~US$1): every step at toy size
curl -sL https://raw.githubusercontent.com/tonythetiger168/qweave/claude/ql6-decode/scripts/runpod_ql6.sh \
  | QUICK=1 bash

# 1. stage S: speed of every route, no QL5 result needed
curl -sL https://raw.githubusercontent.com/tonythetiger168/qweave/claude/ql6-decode/scripts/runpod_ql6.sh \
  | STAGE=S bash
```

Stage S trains its own QL-compressed 0.6B draft (QL5 recipe, 20 M heal
tokens, about 20 min); a draft only needs to be good enough to be accepted
often, not to pass QL5's gate.

Stage Q repeats the measurements on QL5's real models, so it runs **on the
QL5 pod right after QL5 stage A** (QL5 does not upload its `.pt` files):

```bash
R=https://raw.githubusercontent.com/tonythetiger168/qweave/claude/ql6-decode/scripts
curl -sL $R/runpod_ql5.sh | KEEP_POD=1 STAGE=A bash   # QL5 stage A, see QL5_RUN_PLAN.md
curl -sL $R/runpod_ql6.sh | STAGE=Q bash              # uses /workspace/ql5_runs/stageA
```

`KEEP_POD=1` (added to `runpod_ql5.sh` in this PR) stops the QL5 script from
shutting the pod down, so its models are still on disk for stage Q.

Single run by hand:

```bash
python phase4_llm/ql6_decode.py --model Qwen/Qwen3-4B-Base --synthetic-keep 0.9,0.6 \
    --engines eager,compiled --int8 --drafts Qwen/Qwen3-0.6B-Base --gamma 3,5 \
    --prefill 8192,16384,32768 --out /workspace/ql6/4b
```

Each run writes `ql6_report.json`: per-variant bytes/token and ceiling, rank
audit, tokens/s, speedup with CI (vs plain dense and vs matched dense),
acceptance rate, tokens per target call, draft cost, the predicted speedup
from the standard formula, token agreement with dense greedy, prefill times,
and the gate.

## Cost estimate

Assumes ~US$3 per H100-hour (check current RunPod pricing). Times are
estimates from HF decode speeds on H100; the logs print tokens/s per variant,
so a bad estimate shows up early.

| stage | what runs | wall time on 1× H100 | cost |
|---|---|---|---|
| QUICK | toy version of everything | ~15 min | ~US$1 |
| S | draft heal (20 min), 0.6B decode + prefill (~1.3 h), 4B decode + spec + prefill (~2.5 h), vLLM (~20 min) | ~4.5–5 h | **~US$12–20** |
| Q | QL5 models: L1 on 0.6B, L2 drafts for 4B, on the 4× H100 QL5 pod (one GPU used) | ~1 h | ~US$12 (4× H100 rate) |
| **total** | | | **~US$25–35** |

Below the plan's ~US$50 because speed is measured on shape-matched copies and
needs no training except the draft.

## Stage S results (1× H100 SXM, 2026-10-29)

torch 2.12.1, transformers 5.21.1, torchao 0.18. Raw logs and JSON: HF dataset
`chienhsinlin/qweave-runs`, `runs/wg2dxaw82efhjm_ql6/stageS`. The run hit its
5 h cap during the 4B step, so 4B γ=3, 4B prefill and the vLLM references did
not run.

| measurement | result | gate |
|---|---|---|
| 0.6B dense decode | 48.6 tok/s eager, 301 tok/s CUDA graphs | — |
| 0.6B low-rank (keep 0.9 / 0.8 / 0.6), CUDA graphs | **0.91× / 0.95× / 0.84×** dense | FAIL |
| 0.6B torchao INT8, batch 1 | dense 0.92×, low-rank 0.48–0.66× | FAIL |
| gate/up fusion | dense 1.08× (CUDA graphs), 1.13× (eager) | — |
| 0.6B QSA prefill vs SDPA | 8K 0.50×, 16K 1.10×, 32K **1.52×** | FAIL (needs 2×) |
| 4B dense decode | 149 tok/s CUDA graphs | — |
| 4B speculative, γ=5, CUDA graphs, draft Qwen3-0.6B-Base | **1.21× [1.12, 1.31]**, α 0.60 | FAIL (point < 1.3) |
| … draft Qwen3-1.7B-Base | 1.00×, α 0.69 | |
| … draft QL-compressed 0.6B (20 M-token heal) | 0.82×, α 0.40 | |

**What this says.**
- Low rank is *slower* than dense at batch 1 even though it reads 16–45 % fewer
  bytes: two skinny GEMVs per layer (M = 1, r = 384–704) run far below
  bandwidth, and the byte ceiling is not reached. This is the F6 lesson again,
  now at decode. Closing it needs a GEMV kernel that keeps t = x·Bᵀ on chip,
  which the PR argued against; the data now says it is the next thing to try.
- INT8 weight-only kernels in torchao do not help batch-1 decode on H100.
- Plain off-the-shelf speculative decoding (0.6B draft for 4B) gives 1.21×.
  Our compressed draft is worse than the plain 0.6B: its acceptance rate is
  lower and it is not cheaper per step (low rank is slower, above).
- The 32K prefill end to end is 1.52×, short of 2×: QL3's 3.47× was the
  attention kernel alone; MLPs and the selection pass do not shrink.

**Agreement 0.40–0.76 was not a bug.** The report's `agreement` was the
identical-prefix fraction against plain greedy in bf16. bf16 logits are
coarse (spacing 0.125 at |logit| 16–32), the batched verify forward rounds
differently from one-token forwards, so the first near-tie flips and every
later token differs. A 4-layer random Qwen3 on CPU shows the same thing:
0.3–0.5 in bf16, exactly 1.0 in fp32, and every emitted token within one bf16
step of the target's argmax under teacher forcing. The gate now uses
`lossless_report`: fp32 agreement (logic, must be 1.0) plus the teacher-forced
gap of each emitted token, compared with plain greedy's own gaps. The 4B rows
above need a rerun to get those numbers.

## Stage L: 4B speculative rerun (1× H100 SXM, 2026-10-30)

Raw: HF `chienhsinlin/qweave-runs`, `runs/7jcwnq6gkx9ppk_ql6/stageL`. CUDA graphs,
24 prompts × 256 tokens, draft Qwen3-0.6B-Base, γ = 5.

| row | result |
|---|---|
| dense | 160.8 tok/s |
| speculative | **0.81× [0.75, 0.88]**, α 0.59, 3.9 tokens per target call, draft cost c = 0.41 |
| standard formula with those α and c | 0.76× |
| HF assisted generation | 0.58× |
| lossless | every token within one bf16 step of the target's argmax (spec and greedy alike) |

Stage S's 1.21× did not reproduce. The measured speedup now agrees with the
formula. A 0.6B draft costs ~0.4 of a 4B step at batch 1 because both are
launch-bound (28 vs 36 layers, similar kernels per layer), and at c ≈ 0.4
speculative decoding cannot pay off at α ≈ 0.6. **Route L2 in this engine is a
negative result.**

Three harness bugs that this run exposed are fixed in this PR:
- only the last γ produced a row (stage S's "missing γ=3" was this bug, not the cap);
- greedy reference, draft cost and speculative time came from different compile
  sessions and draft cost from 4 prompts; now all three are timed per γ in one
  session (stage S's 1.21× probably came from this);
- the "fp32" check ran with TF32 on (main enables it), which is why fp32
  agreement was 0.89; it now disables TF32 and records the fp32 logit gap at
  every divergence, so a remaining flip is shown to be a tie or not;
- (also) the smoke prefill used 32-token QSA blocks, which flex attention
  rejects on CUDA.

## Stage K: Triton low-rank decode kernels

Stage S showed the problem is not bytes but kernels. Dense 0.6B under CUDA
graphs ran at 3.3 ms/token, about 11 % of the H100 byte floor (~0.36 ms), so
decode time tracks the number of kernels, and torch low rank doubles the
GEMMs per Linear. Its skinny GEMVs (r = 384-704 outputs) also occupy only a
few dozen of the 132 SMs.

`phase4_llm/ql6_kernels.py` (M <= 16 rows: decode and speculative verify):

| kernel | does | why |
|---|---|---|
| `lowrank_b` | t-partials = x·Bᵀ, split along K | ~4x more programs; partials are summed by the next kernel while it loads t, so no atomics and no memset kernel |
| `lowrank_a` | y = A·t (+ bias) | |
| `lowrank_a_glu` | h = silu(A_g·t_g) * (A_u·t_u) | gate/up's two A factors and SwiGLU in one kernel |

Kernels per MLP: torch low rank 6, these 4, dense (fused gate/up) 3. Per
attention Linear: 2 vs dense 1. So expect a gain on 4B (bandwidth is ~36 % of
its time) and little on 0.6B. `use_triton_(model)` switches a model over;
more than 16 rows falls back to torch. Tested on CPU with the Triton
interpreter (validate_ql6 [10]): matches torch to 1e-6, greedy and
speculative outputs unchanged. Not yet run on a GPU, including under
`torch.compile`.

```bash
curl -sL https://raw.githubusercontent.com/tonythetiger168/qweave/claude/ql6-decode/scripts/runpod_ql6.sh \
  | STAGE=K bash        # 1x H100, ~2-2.5 h, ~US$8-10 (QUICK=1 first: ~US$1)
```

It runs a per-layer latency microbench (dense vs torch vs Triton low rank, M = 1
and 6), 0.6B and 4B decode with and without the kernels (CUDA graphs only), the
4B speculative rows again with the bf16-aware lossless check (drafts Qwen3-0.6B
and a QL-compressed 0.6B on the kernel path), and the vLLM references that
stage S did not reach. Decision rule: if the Triton rows do not beat dense on
4B, low-rank decode speed is closed as a negative result and QL6 rests on
speculative decoding.

### Stage K results (1× H100 SXM, 2026-10-30): negative

Raw: HF `chienhsinlin/qweave-runs`, `runs/2296up3smreihy_ql6/stageK`. CUDA graphs, batch 1.

| model | variant | tok/s | vs dense |
|---|---|---|---|
| 0.6B | dense | 370 | 1 |
| 0.6B | keep 0.9 + fuse (torch) | 297 | 0.80× |
| 0.6B | keep 0.9 + fuse + Triton | 167 | **0.45×** |
| 4B | dense | 174 | 1 |
| 4B | keep 0.6 (torch) | 165 | 0.95× |
| 4B | keep 0.9 + fuse + Triton | 81 | **0.46×** |
| 4B | spec, QL-compressed 0.6B draft, γ 3 | — | 0.61× (α 0.56, c 0.50) |
| vLLM dense, 0.6B / 4B | | 667 / 236 | (other engine) |

Per layer (M = 1) the Triton path is 0.18–0.32× of cuBLAS dense (4B q_proj
27 µs vs 6.5 µs); torch low rank is 0.5–0.87×. Outputs match (max abs error
≤ 0.016). The 16-row output tiles that `tl.dot` needs, plus re-reading the
t-partials in every program, cost more than the launches they save. A
competitive kernel would need a hand-tuned split-K GEMV without tensor-core
tiles, and even then low rank still launches more kernels than dense.

**QL6 conclusion.** At batch 1 on H100, none of the routes reaches 1.3×:
low rank (torch 0.75–0.95×, Triton ~0.5×), INT8 weight-only (≤ 1×),
speculative decoding with a 0.6B or QL-compressed draft (0.6–0.8×). Decode at
this size is launch-bound, not bandwidth-bound, so fewer bytes do not buy
speed. The biggest lever measured is the engine: dense vLLM is 1.8× (0.6B)
and 1.36× (4B) faster than HF + CUDA graphs. A fair "+X % inference" claim
therefore has to name its engine, and QL6 records low-rank decode speed as a
negative result. The one gated result in this direction is still QL3's
attention-only QSA (end to end 1.52× at 32K, short of the 2× prefill gate).

## Honest expectations (written before the run)

- L1 at the QL5 plan: expect about 1.1–1.15× with CUDA graphs. That would tie
  FermiQLLM's "+15 %" with a named baseline, not beat 1.3×.
- In **eager** mode a 0.6B model is launch-bound, so low rank can even be
  slower (two GEMMs per layer instead of one). Eager numbers are reported but
  the claim uses CUDA graphs.
- L2 on 4B with a 0.6B draft: the standard formula gives ~1.5–2× if the
  acceptance rate is 0.6–0.8 and the draft costs ~0.2 of a 4B step. Whether a
  QL-compressed draft beats the plain 0.6B draft (cheaper per step, lower
  acceptance) is the open question this stage answers.
- bf16 verification can flip a near-tie, so L2 agreement is reported, not
  assumed to be 100 %.
- 32K prefill: attention is ~80 % of 0.6B's prefill FLOPs at 32K, so QL3's
  3.47× attention speedup suggests ~2.3× end to end. The MLPs do not speed up.
