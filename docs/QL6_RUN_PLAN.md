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

## Honest expectations

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
