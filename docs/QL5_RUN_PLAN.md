# QL5: keep knowledge and reasoning under compression

> 2026-10-29 · Code: `phase4_llm/ql5_compress.py`, `ql5_data.py`, `ql5_gate.py`,
> `validate_ql5.py` (12 groups, all PASS on CPU) · GPU runner: `scripts/runpod_ql5.sh`
> · Plan: [FERMIQLLM_SURPASS_PLAN.md](FERMIQLLM_SURPASS_PLAN.md) §3 QL5, G1, D1, E2.
> **No GPU result yet.** Nothing below is a measured QL5 number.

## Gate

At **at least 20 % of total parameters removed**, versus the uncompressed
`Qwen3-0.6B-Base` (then `Qwen3-4B-Base`), on the **full** test sets:

| benchmark | max drop | how measured |
|---|---|---|
| MMLU (57 subjects, 14,042 questions, micro-average) | 2 pt | lm-eval, `--limit 0` |
| GSM8K (1,319 problems, 5-shot, strict match) | 5 pt | lm-eval, `--limit 0` |

`ql5_gate.py` pairs the two models question by question and reports the
difference with a 95 % paired-bootstrap interval. "PASS" is on the point
estimate. "PASS at CI bound" is the stronger claim we would make in public.

## What changed from QL1.5, and why

QL1.5 lost MMLU −13 and GSM8K −34 at only −11 % params (QL_RESULTS.md §4).

| QL1.5 | QL5 | measured failure it targets |
|---|---|---|
| chat model `Qwen3-0.6B` as reference | `Qwen3-0.6B-Base`, plus a **dense control** healed with the identical data, tokens and KD | F3: fine-tuning alone moves PPL 21.7 → 18.2 |
| calibration and heal on FineWeb-Edu only | mixture: FineWeb-Edu 50 %, OpenWebMath 15 %, MetaMathQA 15 %, GSM8K-train 5 %, MMLU auxiliary-train 15 %; every document 13-gram-checked against the GSM8K and MMLU **test** sets | F4: the directions that encode knowledge and math were never in the calibration Gram matrices |
| all layers weighted equally in rank allocation | allocation minimises activation error × layer Fisher importance `Σ(∂L/∂W ⊙ W)²` on the same mixture | F4 |
| embedding untouched, so all the saving came from the transformer | **D1**: tied embedding / LM head compressed by low rank weighted by token frequency (embedding side) and the final hidden-state Gram (head side), exact as in whitened SVD | 26 % of 0.6B params sat in the embedding, mostly in rare tokens |
| 6 M heal tokens, full-vocab logits | **100 M tokens**, KD computed on checkpointed chunks (no 16k × 152k logits tensor), math sources up-weighted in the loss, multi-GPU (grad all-reduce + ZeRO optimizer) | heal was short and memory-bound |
| — | **G1**: optional re-gauging of every low-rank bond to the balanced canonical form during heal (A B unchanged, AdamW moments reset) | makes the "gauge freedom" claim testable |
| — | INT8 / FP8 weight-only fake-quantised copies of the healed model, evaluated with the same gate | low rank must stack with quantisation to matter on the Pareto front |
| SQA allocator by default | greedy by default | F5: greedy is within 1 % of optimal and 2000× faster on this problem |

### Where the 20 % comes from (Qwen3-0.6B-Base, 596 M params)

| part | params | QL5 default | saved |
|---|---|---|---|
| embedding = LM head (tied), 151,936 × 1,024 | 155.6 M | rank 512 (`--embed-keep 0.5`) | 77.3 M (13.0 % of total) |
| 196 attention/MLP Linears | 440.4 M | Fisher-weighted ranks | 41.9 M (9.5 % of these layers) |
| **total** | 596.0 M | | **119.2 M (20 %)** |

So the transformer is cut *less* than in the QL1.5 −11 % run (15 % of
these layers), and the embedding carries most of the saving. For 4B the
embedding is only ~10 % of the model, so ~17 % of the transformer layers
must go; that is why 4B runs only after 0.6B passes or comes close.

## Ablations in stage A (25 M tokens each, compared with `A3_ref`)

| run | change | question it answers |
|---|---|---|
| `A3_nofisher` | `--alloc-weight none` | does Fisher weighting help? |
| `A3_noembed` | `--embed-keep 1` (all 20 % from layers) | does D1 help? |
| `A3_webonly` | calibration + heal on FineWeb-Edu only | does the knowledge/reasoning mixture help? |
| `A3_regauge` | `--regauge-every 200` | G1 (else reported as a negative result) |
| `A3_embed035` | `--embed-keep 0.35` | how far can the embedding go? |

## Exact commands

On a RunPod pod with **4× H100 80GB SXM**, image `runpod/pytorch` (torch ≥ 2.4),
secret `HF_TOKEN` set (results go to the private dataset `<you>/qweave-runs`):

```bash
# 0. check the pod first (~10 min, ~US$2-3): every stage at toy size
curl -sL https://raw.githubusercontent.com/tonythetiger168/qweave/claude/ql5-quality/scripts/runpod_ql5.sh \
  | QUICK=1 STAGE=A bash

# 1. stage A: Qwen3-0.6B-Base (go/no-go)
curl -sL https://raw.githubusercontent.com/tonythetiger168/qweave/claude/ql5-quality/scripts/runpod_ql5.sh \
  | STAGE=A bash

# 2. stage B: Qwen3-4B-Base, only after stage A is reviewed
curl -sL https://raw.githubusercontent.com/tonythetiger168/qweave/claude/ql5-quality/scripts/runpod_ql5.sh \
  | STAGE=B bash
```

Set `RUNPOD_API_KEY` too if you want the pod to stop itself at the end.
Each run writes `ql5_report.json` (allocation, Fisher scores, per-source
token counts, decontamination counts, PPL curve, tok/s, peak VRAM),
`eval/ql4_compressed.json`, per-question scores and `gate.json`.

Single runs by hand:

```bash
python phase4_llm/ql5_compress.py --model Qwen/Qwen3-0.6B-Base --total-saving 0.2 \
    --embed-keep 0.5 --heal-tokens 100e6 --quant int8,fp8 --save-model --out /workspace/ql5/A1
python phase4_llm/eval/run_eval.py --dense Qwen/Qwen3-0.6B-Base --compressed /workspace/ql5/A1/model.pt \
    --limit 0 --log-samples --out /workspace/ql5/A1/eval
python phase4_llm/ql5_gate.py --dense /workspace/ql5/A1/eval/ql4_dense.json \
    --compressed /workspace/ql5/A1/eval/ql4_compressed.json --ql5-report /workspace/ql5/A1/ql5_report.json
```

## Cost estimate

Assumes ~US$3 per H100-hour (check current RunPod pricing). Throughput is
extrapolated from QL1.5 (17–23 k train tok/s on 0.6B with KD on one H100)
and is the main uncertainty: the script prints tok/s every 50 steps, so a
bad estimate shows up in the first minutes.

| stage | what runs | wall time on 4× H100 | cost |
|---|---|---|---|
| QUICK | toy versions of everything | ~10 min | ~US$2–3 |
| A (0.6B) | 2 × 100 M-token heals + 6 × 25 M ablations in parallel, then 10 full MMLU+GSM8K evals | ~3.5 h | **~US$45–70** |
| B (4B) | 2 × 100 M-token heals with 4-GPU data parallel (~2.3 h each), 4 evals | ~6–7 h | **~US$85–130** |
| **total** | | | **~US$130–200** |

That is below the plan's US$300–600 because everything runs on one node and
the 0.6B stage decides whether 4B is worth running.

## Honest expectations

- At −11 % QL1.5 lost 13 MMLU points. QL5 cuts the transformer less (−9.5 %
  of the layers vs −15 %), calibrates on the skills that collapsed, heals
  16× longer and against the Base model. Each of those helps in the
  literature, but we know of no published low-rank method that keeps MMLU
  within 2 points at −20 % on a model this small without much longer training. A miss is a real
  possibility, and the ablations will show which part helped.
- If stage A misses, the cheapest next step is a longer heal (the PPL curve
  in the report shows whether it is still improving), not 4B.
- Embedding compression mostly removes rare tokens (CJK, code, rare words).
  MMLU and GSM8K are English and will not show that cost, so stage A also
  reports WikiText PPL and the robustness suite, and a Chinese eval (e.g.
  CMMLU) should be added before claiming anything for Chinese use.

## Stage C: ablation evals + Pareto sweep (added 2026-10-29, after stage A)

Stage A's models were deleted with its pod, so stage C retrains the six 25M-token
ablations (`C_*`) and adds a −5 / −10 / −15 % sweep (`S05`, `S10`, `S15`) that uses the
no-Fisher allocation and keeps the embedding's share of the saving at ~65 %. Every
model is evaluated on full MMLU + GSM8K right after training, on the same GPU, and
gated with `--min-saving 0`, so each row reports whether its quality drop is within
−2 / −5. All runs use 25M tokens and are compared with `C_ref` (the −20 % main recipe at
25M). Estimated ~3.5–4 h on Secure 4× H100 (~US$55–65), so use a 270-minute cap.

```bash
R=https://raw.githubusercontent.com/tonythetiger168/qweave/claude/ql5-quality/scripts
nohup bash -c "curl -sL $R/runpod_ql5.sh | STAGE=C timeout 270m bash" > /workspace/ql5_C.out 2>&1 &
```
