# QL5 stage A: measured results (Qwen3-0.6B-Base, 2026-10-29)

> 4× H100 SXM (RunPod pod `a66qez2l8zvsz8`), torch 2.12.1, transformers 5.21.1,
> lm_eval 0.4.13, commit `79b4fb1`. Raw JSON: HF dataset `chienhsinlin/qweave-runs`
> (private), `runs/a66qez2l8zvsz8_ql5/stageA`. Plan and commands: [QL5_RUN_PLAN.md](QL5_RUN_PLAN.md).
> Every train/eval job finished with rc=0. The 5 h pod cap ended the run before the
> ablation task evals, the INT8 task eval and the on-pod gate step, so the gate was
> computed afterwards with `ql5_gate.py` on the uploaded full-test-set JSONs.

## Gate: FAIL

Full test sets (MMLU 14,042 questions, GSM8K 1,319 problems), paired bootstrap 95 % CI.

| model | params cut | MMLU | Δ [95 % CI] | GSM8K | Δ [95 % CI] |
|---|---|---|---|---|---|
| Qwen3-0.6B-Base | 0 % | 50.3 | | 50.1 | |
| **A1** main recipe, 100 M-token heal | **20.0 %** | 41.7 | **−8.65** [−9.50, −7.78] (gate −2) | 42.2 | **−7.88** [−10.54, −5.23] (gate −5) |
| A2 dense control, same data / tokens / KD | 0 % | 50.7 | +0.35 [−0.15, +0.85] | 54.1 | +3.94 [+1.59, +6.22] |

Other tasks, A1 vs Base: ARC-e 53.5 vs 58.1, ARC-c 30.8 vs 38.7, HellaSwag 44.3 vs 53.8,
PIQA 64.8 vs 70.3, WinoGrande 55.8 vs 58.3.

## Perplexity

| model | before heal | healed |
|---|---|---|
| Base | 13.09 | — |
| A2 dense control | 13.09 | 13.25 |
| A1 (100 M tokens) | 38.13 | 21.59 (+65 %) |
| A1 INT8 / FP8 weight-only copy | | 21.73 / 21.94 |

## Ablations (25 M-token heals, PPL only; task evals did not run)

| run | change | before heal | healed |
|---|---|---|---|
| A3_ref | main recipe | 38.13 | 26.30 |
| A3_nofisher | no Fisher weighting | 52.68 | **24.28** |
| A3_regauge | G1 re-gauge every 200 steps | 38.13 | 25.10 |
| A3_embed035 | embedding keep 0.35 | 47.29 | 25.49 |
| A3_noembed | embedding untouched, all 20 % from layers | 108.42 | 25.59 |
| A3_webonly | FineWeb-only calibration and heal | 48.33 | 25.86 |

## Reading

1. **Much better than QL1.5, still well short of the gate.** QL1.5 lost 13 MMLU and
   34 GSM8K points at −11 % (chat model). QL5 loses 8.7 and 7.9 at −20 % (Base model).
   The reference models differ, so this is not a clean comparison, but GSM8K no longer
   collapses. Its CI upper bound (−5.2) almost reaches the −5 gate. MMLU is the blocker.
2. **The loss comes from compression, not from the heal data.** The dense control
   healed on the same mixture and tokens keeps MMLU and gains 3.9 GSM8K points. That
   also means decontamination plus the mixture did not leak test answers into MMLU.
3. **Fisher weighting is a PPL trap.** It gives the best *pre-heal* PPL (38 vs 53) but
   the worst *healed* PPL of all runs (26.3 vs 24.3). Putting the cuts where first-order
   loss is smallest leaves layers that heal less well. Unless the missing task evals
   say otherwise, `--alloc-weight none` should be the default.
4. **Every ablation beat the reference after healing**, and the spread is small
   (24.3–25.9 vs 26.3). That includes no-embedding, no-mixture and harder embedding
   compression. The reference recipe combined choices that each look good before
   healing and neutral or worse after it. PPL alone cannot rank these runs (F4), so
   the ablation MMLU/GSM8K evals are needed before choosing a recipe.
5. **Healing has diminishing returns.** 4× the tokens (25 M → 100 M) moved PPL only
   from 26.3 to 21.6, still +65 % over Base. A longer heal alone will not close an
   8.7-point MMLU gap.
6. **Quantization stacks almost for free.** INT8 costs +0.7 % PPL and FP8 +1.7 %
   on top of the low-rank model. INT8 task scores were not run.

## Next

- Run the 6 ablation and INT8 MMLU/GSM8K evals (models saved on the pod volume).
- Pareto sweep (surpass plan E2) at −5 / −10 / −15 % with the best ablation recipe, to
  find the largest cut that passes the gate on 0.6B.
- Hold stage B (4B) until a 0.6B recipe passes or comes within ~1 point.
