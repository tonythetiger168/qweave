# QL5-R: why stage A missed the gate, and the reworked recipe

> 2026-11-03 · Code: `phase4_llm/tn_layers.py` (`SplitEmbedding`, `SplitHead`,
> `split_embedding_factor`), `phase4_llm/ql5_compress.py` (new defaults),
> `phase4_llm/validate_ql5.py` (group 13 + both recipes end to end),
> `scripts/runpod_ql5.sh` `STAGE=R`. Stage A data: [QL5_RESULTS.md](QL5_RESULTS.md).
> **No GPU result for QL5-R yet.** The diagnosis below is read from the stage A
> reports; the attribution runs in stage R are there to confirm or refute it.

## Gate (unchanged)

−20 % of total parameters on Qwen3-0.6B-Base, MMLU within 2 points and GSM8K within
5 points of the uncompressed model, full test sets, paired bootstrap.
Stage A: MMLU −8.65, GSM8K −7.88 (FAIL).

## Diagnosis

1. **The embedding compression did most of the pre-heal damage, not the layer cuts.**
   The stage A reports allow one clean comparison (pre-heal PPL, dense 13.1):

   | run | embedding | layer params removed | PPL before heal |
   |---|---|---|---|
   | A3_ref | rank 512 | 9.5 % | 38.1 |
   | A3_embed035 | rank 384 | **5.1 %** | **47.3** |
   | A3_noembed | untouched | 27.1 % | 108.4 |

   Going from rank 512 to 384 on the embedding while cutting the layers *half as much*
   made the model worse. The weighted reconstruction error of that embedding looked
   tiny (0.31 % at rank 512, 0.46 % at 384), so the metric was hiding the damage.

2. **Why the error metric hid it.** D1 weights every row by token frequency. On a
   2,300-window calibration sample, the few thousand most frequent tokens carry almost
   all of the weight, so the rank-512 subspace is fitted to them. The tokens that carry
   knowledge (technical terms, names, units, rarer English words) are moderately rare,
   get projected onto that subspace and lose their identity, both as inputs and as
   LM-head rows (the head is tied). That matches the eval pattern: the loss is broad
   across all 57 MMLU subjects (computer security −20, management −18, world religions
   −18, clinical knowledge −17, high-school biology −17), long-context PPL is +63 %, and
   healing with 4× more tokens only moved PPL 26.3 → 21.6. A heal on 100 M tokens cannot
   re-learn 150 k token identities that were collapsed into 512 directions.

3. **The heal froze ~70 % of the network.** `--heal-scope tn` trained only the low-rank
   factors, norms and biases. The 153 uncompressed Linears could not adapt around the
   cuts. The dense control (which trained every weight on the same data) gained
   +3.9 GSM8K, so full-weight training on this mixture is safe.

4. **Fisher weighting hurt after healing** (A3_nofisher healed to 24.3 vs 26.3).

5. The ablation MMLU/GSM8K evals never ran (stage A hit its cap; stage C was not run),
   so points 1 and 2 rest on PPL. Stage R measures them directly.

## The QL5-R recipe

| | stage A | QL5-R (new defaults) |
|---|---|---|
| embedding / tied head | rank 512 for all 151,936 tokens | **split**: top 32,768 tokens by calibration frequency kept as exact rows, the remaining 119,168 at rank 128 (`--embed-mode split --embed-top 32768 --embed-tail-rank 128`) |
| embedding saving | 77 M (13.0 % of total) | 107 M (17.9 % of total) |
| layer params removed | 9.5 % | **2.9 %** |
| rank allocation | Fisher-weighted | unweighted (`--alloc-weight none`) |
| heal | factors + norms only | **every weight** (`--heal-scope all`): factors at 5e-5, untouched weights at 1e-5 (`--lr-dense`) |

Why this should work: the transformer is barely touched (2.9 % instead of 9.5 %), and
the embedding keeps every token an English/math text is likely to use as an exact row.
The tail is mostly other scripts and rare multi-byte pieces. The parameter saving is
real: `SplitEmbedding` stores `K·d + (V−K)·r + r·d` numbers and the tied `SplitHead`
reuses them.

What it costs, said plainly: tokens outside the top 32 k (by frequency in our
English + math calibration mixture) get a rank-128 representation. That will hurt
Chinese, Japanese, Korean and other low-frequency scripts. MMLU and GSM8K are English
and will not show this, so the report logs the calibration-token coverage of the top-K
block, and any public claim must add a Chinese eval (CMMLU or C-Eval) and a
multilingual PPL. A version for Chinese use would build the top-K from a mixture that
includes Chinese text.

For Qwen3-4B the embedding is only ~10 % of the model, so the same split saves
~7 % and the layers must still give ~13 %. QL5-R is a 0.6B recipe first; 4B needs the
layer side to improve as well, and stage R's `RA_layers_only` run tells us how costly
the layer cuts are on their own.

## Stage R run (4× H100, one pod, ~3.3 h)

```bash
R=https://raw.githubusercontent.com/tonythetiger168/qweave/claude/ql5-rework/scripts
nohup bash -c "curl -sL $R/runpod_ql5.sh | BRANCH=claude/ql5-rework STAGE=R timeout 220m bash" \
    > /workspace/ql5_R.out 2>&1 &
```

| GPU | jobs | answers |
|---|---|---|
| 0 | dense eval (MMLU + GSM8K); then MMLU on three un-healed models: stage A embedding only, stage A layer cuts only, new split embedding only | where stage A's MMLU loss came from, and how much the split keeps before any heal |
| 1 | R1: QL5-R at 25 M tokens + gate eval | direct comparison with stage A's 25 M runs |
| 2 | R2: R1 with stage A's heal scope; R3: R1 with top 49,152 exact (6.2 % layer cut) | is full-weight healing the fix; how much layer cut the recipe tolerates |
| 3 | R4: QL5-R at 100 M tokens + INT8 copy + gate eval | the gate candidate |

Cost: ~3.3 h of a Secure 4× H100 SXM pod at ~US$15/h (stage A/C rates), so
**~US$50–60** with setup. A `timeout 220m` cap bounds it at ~US$60. Results go to the
HF dataset and the pod is stopped at the end, as for stage A.

Decision after stage R:

- R4 passes: run the 4B stage with QL5-R plus whatever R3 says about layer cuts, and
  add CMMLU before any public claim.
- R1/R4 miss but RA shows the split alone is near-lossless: the remaining loss is in
  the layers; next is sequential (error-propagating) calibration of the layer cuts.
- RA shows the split alone already loses > 2 MMLU points: the diagnosis is wrong;
  report it and re-plan.
