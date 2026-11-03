# QL8 — DiracLM: an open-weights quantum-inspired LLM

> 2026-11-03 · Code: `phase4_llm/release/` (recipe gating, MATH-500 / GPQA-Diamond / BBH
> harness, model card, packaging, loader), `phase4_llm/validate_ql8.py` (CPU, in CI),
> `scripts/runpod_ql8.sh`. **No GPU run has been done and nothing has been published.**
> Every GPU stage and any public upload needs Tony's explicit approval.

## 0. In one paragraph

qweave stops being only "a toolkit that checks FermiQLLM" and ships its own model line,
**DiracLM**. The first release, **DiracLM-4B**, is built on Qwen3-4B-Base. It includes
only the quantum-inspired components that passed their gate *on that base*: today that
is likely just QSA sparse attention, plus QL7 RL selection and QL5 compression if their
pending runs pass. It is benchmarked against the plain base on FermiQLLM's own tests
(MATH-500, GPQA-Diamond, BBH), with per-question results and confidence intervals. A
second base (Phi-4-mini) shows that the method is not tied to one architecture.
FermiQLLM 1.0 has published no weights, baseline, protocol or per-question output.
DiracLM publishes all four, negative results included.

## 1. Identity and positioning

**Name.** DiracLM (chosen 2026-11-03). Fermi–Dirac statistics makes it a natural
counterpart to FermiQLLM. The name was checked on 2026-11-03: no model, dataset, space
or org named BoseLM / DiracLM / PlanckLM / WeaveLM exists on Hugging Face or GitHub,
and no product with those names turned up in a web search. BoseLM was dropped because
of trademark risk (Bose Corporation; Boson AI also exists). WeaveLM collides with W&B
Weave. PlanckLM risks a conflict with the Max Planck Society's naming. These checks are
not a legal trademark search; do one before any commercial use.

**Naming scheme.** `DiracLM-<size>[-<base family>]-v<version>`, for example `DiracLM-4B`
(Qwen3 base, the flagship) and `DiracLM-Phi4-mini`. The code stays in the qweave repo
(`phase4_llm/`). The Hugging Face org is `DiracLM`, which is free as of today; Tony
has to create it, because this project's tooling cannot create orgs.

**One-line description (README / HF).** "DiracLM: open, measured, quantum-inspired
LLMs. Classical methods borrowed from quantum physics, running on GPUs, shipped only
when they beat their own baseline."

**How we describe it against FermiQLLM 1.0.**

| | FermiQLLM 1.0 | DiracLM |
|---|---|---|
| Weights | not released | Apache-2.0 / MIT on Hugging Face |
| Claimed gains | "+15 % inference, −25 % RL cost, +10–20 % on MATH-500 / GPQA-D / BBH" | Only what passed a pre-registered gate on the release base, with 95 % CIs |
| Baseline | none named | plain base, same-recipe dense control, official post-trained model |
| Per-question results | none | in the HF repo under `eval/` (GPQA as ids and scores only) |
| Negative results | none | published (QL1.5 PPL gate, QL2 annealing, QL5 stage A, QL6 decode) |
| "Quantum" wording | "quantum-enhanced" | "quantum-inspired"; no quantum hardware and no quantum-advantage claim |

We do not claim to beat FermiQLLM's numbers, because those numbers cannot be checked.
We claim to be the open, reproducible version of the same idea.

## 2. What goes into DiracLM-4B v0.1

| Component | Where it stands (2026-11-03) | Gate on the release base | In v0.1 if |
|---|---|---|---|
| **QSA** sparse attention (QL3) | PASS on Qwen3-0.6B: 1.49× attention at 16K, +0.95 % PPL (top-16) | re-run `bench_ql3.py` on Qwen3-4B-Base: ≥1.2× at 16K and ≤+1 % 16K-window PPL | S1 passes |
| **QL7** QUBO prompt selection for GRPO | code merged (#12); stage A on 0.6B not run yet | `ql7_gate.py`: ≤0.75× GPU-hours to the baseline's score, non-inferior final score | QL7 stage A passes on 0.6B **and** stage B passes on 4B |
| **QL5** activation-aware TN compression | stage A FAILED (−20 % params: MMLU −8.7, GSM8K −7.9); being reworked in another thread | the rework thread's gate (−20 % params, MMLU −≤2, GSM8K −≤5), re-measured on 4B | the rework passes on 4B |

`phase4_llm/release/recipe.py` applies these rules mechanically. A component goes in
only if its gate report says PASS **and** that report was measured on the release base,
so the 0.6B QSA result does not count for the 4B model.

**Two facts to be honest about.**

1. QSA does not change the weights, and it only activates for prompts of at least
   `min_len` tokens (16K by default). MATH-500, GPQA and BBH prompts are short, so QSA
   cannot move those scores in either direction. Its value is long-context prefill speed
   at a ≤1 % PPL cost, and the card states it that way.
2. Only RL (QL7) or compression (QL5) produce new weights. If neither passes, DiracLM-4B
   v0.1 is Qwen3-4B with QSA switched on: a runtime release with an eval report, not a
   new model. That outcome is still publishable and honest, but it is small.

**Three outcomes, decided by the pending runs:**

| Outcome | Condition | What ships |
|---|---|---|
| A (target) | QL7 passes on 0.6B and on 4B | DiracLM-4B = Qwen3-4B-Base + GRPO (QL7 selector) + QSA; claims: same reasoning scores at ≤0.75× RL GPU-hours, and long-context prefill speed |
| B | QL7 fails | GRPO weights are then plain GRPO with no quantum-inspired part, so we do **not** brand them DiracLM. Ship QSA on the official Qwen3-4B as `DiracLM-4B-QSA` with the long-context claim only, and publish QL7 as a negative result |
| C | QL5 rework also passes on 4B | add the compressed variant `DiracLM-4B-C` (−20 % params) as a separate repo, compared against the uncompressed DiracLM-4B |

## 3. Base models

* **Primary: `Qwen/Qwen3-4B-Base`** (Apache-2.0). It is the same family as every
  earlier QL measurement, so the 0.6B results carry over in design. We start RL from the
  *Base* checkpoint because that is the only start where "same quality at lower RL cost"
  can be checked against a same-recipe control. The official post-trained `Qwen/Qwen3-4B`
  is reported as a reference row, not as the gate. It was trained with far more RL than
  we will spend, so DiracLM-4B will very likely score below it, and the card says so.
* **Second: `microsoft/Phi-4-mini-instruct`** (3.8B, MIT). This is the recommended pick
  over DeepSeek-R1-Distill-Qwen, which is a Qwen2.5 architecture and would not show
  generalization. Phi-4-mini has a different tokenizer, attention layout and training
  recipe. v0.1 ships QSA only on it (gate re-measured in S1, evals in S2), with no RL, so
  its release is a runtime config plus eval report. RL on Phi-4-mini is a v0.2 decision.

## 4. Evaluation protocol and release gate (frozen before any GPU run)

Harness: `python -m phase4_llm.release.reasoning_eval run|compare` (vLLM backend for
plain checkpoints; transformers backend for QSA or QL5 layers).

| Benchmark | Data | Items | Samples per item | Answer format | Grader |
|---|---|---|---|---|---|
| MATH-500 | `HuggingFaceH4/MATH-500` test | 500 | 4 | `\boxed{}` | Hendrycks/PRM800K normaliser + numeric check; `math-verify` on pods |
| GPQA-Diamond | `Idavidrein/gpqa` `gpqa_diamond` | 198 | 8 | `Answer: X`, options shuffled with a fixed per-item seed | exact letter |
| BBH | `lukaemon/bbh`, 27 subtasks | 6,511 | 1 | zero-shot CoT, "So the answer is …" | normalised exact match (MC letter, words, numbers) |

* Scores are avg@k: the mean over items of each item's mean correctness over k samples.
  BBH also reports a macro average over its 27 subtasks.
* Sampling: Qwen3's recommended settings (thinking T=0.6, top-p 0.95, top-k 20;
  non-thinking T=0.7, top-p 0.8). The Base-start arms (Qwen3-4B-Base, DiracLM-4B) use the
  same chat template with thinking off and 4,096 new tokens. Qwen3-4B (reference) uses
  thinking mode with 32,768 new tokens.
* A reply whose think block never closes, or that states no answer, scores 0. Its rate
  is reported (`truncated_rate`, `no_answer_rate`) so formatting failures are visible.
* **Comparison:** paired bootstrap over items (10,000 resamples), difference in points
  with a 95 % CI.
* **Release gate** (`recipe.release_gate`): at least one component included, and on each
  of the three benchmarks the CI lower bound is ≥ −2 pt against the plain base. A
  benchmark is called "better than base" only if its CI lower bound is > 0; otherwise the
  card says "within X pt". A failed gate still builds a folder, but the card is stamped
  "Not a release candidate", and `package push --public` refuses to run.
* **Named controls** (from the surpass plan): plain base; the same-recipe dense GRPO run
  (QL7's `base` arm), which separates "RL helped" from "the quantum-inspired selector
  helped"; the official Qwen3-4B. Truncated-SVD and CompactifAI rows apply only to the
  QL5 variant.
* **Long context (QSA's actual claim):** the S1 gate (16K-window PPL + kernel speed) is
  what the card cites. RULER / LongBench at 16K–32K is follow-up work, not part of v0.1.
* **Contamination:** the GRPO data is GSM8K train + the MATH *train* split with the 500
  MATH-500 problems removed by `unique_id`. That filter is a required check for QL7
  stage B on 4B.
* **GPQA terms:** the dataset asks users not to publish its questions in plain text.
  Release folders keep only ids, extracted answers and scores for GPQA
  (`package._public_eval_copy`, checked in CI). Full replies stay in the private
  `qweave-runs` dataset.

## 5. Packaging

A DiracLM Hugging Face repo is a normal transformers checkpoint, so it loads without
`trust_remote_code` and runs in vLLM:

```
DiracLM-4B/
  model-0000x-of-0000y.safetensors, config.json, generation_config.json, tokenizer files
  diraclm_config.json   # components + their gate numbers, excluded components + reasons, release gate
  README.md             # generated by model_card.py from the reports, nothing typed by hand
  LICENSE, NOTICE       # the base model's license and notices, copied by stage F
  eval/base/*, eval/model/*, eval/comparison.json   # per-item results (GPQA: ids + scores)
  MANIFEST.json         # sha256 + size of every file
```

* `phase4_llm.release.loader.load_diraclm(repo)` reads `diraclm_config.json` and turns on
  QSA (`attn_implementation="qsa"`). Plain `AutoModelForCausalLM` gets dense attention.
* This PR fixes a bug in `qsa_hf_attention`: for a 1-token decode step against a KV
  cache it used SDPA `is_causal=True`, which aligns the mask top-left, so the query
  attended only to key 0. It also ignored padding masks. Neither was hit by the QL3
  PPL runs (one full window, no cache), but generation would have been wrong. Decode,
  chunked prefill and padded batches now use the right mask (`validate_ql8.py` [11];
  the old code fails that check).
* **Publishing steps** (each needs Tony): (1) create the `DiracLM` org on HF; (2)
  `package push` to a **private** repo for review; (3) flip to public, or rerun with
  `--public`, which is allowed only when the release gate passed. `push` refuses to run
  without `--approved-by`.

## 6. Licensing

| Item | License | What we must do |
|---|---|---|
| Qwen3-4B-Base / Qwen3-4B | Apache-2.0 | ship its LICENSE (and NOTICE if any), state that we modified it; no naming requirement, so "DiracLM" is fine |
| Phi-4-mini-instruct | MIT | keep the copyright and license text |
| DiracLM-4B weights | Apache-2.0 (same as base) | `license: apache-2.0`, `base_model:` in the card front matter |
| DiracLM-Phi4-mini | MIT | same |
| qweave code | Apache-2.0 | unchanged |
| GSM8K, MATH, BBH | MIT | cite |
| GPQA | CC-BY-4.0 + do-not-publish-examples request | ids and scores only in public files |

Not a substitute for legal review if DiracLM is used commercially.

## 7. Costed GPU plan (RunPod, 1× H100 SXM Secure ≈ US$3.99/h, 4× ≈ US$15.96/h)

| Stage | Pod | What | Wall | Cost | Needed for |
|---|---|---|---|---|---|
| Q | 1× H100 | CPU checks, harness smoke on Qwen3-0.6B (20 items/task, vLLM + HF) | ~20–30 min | ~US$2 | all outcomes |
| S1 | 1× H100 | QSA gate on Qwen3-4B-Base and Phi-4-mini (`bench_ql3.py`) | ~1–1.5 h | ~US$4–6 | all outcomes |
| S2 | 1× H100 | the three benchmarks for Qwen3-4B-Base, Phi-4-mini, Qwen3-4B (thinking) | ~2–3 h (cap 4 h) | ~US$8–12 (≤ US$16) | all outcomes |
| QL7 A | 1× + 4× H100 | already planned in the QL7 thread (0.6B) | — | ~US$75–90 (QL7 budget) | outcome A |
| QL7 B | 4× H100 | GRPO on Qwen3-4B-Base: `base` ×3 seeds + winning arm ×3 seeds + `dyn` ×1, ~300 steps; ≈ 7 GPU-h per run | ~12–15 h | ~US$170–260 (cap US$300) | outcome A |
| F | 1× H100 | eval the trained checkpoint, compare, build the release folder | ~1–2 h | ~US$4–8 | A or B |

* **Without RL** (outcome B): Q + S1 + S2 + F ≈ **US$20–30**.
* **With RL** (outcome A): add QL7 stage B ≈ **US$190–290 total** (cap ≈ US$330),
  excluding QL7 stage A, which is already budgeted in its own thread.
* The QL7 B estimate scales the 0.6B stage-A timing estimate (~1 GPU-h per 200-step run, not yet measured) by ~7×
  in compute and 1.5× in steps. Stage A1 measures the real phase fractions. Re-cost B
  from those numbers before asking for approval.
* Each pod pushes its reports to the private `qweave-runs` dataset and is then stopped,
  as for every earlier stage. Trained checkpoints go to a **private** HF model repo.

**Recommended order:** Q + S1 + S2 now (~US$15–20; needed whatever QL7 does), then QL7
stage A in its own thread, then decide on stage B, then F.

## 8. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| QL7 fails on 0.6B or 4B | no RL claim; outcome B | publish as a negative result; still ship QSA |
| QSA gate fails on 4B (GQA ratio 4:1 vs 2:1 on 0.6B, 36 layers) | nothing quantum-inspired to ship | try top-24 / top-32 within the same gate; otherwise QL8 v0.1 is the eval report only |
| GRPO from Base learns the answer format but not reasoning | MATH/GPQA gains are format effects | `no_answer_rate` reported per arm; the same-recipe dense control isolates the selector's effect |
| Benchmark contamination | inflated scores | MATH-500 ids filtered out of RL data; GPQA/BBH not used in training |
| Name or trademark challenge | rename | the name is one field in each recipe; the code is name-agnostic |

## 9. In this PR (code only, no GPU)

* `phase4_llm/release/`: `reasoning_eval.py`, `gates.py`, `recipe.py`, `model_card.py`,
  `package.py`, `loader.py`, `recipes/diraclm-4b.json`, `recipes/diraclm-phi4mini.json`
* `phase4_llm/qsa_attention.py`: decode / padding mask fix (section 5)
* `phase4_llm/validate_ql8.py`: 11 CPU checks, run in CI
* `scripts/runpod_ql8.sh`: stages Q / S1 / S2 / F; it never uploads a model
