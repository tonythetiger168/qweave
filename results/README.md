# GPU run results

Raw reports copied from the Hugging Face dataset `chienhsinlin/qweave-runs`
(`runs/<folder>/`). Only finished runs are included; logs, checkpoints and quick
smoke runs are left out. Each folder holds the JSON reports, the `STAGES` file
(one line per stage with its exit code and wall time) and, where present, the
GPU / version / commit the run used.

| Folder | Phase | Hardware | Date | Write-up |
|---|---|---|---|---|
| `vbhwminemznaek` | Phases I–III validation, QL1.5 (WikiText heal, ablations), QL2, QL3, QL4 | 1× H100 80GB | 2026-10-28 | `docs/QL_RESULTS.md` ([#1](https://github.com/tonythetiger168/qweave/pull/1)) |
| `0il1br2zf5k25e` | QL1.5 (FineWeb-Edu heal), QL2 layer bench, QL3 (log budgets), QL4 dense vs −11% / −22% | 1× H100 80GB | 2026-10-28 | `docs/QL_RESULTS.md` ([#1](https://github.com/tonythetiger168/qweave/pull/1)) |
| `a66qez2l8zvsz8_ql5` | QL5 stage A: Qwen3-0.6B-Base, −20% params, dense control, ablations, full-test-set MMLU/GSM8K | 4× H100 SXM | 2026-10-29 | `docs/QL5_RESULTS.md` ([#4](https://github.com/tonythetiger168/qweave/pull/4)) |
| `wg2dxaw82efhjm_ql6` | QL6 stage S: batch-1 decode, INT8, fusion, QSA prefill, 4B speculative | 1× H100 | 2026-10-29 | `docs/QL6_RUN_PLAN.md` ([#5](https://github.com/tonythetiger168/qweave/pull/5)) |
| `7jcwnq6gkx9ppk_ql6` | QL6 stage L: 4B speculative rerun with the lossless check | 1× H100 SXM | 2026-10-30 | `docs/QL6_RUN_PLAN.md` ([#5](https://github.com/tonythetiger168/qweave/pull/5)) |
| `jcal9lo2xl3j4y_e5` | E5: torch state vector on CUDA, gross-code decoder timings | 1× H100 80GB | 2026-10-30 | `docs/E5_GPU_RESULTS.md` ([#8](https://github.com/tonythetiger168/qweave/pull/8)) |
| `pfpjronh2bj0t1_e5b` | E5b: cuStateVec scaling to n=32, batched torch BP+OSD on GPU, CUDA-Q QEC | 1× H100 80GB | 2026-10-30 | not written up yet; headline numbers in the top-level README |

Not included: QL6 stage K (`2296up3smreihy_ql6`, still running) and an
unfinished stage L attempt (`rt124qebagr57z_ql6`, no DONE marker).
