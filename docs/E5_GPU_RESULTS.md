# E5 GPU results (pod jcal9lo2xl3j4y, 1x H100 80GB HBM3, 2026-10-30)

torch 2.12.1+cu129, Python 3.12. Script: `scripts/e5_gpu_bench.py` (run by `scripts/runpod_e5.sh`).
Raw data: HF dataset `chienhsinlin/qweave-runs`, `runs/jcal9lo2xl3j4y_e5/`. Wall time 7 min, about US$0.50.

## 1. Torch state-vector backend on CUDA: first GPU run, correct
Fidelity against the numpy engine after 4 random layers: 1 - 2e-15 or better at n = 6, 10, 14, 18.

## 2. Speed (one layer = RZ·RY·RZ on every qubit + CNOTs on even pairs, complex128)
| n | numpy CPU (s) | torch H100 (s) | speedup | effective GB/s |
|---|---|---|---|---|
| 20 | 0.300 | 0.0234 (first call, warm-up effects) | 13x | 43 |
| 22 | 1.43 | 0.0117 | 122x | 377 |
| 24 | 6.16 | 0.0476 | 129x | 406 |
| 26 | 26.2 | 0.203 | 129x | 413 |
| 28 | | 0.875 | | 412 |
| 30 | | 4.12 | | 375 |

- n=31 does not fit: the einsum path needs about 3x the state in memory (96 GiB). Max is n=30 in complex128.
- About 410 GB/s is roughly 12% of the H100's 3.35 TB/s peak. cuStateVec reaches about 90%, so there is about 8x headroom.
- The Qiskit Aer GPU (cuStateVec) rows are **not a fair comparison**. Their time includes transpiling and copying the full state back to the host (16 GiB at n=30). The n=32 row (0.13 s) is implausible and is excluded.

## 3. Gross code [[144,12,12]] decoding, code capacity p = 4%, 5000 shots
| decoder | median single-syndrome latency | batched time per shot | logical failures |
|---|---|---|---|
| `ldpc` BP+OSD-0 (C++, CPU) | 14 us | 29 us | 123 / 5000 (2.5%) |
| CUDA-Q QEC `nv-qldpc-decoder` (GPU) | 44 us | 103 us (bp_batch_size=1, not parallel) | 34 / 2000 (1.7%) |
| qweave numpy BP+OSD-0 (CPU) | 155 us | 2677 us | 123 / 5000 (2.5%) |

- Our decoder makes exactly the same decisions as `ldpc`, but it is 11x slower in latency and 90x slower batched.
- CUDA-Q's lower failure rate comes from different default BP/OSD settings. Its default batch size of 1 leaves the GPU's parallelism unused.

## Next
1. Port the BP part of `BpOsdDecoder` to batched torch on GPU; OSD stays on CPU for the shots BP fails.
   Target: batched time per shot below `ldpc`'s 29 us.
2. Add an optional cuStateVec path (cuquantum) for the state vector, keeping torch as the fallback.
   Also cut the 3x memory overhead so n=31 fits.
3. Rerun CUDA-Q with bp_batch_size > 1 and matched BP settings, so the comparison is fair.
