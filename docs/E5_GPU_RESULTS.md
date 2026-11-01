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

---

# E5b/E5c: cuStateVec backend + batched GPU decoder (2026-10-30)

Pods pfpjronh2bj0t1 (run 1) and psaxrxkubht9ij (run 2, with batched OSD), 1x H100, about 2 min of benchmark each.
Raw data: `runs/pfpjronh2bj0t1_e5b/`, `runs/psaxrxkubht9ij_e5c/`. Script: `scripts/e5b_gpu_bench.py`.

## cuStateVec backend (`phase1_basic/custatevec_backend.py`)
Fidelity vs the torch backend is 1 to within 1e-14 at n = 4..20, including the x/z/cz/generic two-qubit paths.

| n | torch (s/layer) | cuStateVec (s/layer) | speedup |
|---|---|---|---|
| 22 | 0.0119 | 0.0034 | 3.5x |
| 26 | 0.202 | 0.026 | 7.7x |
| 30 | 4.13 | 0.441 | 9.4x |
| 31 | does not fit | 0.886 | |
| 32 | does not fit | 1.87 | |

cuStateVec runs at the H100's memory-bandwidth limit for n >= 27. It works in place, so n = 32 (64 GiB, complex128) now fits.

## Gross code decoding, p = 4%, code capacity (run 2)
| decoder | batch | time per shot | failures |
|---|---|---|---|
| **qweave torch-cuda (batched BP + batched OSD)** | 100,000 | **2.6 us** | 2657 / 100k |
| qweave torch-cuda | 10,000 | 15.4 us | 274 / 10k |
| CUDA-Q nv-qldpc-decoder, raw call (bp_batch_size 1000) | 10,000 | 6.2 us | 273 / 10k |
| CUDA-Q through `make_decoder("cudaq")` | 100,000 | 15.8 us (Python list conversion) | 2656 / 100k |
| `ldpc` (C++, CPU, one shot at a time) | 10,000 | 32.7 us | 274 / 10k |

- All decoders make the same decisions: their failure counts differ by at most 1.
- At large batches, qweave's GPU decoder is 12x faster than `ldpc` and 2.3x faster than CUDA-Q's raw call. That is the offline-simulation use case: threshold sweeps, sampling-heavy experiments.
- In run 1 the OSD stage ran per shot on CPU, which cost about 40 us/shot. Batching OSD on the GPU removed that bottleneck: 50 us dropped to 2.6 us per shot.
- Single-syndrome latency is still about 1.5 ms. That is kernel-launch and Python overhead, so it is not suitable for real-time decoding. For one syndrome at a time, `ldpc` (14 us) remains the right backend.

## Decoder backends now in qweave (`qldpc.make_decoder`)
`auto` picks `cudaq` > `torch-cuda` > `ldpc` > `numpy`, whichever is installed. For large offline batches, pass `backend="torch-cuda"`.
