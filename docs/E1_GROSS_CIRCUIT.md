# E1 part 2: gross code at circuit level + Relay-BP (2026-10-30)

Code: `phase2_noise/bb_circuit.py` (circuit) and `phase2_noise/qldpc_gpu.py::RelayBpDecoder`.
Checks: `phase2_noise/validate_e1.py` section [7]. Everything here ran on CPU; no GPU was used.

## Circuit
- This is the depth-8 syndrome cycle of Bravyi et al. (Nature 627, 778, 2024) for bivariate bicycle codes.
  The CNOT order follows the open implementation in gongaa/SlidingWindowDecoder. arXiv was not reachable from the build environment.
- For the gross code it uses 288 qubits: 144 data, 72 X checks and 72 Z checks. It is a Z-basis memory experiment with 12 logical observables.
  The noiseless circuit fires no detector.
- Noise: DEPOLARIZE2 after every CNOT, flips after reset and before measurement, and DEPOLARIZE1 on each data qubit once per cycle, all at rate p.
  IBM's own model adds idle errors at every idle step, so its numbers are slightly worse at the same p.
- Distance: stim's heuristic search finds an undetectable logical error of weight 12 at circuit level. That is an upper bound only; the paper reports a circuit distance of 10 or less.

## Pseudo-threshold (BP+OSD-CS order 7, `ldpc`, 12 cycles, 200 shots per point)
| p | logical error per cycle (all 12 qubits) | 12 p |
|---|---|---|
| 0.7% | 0.061 | 0.084 |
| 0.8% | 0.117 | 0.096 |
| 0.9% | 0.235 | 0.108 |

The curves cross at about **0.75%**. Error Correction Zoo lists **0.8%** for the gross code with BP-OSD, so this reproduces the published value.

## Relay-BP vs BP+OSD-CS7 (same samples, 12 cycles)
| p | shots | Relay-BP failures | BP+OSD-CS7 failures | Relay-BP ms/shot | BP+OSD ms/shot |
|---|---|---|---|---|---|
| 0.4% | 600 | **0** | 6 | 121 | 450 |
| 0.5% | 400 | **3** | 26 | 181 | 518 |

Relay-BP settings follow IBM's Relay-BP-S: 301 legs, 80 iterations in the first leg and 60 in later legs, 5 solutions, gamma0 0.125, gamma in [-0.24, 0.66], plain min-sum.
Relay-BP has no OSD stage. It fails about **9x less often** than BP+OSD-CS7 at p = 0.5% and is about **3x faster** on CPU.
This matches IBM's 2025 result that Relay-BP beats BP-OSD on the gross code. Every shot Relay-BP failed at p = 0.5% also failed under BP+OSD.

Code capacity at p = 4%: Relay-BP had 3 failures in 300 shots and BP+OSD-0 had 9 (validate_e1 [7]).

## Not done
- Comparing against IBM's exact per-cycle numbers at p = 0.1%. Those need about 10^7 shots and IBM's full idle-noise model.
- Real-time latency. Relay-BP runs here batched in PyTorch; IBM's real-time FPGA/ASIC path is out of scope.
