# qweave

Quantum simulation suite plus a quantum-inspired LLM track, with every claim
checked against a named baseline and measured results checked into the repo.

- **Quantum simulators:** state-vector (numpy + torch/CUDA + cuStateVec),
  Clifford/stabilizer with Pauli-frame sampling, MPS, trajectory noise,
  circuit IR + OpenQASM I/O.
- **Error correction:** detector error models, PyMatching, rotated surface code
  and IBM gross code with BP+OSD.
- **Algorithms:** VQE (sparse), QAOA, QPE / IPEA.
- **Quantum-inspired LLM work (QL track):** tensor-network / low-rank compression,
  sparse attention and decode speed on Qwen3, benchmarked against
  [FermiQLLM 1.0](docs/FERMIQLLM_GLOBAL_INVESTIGATION.md)'s claims.

**Hardware project:** [quantum-rd-plan](https://github.com/tonythetiger168/quantum-rd-plan) ·
**Live roadmap:** [interactive site](https://deploy-mpxqplkexa.cn-beijing-plat.fcapp.run) ·
**License:** [Apache-2.0](LICENSE)

![Roadmap Demo](demo.gif)

---

## Status and latest results (2026-10-30)

All GPU numbers are from RunPod H100 80GB runs; raw reports are in
[`results/`](results/README.md). Negative results are kept on purpose.
Several phases are still in open pull requests, linked in the table.

### Quantum simulation and error correction

| Phase | What | Status | Headline result |
|---|---|---|---|
| I | State-vector, Clifford, MPS simulators | ✅ on `main` | All validation suites pass on CPU and H100 |
| II | Noise, DEM + PyMatching decoding | ✅ fixed in [#1](https://github.com/tonythetiger168/qweave/pull/1) | Decoder bug fixed: d=3 repetition code at p=0.1% goes from 2.0% (worse than undecoded 0.65%) to **0.00%**; 33/33 single faults decode correctly |
| III | VQE, QAOA, QPE/IPEA, backends | ✅ on `main` (GPU-host test fix in [#1](https://github.com/tonythetiger168/qweave/pull/1)) | 27 checks pass on CPU and H100 |
| E1 | Rotated surface code + gross code [[144,12,12]], numpy BP+OSD | 🔄 [#7](https://github.com/tonythetiger168/qweave/pull/7) | Matches Stim's generator (d=3: 0.0167 vs 0.0168); threshold ≈ 1.2%, Λ = 2.0 at p=0.5%; gross code 2.5% vs 9.0% for 12 d=7 patches with 4.1× fewer data qubits |
| E5 | GPU state vector + decoder timings | 🔄 [#8](https://github.com/tonythetiger168/qweave/pull/8) | Torch SV correct on CUDA, ~129× numpy at n=26. cuStateVec path: 3.5–9.4× faster than torch (~3.5 TB/s effective read+write), up to n=32. Gross-code decoding p=4% per shot: CUDA-Q QEC 7 µs, `ldpc` CPU 40 µs, qweave batched torch 50 µs |

### Quantum-inspired LLM track (Qwen3)

| Phase | What | Status | Headline result |
|---|---|---|---|
| QL1.5 | TN / low-rank compression + heal | ✅ measured, gate **not met** ([#1](https://github.com/tonythetiger168/qweave/pull/1)) | Raw TT/MPO is unusable on trained weights (PPL 21.7 → 7.5×10⁵). Activation-whitened low rank + distillation recovers 97–99% of the gap, but at −11% params MMLU drops 13 pt and GSM8K 34 pt |
| QL2 | Kernels + "quantum annealing" rank allocation | ✅ measured | Greedy is within 0.6–0.9% of optimal; simulated quantum annealing is 19–32% worse and ~2000× slower. Aligned low rank (r=384) is 1.14× dense, misaligned 0.39× |
| QL3 | Quantum-inspired sparse attention (QSA) | ✅ **gate passed** | 1.49× FlashAttention at 16K with +0.96% PPL; 3.47× at 32K (kernel only) |
| QL4 | Open benchmark harness | ✅ delivered | lm-eval + robustness suite; shows PPL hides reasoning collapse (in-domain heal: PPL better than dense, GSM8K 0.0) |
| QL5 | Knowledge-preserving compression (Qwen3-0.6B-Base) | 🔄 gate **FAIL** ([#4](https://github.com/tonythetiger168/qweave/pull/4)) | At −20% params: MMLU −8.7 pt [−9.5, −7.8], GSM8K −7.9 pt (gate −2 / −5); the dense control trained the same way does not drop |
| QL6 | Batch-1 decode speed | 🔄 gate **FAIL** ([#5](https://github.com/tonythetiger168/qweave/pull/5)) | Low rank is 0.84–0.95× dense at batch 1 (kernel-bound, not byte-bound); 4B speculative decoding 0.81× after harness fixes; 32K prefill end to end 1.52× (needs 2×). Triton kernels (stage K) still running |

**Bottom line so far.** Tensor-network ideas only work on LLM weights after
activation-aware whitening, and even then small models lose knowledge and math
first. Quantum-inspired annealing loses to greedy on separable problems. The one
clear win is QSA sparse attention at long context. Full write-ups:
[QL results](https://github.com/tonythetiger168/qweave/blob/claude/ql-all-phases/docs/QL_RESULTS.md) (in #1), [surpass plan](https://github.com/tonythetiger168/qweave/blob/claude/ql-all-phases/docs/FERMIQLLM_SURPASS_PLAN.md),
[QL5 results](https://github.com/tonythetiger168/qweave/blob/claude/ql5-quality/docs/QL5_RESULTS.md),
[QL6 run plan + results](https://github.com/tonythetiger168/qweave/blob/claude/ql6-decode/docs/QL6_RUN_PLAN.md),
[E5 GPU results](https://github.com/tonythetiger168/qweave/blob/claude/e5-gpu-bench/docs/E5_GPU_RESULTS.md).

### Plans
- [Simulator survey](docs/SURVEY.md) and [2025–2026 quantum research plan](https://github.com/tonythetiger168/qweave/blob/claude/quantum-research-plan/docs/QUANTUM_RESEARCH_PLAN_2026.md) ([#6](https://github.com/tonythetiger168/qweave/pull/6))
- [Quantum LLM roadmap](docs/QUANTUM_LLM_ROADMAP.md) and [FermiQLLM investigation](docs/FERMIQLLM_GLOBAL_INVESTIGATION.md)

---

## Simulators

| Simulator | Qubit scale | Memory | Best for |
|---|---|---|---|
| **State-vector** (numpy / torch CUDA / cuStateVec) | ~26 CPU, 30–32 on one H100 | O(2ⁿ) | General circuits, NISQ algorithms |
| **Clifford / stabilizer** | 1000+ | O(n²) | Error correction, surface codes |
| **MPS (tensor network)** | ~100 at low entanglement | O(n·χ²) | Low-entanglement systems |
| **Density matrix / trajectories** | ~12 exact, ~20+ by trajectories | O(4ⁿ) / O(2ⁿ) | Noise modeling, decoherence |

---

## Quick start

```bash
pip install -e .            # numpy + scipy core
pip install -e .[dev]       # + torch, stim, pymatching, qiskit-aer for cross-checks
```

```bash
qweave                      # run all simulator milestones
qweave --phase 1            # or: python __main__.py --phase 1
```

Validation suites (CI runs the first three):

```bash
python phase1_basic/validate_p0.py
```

```bash
python phase2_noise/validate_p0b.py
```

```bash
python phase3_scale/validate_p0c.py
```

```bash
python phase4_llm/validate_ql1.py
```

### Example: Bell state

```python
from phase1_basic.simulators import StateVectorSimulator

sim = StateVectorSimulator(n_qubits=2)
sim.h(0)
sim.cnot(0, 1)
sim.print_state()
# |00>: 0.707107
# |11>: 0.707107

counts = sim.measure_all(shots=1024)
# {'00': ~512, '11': ~512}
```

### Example: 1000-qubit Clifford circuit

```python
from phase1_basic.simulators import CliffordSimulator

sim = CliffordSimulator(n_qubits=1000)
sim.h(0)
for i in range(999):
    sim.cnot(i, i + 1)
```

### Example: VQE for H₂

```python
from phase3_scale.algorithms import VQE

vqe = VQE(n_qubits=2, n_layers=2)
ham = VQE.hydrogen_molecule_hamiltonian()
result = vqe.optimize(hamiltonian=ham, max_iter=100)
print(f"Ground state energy: {result['best_energy']:.6f} Ha")   # ~ -1.137 Ha
```

### GPU runs on RunPod

Each runner installs its dependencies, runs its stages under a time cap, uploads
reports to a Hugging Face dataset (env `HF_TOKEN`), then stops the pod. For
example, all QL1.5–QL4 phases on one H100 (from PR #1):

```bash
curl -sL https://raw.githubusercontent.com/tonythetiger168/qweave/claude/ql-all-phases/scripts/runpod_all_phases.sh | bash
```

---

## Project structure

```
qweave/
├── __main__.py              # CLI entry (same as the `qweave` console script)
├── phase1_basic/            # SV / Clifford / MPS simulators, circuit IR, backends, GPU backend
├── phase2_noise/            # noise channels, trajectories, DEM + PyMatching, surface code
├── phase3_scale/            # VQE, QAOA, QPE / IPEA, MPS canonical form, sparse VQE
├── phase4_llm/              # quantum-inspired LLM track (TN layers, heal, QSA, eval)
├── scripts/                 # RunPod GPU runners
├── results/                 # raw reports from the GPU runs (see results/README.md)
├── docs/                    # surveys, roadmaps, result write-ups
└── index.html               # interactive roadmap site
```

---

## Integration with the hardware roadmap

| Hardware milestone | Software validation |
|---|---|
| M1.3: single-qubit T1 > 100 µs | P2.1: amplitude damping with γ = 0.001 |
| M1.4: two-qubit gate fidelity > 99% | P2.4: process tomography |
| M2.3: first logical qubit (d=3 surface code) | P2.2 / E1: surface-code syndrome extraction and decoding |
| M2.4: 1,024-qubit engineering sample | P1.2: Clifford simulation of the full chip |
| M3.3: quantum LDPC code | E1: gross code [[144,12,12]] with BP+OSD |
| M3.4: 100+ logical qubits | P3.1–3.3: algorithm benchmarks |

---

## License

Apache License 2.0 — see [LICENSE](LICENSE). Copyright 2026 tony.
