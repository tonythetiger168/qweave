# qweave

![validation](https://img.shields.io/badge/validation-63%2F63%20checks-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![backends](https://img.shields.io/badge/backends-numpy%20%7C%20torch%20%7C%20Aer%20%7C%20Stim-informational)
![platform](https://img.shields.io/badge/platform-CPU%20%7C%20CUDA-lightgrey)
![license](https://img.shields.io/badge/license-Apache--2.0-blue)

A complete quantum computing simulation platform with three development phases, designed to support the Quantum Computing Hardware R&D project.

**Version:** 1.0  
**Date:** September 2026  
**Hardware Project:** [quantum-rd-plan](https://github.com/tonythetiger168/quantum-rd-plan)

---

## Quickstart

```bash
git clone https://github.com/tonythetiger168/qweave.git
cd qweave
pip install numpy scipy qiskit-aer stim pymatching torch   # optional accelerators

python3 __main__.py --phase 1     # run milestone demos
```

```python
from phase1_basic.circuit import Circuit

# one IR drives every backend
c = Circuit(2).h(0).cnot(0, 1)
c.measure(0, 0).measure(1, 1)

c.run_statevector().get_probabilities()          # native numpy engine
c.run_clifford_shots(1000, seed=1)               # Pauli-frame sampler
c.run_clifford_shots(1000, noise={'after_2q': ('depolarize', 0.01)})

from phase1_basic.backends import BACKENDS, auto_backend
auto_backend(c)                                  # -> 'native-clifford-sampler'
BACKENDS['torch-statevector'].run(c)             # GPU when CUDA available
```

Validate everything (58 checks incl. gold-standard comparison with Google Stim):

```bash
python3 phase1_basic/validate_p0.py     # P0 engines          (14 checks)
python3 phase2_noise/validate_p0b.py    # IR/trajectory/QEC   (22 checks)
python3 phase3_scale/validate_p0c.py    # backends/MPS/IPEA/VQE (22 checks)
```

---

## Overview

This software suite provides four complementary quantum simulators and a complete algorithm library for validating quantum hardware designs before fabrication.

| Simulator | Qubit Scale | Memory | Best For |
|-----------|------------|--------|----------|
| **State-Vector** (numpy / torch-CUDA) | ~20 qubits CPU, ~30+ GPU | O(2^n) | General circuits, verification |
| **Clifford/Stabilizer** | 1000+ qubits | O(n^2) | Error correction, surface codes |
| **MPS (canonical, adaptive chi)** | ~100 qubits | O(n*chi^2) | Low-entanglement systems |
| **Trajectory (noisy)** | ~25 qubits | O(2^n) | Noise modeling at scale (DM would need O(4^n)) |

---

## Three-Phase Development Roadmap

### Phase I: Basic Simulators (2026 Q4 - 2027 Q1)

**Goal:** Establish core simulation infrastructure

| Milestone | Target | Status |
|-----------|--------|--------|
| **P1.1** | State-vector simulator (up to 20 qubits) | Complete |
| **P1.2** | Clifford/stabilizer simulator (1000+ qubits) | Complete |
| **P1.3** | MPS tensor network simulator (up to 100 qubits) | Complete |
| **P1.4** | Gate fidelity validation against theory | Complete |

**Key Deliverables:**
- Full gate library: H, X, Y, Z, S, T, RX, RY, RZ, U3, CNOT, CZ, SWAP, Toffoli
- Measurement: projective, multi-shot, all-qubit
- Analysis: probabilities, Pauli expectations, entanglement entropy, fidelity

### Phase II: Noise & Error Correction (2027 Q2 - 2027 Q4)

**Goal:** Model realistic hardware noise and validate error correction codes

| Milestone | Target | Status |
|-----------|--------|--------|
| **P2.1** | Depolarizing, amplitude damping, phase damping channels | Complete |
| **P2.2** | Surface code distance-3 encoding & syndrome extraction | Complete |
| **P2.3** | Logical error rate measurement vs physical error rate | Complete |
| **P2.4** | Quantum process tomography for gate characterization | Complete |

**Key Deliverables:**
- Kraus operator implementation for all standard noise channels
- Surface code with 9 data + 8 ancilla qubits
- Repetition code logical error rate benchmark
- Process tomography framework

### Phase III: Large-Scale & Hybrid Algorithms (2028 - 2029)

**Goal:** Simulate algorithms that will run on the hardware

| Milestone | Target | Status |
|-----------|--------|--------|
| **P3.1** | VQE for molecular ground state (H2) | Complete |
| **P3.2** | QAOA for MaxCut optimization | Complete |
| **P3.3** | Quantum Phase Estimation | Complete |
| **P3.4** | Benchmark suite comparing all simulators | Complete |

**Key Deliverables:**
- Hardware-efficient ansatz with parameter optimization
- QAOA with p-layer circuit and mixer Hamiltonian
- QPE with inverse QFT
- Comprehensive benchmark suite

---

## Quick Start

```bash
# Run all phases
python -m qpu_sim_suite

# Run specific phase
python -m qpu_sim_suite --phase 1
python -m qpu_sim_suite --phase 2
python -m qpu_sim_suite --phase 3
```

### Example: Bell State

```python
from phase1_basic.simulators import StateVectorSimulator

sim = StateVectorSimulator(n_qubits=2)
sim.h(0)
sim.cnot(0, 1)
sim.print_state()
# Output:
# |00>: 0.707107
# |11>: 0.707107

counts = sim.measure_all(shots=1024)
# {'00': ~512, '11': ~512}
```

### Example: VQE for H2

```python
from phase3_scale.algorithms import VQE

vqe = VQE(n_qubits=2, n_layers=2)
ham = VQE.hydrogen_molecule_hamiltonian()
result = vqe.optimize(hamiltonian=ham, max_iter=100)
print(f"Ground state energy: {result['best_energy']:.6f} Ha")
# Output: ~-1.137 Ha (exact)
```

### Example: 1000-Qubit Clifford Circuit

```python
from phase1_basic.simulators import CliffordSimulator

sim = CliffordSimulator(n_qubits=1000)
sim.h(0)
for i in range(999):
    sim.cnot(i, i+1)
# Completes in < 1 second, uses only ~4 MB memory
```

---

## P0 Optimization (2026-09)

Phase-1 simulators were re-engineered with three techniques, keeping the
original naive implementations (`*Naive` classes) as cross-validation
references. All results verified by `phase1_basic/validate_p0.py`
(14/14 checks PASS), including head-to-head comparison against Google's
Stim (gold standard).

**Techniques**
1. **State-vector**: gate fusion (consecutive 1q gates merged into one
   2x2 before touching amplitudes), strided axis application (no
   full-array transpose per gate), diagonal-gate fast paths (RZ/Z/S/T/CZ
   as O(N) phase multiply), slice-swap bit flips (X/CNOT/SWAP/Toffoli).
2. **Clifford engine**: bit-packed uint8 tableau, 16-entry LUT rowsum
   (vectorized over qubits), correct stabilizer-row measurement
   criterion with GF(2) representation solve for deterministic outcomes.
3. **Affine Pauli-frame sampler** (`CliffordSimulator.sample`): one
   noiseless reference pass records each measurement as an affine GF(2)
   form `out = base XOR <coins,c> XOR <errs,e>`; a symbolic frame pass
   tracks error Paulis as GF(2) coefficient masks; shots are evaluated
   vectorized over uint64 limbs (the same architecture as Stim).
   Measurement correlations (e.g. GHZ) are exact.

**Measured speedups** (this machine, numpy single-threaded)

| Benchmark | naive | optimized | speedup |
|---|---|---|---|
| SV 2q-gate latency, n=16 | 0.924 ms | 0.050 ms | **18.5x** |
| SV 2q-gate latency, n=20 | 4.20 ms | 0.864 ms | 4.9x |
| SV 2q-gate latency, n=24 | 332 ms | 158 ms | 2.1x (memory-bound) |
| Clifford throughput, n=200 | 59k gates/s | 2.48M gates/s | **42x** |
| Clifford throughput, n=1000 | 2.3k gates/s | 251k gates/s | **110x** |
| Noisy sampling, n=1000, 20k shots | 7.2 shots/s | 5,489 shots/s | **>760x** |

Validation coverage: random-circuit fidelity vs naive (n=6/10/14,
bit-endian corrected), toffoli/Bell/correlated-measurement, Clifford
vs naive tableau (n=5/40), Clifford-vs-statevector statistics, and
distribution agreement with Stim for noiseless, X-error and depolarizing
circuits (marginals and GHZ parity).

---

## Phase B (2026-09): Interop, Noisy Scaling, QEC Pipeline

All 22 checks in `phase2_noise/validate_p0b.py` PASS.

**B-1 Circuit IR + OpenQASM** (`phase1_basic/circuit.py`)
- `Gate`/`Circuit` dataclass IR drives every backend (state-vector,
  Clifford sampler).
- OpenQASM 2.0 export/import: full qelib1 subset incl. parametrized
  gates (`rx/ry/rz/u3`) with `pi`-expression round-tripping, `sdg`/`tdg`,
  `measure -> c[i]`; verified by parse-back equivalence.
- `to_sampler_ops(noise=...)`: IR -> Clifford affine-sampler op list with
  per-gate noise hooks (`after_1q`/`after_2q`).

**B-2 Monte-Carlo trajectory simulator** (`phase2_noise/trajectory.py`)
- Replaces O(4^n) density-matrix evolution with pure-state trajectories:
  per-shot memory O(2^n) -> noisy simulation at n=20 in ~1.7 s for 50
  shots where the exact density matrix would need ~17.6 TB.
- Pauli (depolarizing) channels pre-sampled vectorized over shots with
  the exact one-of-{I,X,Y,Z} convention; amplitude/phase damping sample
  their Kraus index on the fly from state-dependent probabilities.
- Validated against analytic results (<X> = sqrt(1-lam) under phase
  damping; <Z> = 2g-1 under amplitude damping; 1-4p/3 depolarizing) and
  cross-checked against the exact density-matrix channels.
- Also fixed a latent bug in the original `NoiseChannels._apply_kraus`:
  the bra-side contraction hit the ket axis, silently turning every
  single-qubit channel into a no-op.

**B-3 Detector error model + PyMatching** (`phase2_noise/detector_model.py`)
- End-to-end QEC workflow on our own engines: Circuit IR builds a
  distance-d repetition-memory experiment; the Clifford sampler produces
  shots; detectors are syndrome differences; the DEM is enumerated by
  injecting each single fault and propagating it through the noiseless
  circuit (Clifford simulator); PyMatching decodes with MAP weights
  -ln(p) (boundary edges for measurement-type faults, star-decomposed
  auxiliary nodes for >2-detector hyperedges).
- Validated: noiseless run gives zero detection events; every enumerated
  fault decodes to its actual logical flip; logical error rate with
  decoding < without, increasing with physical p.

---

## Phase D (2026-09): GPU-ready Torch backend

**`phase1_basic/gpu_backend.py`** -- device-agnostic state-vector engine in
PyTorch: identical code path on CPU (validated here, fidelity 1.0 vs the
numpy engine at n=6/10/14) and on CUDA (auto-selected by `auto_backend()`
when `torch.cuda.is_available()`). On CPU the numpy engine remains faster
(torch-cpu is 3-14x slower at n=20-24 due to copy overhead); the torch
backend exists to unlock GPU memory bandwidth on H100-class hardware,
where the survey (arXiv:2308.01999) shows gate application at ~90% of
peak memory bandwidth (state-vector ceiling ~2^33-2^35 amplitudes).

---

## Phase C (2026-09): Backends, MPS, IPEA, Sparse VQE

All 22 checks in `phase3_scale/validate_p0c.py` PASS.

**C-1 Pluggable backends** (`phase1_basic/backends.py`)
- `NativeSVBackend`, `NativeCliffordBackend`, `AerBackend` (Qiskit Aer
  statevector), `StimBackend` (Google Stim) -- one Circuit IR drives all.
- `auto_backend()`: decision-tree selection (Clifford-only -> frame
  sampler; <=24 qubits -> state-vector; otherwise escalate).
- Verified: Aer agrees with the native engine to fidelity > 1-1e-6;
  Stim agrees with the native sampler on measurement statistics.

**C-2 Canonical MPS** (`phase3_scale/mps_canonical.py`)
- Mixed canonical form maintained by QR sweeps; two-site updates with
  ADAPTIVE truncation (keep singular values up to a discarded-weight
  threshold eps, capped by chi_max); Schmidt-spectrum entanglement
  entropy at any bond.
- Validated against the state-vector simulator at n=6/10/14 (fidelity
  1.0); supersedes the P1.3 MPS, whose gate contraction was never
  cross-validated.

**C-3 IPEA** (`phase3_scale/ipe.py`)
- Iterative phase estimation with ONE ancilla + classical feedback:
  n+1 qubits instead of n+k for standard QPE (e.g. k=20 counting
  qubits: 4 amplitudes vs 2^21).
- Exactly-representable phases extracted to machine precision; arbitrary
  phases via majority voting (single-shot IPEA has the standard 4/pi^2
  success probability).

**C-4 Sparse VQE** (`phase3_scale/vqe_sparse.py`)
- `SparsePauliHamiltonian`: memory O(#terms), expectations evaluated on
  amplitudes with vectorized bit-twiddling.
- Exact parameter-shift gradients (verified against finite differences
  to <1e-4); simple gradient-descent driver converges to the -ZZ ground
  state.

---

## Project Structure

```
qweave/
|-- __main__.py                 # Entry point (--phase 1|2|3 milestones)
|-- phase1_basic/
|   |-- simulators.py           # State-vector, Clifford, MPS (+ Naive refs)
|   |-- circuit.py              # B-1: Circuit IR + OpenQASM 2.0 I/O
|   |-- backends.py             # C-1: Aer/Stim/native adapters + auto-select
|   |-- gpu_backend.py          # D: torch CUDA/CPU state-vector engine
|   |-- validate_p0.py          # P0 cross-validation suite vs naive & Stim
|-- phase2_noise/
|   |-- noise_and_ec.py         # Noise channels, surface code, QPT
|   |-- trajectory.py           # B-2: Monte-Carlo trajectory simulator
|   |-- detector_model.py       # B-3: detector error model + PyMatching
|   |-- validate_p0b.py         # Phase B validation suite (22 checks)
|-- phase3_scale/
|   |-- algorithms.py           # VQE, QAOA, QPE, benchmarks
|   |-- mps_canonical.py        # C-2: canonical MPS, adaptive chi
|   |-- ipe.py                  # C-3: iterative phase estimation
|   |-- vqe_sparse.py           # C-4: sparse Hamiltonians + parameter-shift
|   |-- validate_p0c.py         # Phase C validation suite (27 checks)
|-- docs/
|   |-- README.md               # This file
```

---

## Integration with Hardware Roadmap

| Hardware Milestone | Software Validation |
|-------------------|---------------------|
| M1.3: Single-qubit T1 > 100 us | P2.1: Amplitude damping with gamma=0.001 |
| M1.4: Two-qubit gate fidelity > 99% | P2.4: Process tomography verification |
| M2.3: First logical qubit (d=3 surface code) | P2.2: Surface code syndrome extraction |
| M2.4: 1,024-qubit engineering sample | P1.2: Clifford simulation of full chip |
| M3.3: Quantum LDPC code | P2.3: Logical error rate < 10^-5 |
| M3.4: 100+ logical qubits | P3.1-3.3: Algorithm benchmarks |

---

## Dependencies

- **Core:** NumPy (only dependency for basic simulators)
- **Optional:** Qiskit, Cirq, PennyLane, Stim, QuTiP, Quimb (for cross-validation)

---

## License

Proprietary - Quantum Computing Hardware R&D Project
