# E2: Pauli propagation and IBM's 127-qubit kicked Ising (2026-11-01)

Code:
- `phase3_scale/pauli_prop.py`: numpy engine for general Pauli rotations. It also has the IBM Eagle coupling map and an exact light-cone statevector.
- `phase3_scale/pauli_prop_fast.py`: compiled engine (numba) for the kicked-Ising circuit, about 14x faster.

Checks: `phase3_scale/validate_e2.py`, which runs in CI in about 20 s. Everything here ran on a 4-core CPU with no GPU.

The circuit is the one from Kim et al., Nature 618, 500 (2023). It runs on 127 qubits of an Eagle heavy-hex processor with 144 couplers. Each Trotter step is RX(θh) on every qubit followed by RZZ(-π/2) on every coupler.

## Correctness
- Without truncation, Pauli propagation equals the exact statevector to 1e-15. This holds on 16 cases: an 8-qubit ring, Clifford and non-Clifford angles, and X, Y, Z and mixed observables.
- The compiled engine returns the same numbers as the numpy engine to 1e-9.

## 5 steps: the regime IBM could check exactly
At 5 steps each single-qubit observable depends on at most 19 qubits (its light cone), so the exact answer is a small statevector. The table gives the mean magnetization over all 127 qubits:

| θh | exact | Pauli prop, threshold 1e-2 | 1e-3 | 1e-4 | max terms at 1e-4 |
|---|---|---|---|---|---|
| 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1 |
| π/8 | 0.9048 | 0.8971 | 0.9033 | 0.9045 | 4.5k |
| π/4 | 0.4528 | 0.4229 | 0.4525 | 0.4535 | 440k |
| 3π/8 | 0.0373 | 0.0233 | 0.0352 | 0.0372 | 617k |
| π/2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |

At a threshold of 1e-4 the error is below 1e-3 at every angle. At 2e-5, single-qubit values agree with the exact ones to machine precision. The full 127-qubit sweep took under 2 minutes per angle with the numpy engine.

## 20 steps: ⟨Z62⟩, the regime IBM called beyond exact simulation
These are the compiled-engine results as the coefficient threshold is tightened:

| θh | 1e-3 | 3e-4 | 1e-4 | 3e-5 | 1e-5 | 3e-6 |
|---|---|---|---|---|---|---|
| π/8 | 0.857 | 0.847 | 0.837 | 0.841 | 0.841 | 0.844 (12M terms, 142 s) |
| 0.2π | 0.493 | 0.384 | 0.374 | 0.457 (26M terms, 297 s) | | |
| π/4 | 0.253 | 0.198 | 0.157 (8.6M terms) | | | |

- **θh = π/8:** converged to about **0.84 ± 0.005**.
- **θh ≥ 0.2π:** **not converged**. The value still moves by about 0.1 between thresholds, and the next threshold would need more than 15 GB of memory. With plain coefficient truncation the 20-step, larger-angle points are out of reach on this machine.

The published classical reproductions (Begušić, Gray and Chan, Sci. Adv. 2024; Rudolph et al. 2025) reached convergence. Their engines add steps that qweave doesn't have yet:
- merging strings related by the circuit's symmetries
- truncating by weight as well as by coefficient
- a GPU or C++ backend

I could not check their exact ⟨Z62⟩ numbers because arxiv.org is not reachable from this environment.

### Weight truncation does not fix it
`kicked_ising_expectation_fast(..., max_weight=w)` drops strings with more than w non-identity sites after each ZZ layer. The table shows θh = 0.2π, 20 steps, coefficient threshold 1e-4:

| max weight | 6 | 8 | 10 | 12 | none |
|---|---|---|---|---|---|
| ⟨Z62⟩ | 0.652 | 0.555 | 0.555 | 0.410 | 0.374 |

The value moves back toward the uncapped result as w grows, so the coefficient threshold is the real limit.
The numpy engine applies the cap after every gate, while the compiled engine applies it after each whole ZZ layer. Their capped numbers therefore differ slightly (0.584 vs 0.580 at w = 8, threshold 1e-3).

## Next steps
1. Merge strings related by the circuit's symmetries, as Begušić & Chan do.
2. Run the compiled engine's hash table on a GPU (one H100, a few dollars) to reach thresholds of 1e-6.
3. Add Pauli noise (damping each string by its weight) so noisy hardware data can be compared directly.
