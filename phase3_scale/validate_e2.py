"""
Phase E2 validation suite: Pauli propagation.

[1] Exact agreement with a statevector (8-qubit ring, Clifford and non-Clifford angles)
[2] IBM Eagle coupling map (127 qubits, 144 couplers, heavy-hex)
[3] 127-qubit kicked Ising, 5 steps: Pauli propagation vs exact light-cone statevector
[4] Clifford point theta_h = pi/2 needs no truncation and stays a single string
[5] Compiled (numba) engine gives the same numbers as the numpy engine
"""

import sys, time
import numpy as np

sys.path.insert(0, ".")   # run from project root

from phase3_scale.pauli_prop import (
    PauliSum, eagle_edges, kicked_ising_expectation, kicked_ising_statevector, light_cone)

PASS = True
def check(name, cond, detail=""):
    global PASS
    s = "PASS" if cond else "FAIL"
    if not cond: PASS = False
    print(f"  [{s}] {name} {detail}")

# ------------------------------------------------------------------
print("=" * 70); print("[1] Exact vs statevector"); print("=" * 70)

ring = [(i, i + 1) for i in range(7)] + [(0, 7)]
worst = 0.0
for ops in ({3: 'Z'}, {2: 'X'}, {5: 'Y'}, {1: 'Z', 4: 'X', 6: 'Y'}):
    for th, tj in ((0.3, -np.pi / 2), (1.1, 0.7)):
        for fr in (False, True):
            pp = kicked_ising_expectation(PauliSum.from_label(8, ops), ring, 3, th, tj,
                                          min_abs=0, final_rx=fr)
            sv = kicked_ising_statevector(range(8), ring, ops, 3, th, tj, final_rx=fr)
            worst = max(worst, abs(pp - sv))
check("untruncated propagation equals statevector (16 cases)", worst < 1e-10,
      f"(max error {worst:.1e})")

# ------------------------------------------------------------------
print("=" * 70); print("[2] IBM Eagle coupling map"); print("=" * 70)

E = eagle_edges()
deg = np.bincount(np.array(E).ravel(), minlength=127)
check("127 qubits, 144 couplers, degree <= 3", len(E) == 144 and len(deg) == 127
      and deg.min() >= 1 and deg.max() == 3)

# ------------------------------------------------------------------
print("=" * 70); print("[3] 127 qubits, 5 steps vs exact light cone"); print("=" * 70)

th = np.pi / 4
qs = [0, 13, 62, 100, 126]
t0 = time.time()
ex = [kicked_ising_statevector(light_cone(E, [q], 4), E, {q: 'Z'}, 5, th) for q in qs]
t_ex = time.time() - t0
t0 = time.time()
pp = [kicked_ising_expectation(PauliSum.from_label(127, {q: 'Z'}), E, 5, th, min_abs=2e-5)
      for q in qs]
t_pp = time.time() - t0
err = max(abs(a - b) for a, b in zip(ex, pp))
check("theta_h = pi/4: <Z_q> within 1e-3 of exact on 5 qubits (threshold 2e-5)", err < 1e-3,
      f"(max error {err:.1e}; exact {t_ex:.1f}s, Pauli propagation {t_pp:.1f}s)")

# ------------------------------------------------------------------
print("=" * 70); print("[4] Clifford point"); print("=" * 70)

st = {}
v = kicked_ising_expectation(PauliSum.from_label(127, {62: 'Z'}), E, 20, np.pi / 2,
                             min_abs=0, stats=st)
check("theta_h = pi/2, 20 steps: one string throughout, exact", st['peak_terms'] == 1,
      f"(<Z_62> = {v:.3f}, peak terms {st['peak_terms']})")

# ------------------------------------------------------------------
print("=" * 70); print("[5] Compiled engine"); print("=" * 70)

from phase3_scale import pauli_prop_fast
if pauli_prop_fast.available():
    a = kicked_ising_expectation(PauliSum.from_label(127, {62: 'Z'}), E, 20, 0.2 * np.pi,
                                 min_abs=1e-3)
    b = pauli_prop_fast.kicked_ising_expectation_fast(127, {62: 'Z'}, E, 20, 0.2 * np.pi,
                                                      min_abs=1e-3)
    w = max(abs(pauli_prop_fast.kicked_ising_expectation_fast(8, ops, ring, 3, 1.1, min_abs=0)
                - kicked_ising_statevector(range(8), ring, ops, 3, 1.1))
            for ops in ({3: 'Z'}, {5: 'Y'}, {1: 'Z', 4: 'X', 6: 'Y'}))
    check("numba engine: exact on the ring, equals numpy engine at 127 qubits x 20 steps",
          w < 1e-10 and abs(a - b) < 1e-9, f"(ring error {w:.1e}; <Z_62> {a:.6f} vs {b:.6f})")
else:
    print("  [SKIP] numba not installed")

print("=" * 70)
print("ALL CHECKS PASS" if PASS else "SOME CHECKS FAILED")
sys.exit(0 if PASS else 1)
