"""
Phase C validation suite.

[1] Backend adapters: Aer & Stim agree with native engines; auto-select
[2] MPS canonical: adaptive chi fidelity vs state-vector; entropy check
[3] IPEA: phase reconstruction error vs bit precision; memory claim
[4] Sparse VQE: expectation vs exact; parameter-shift vs finite diff; H2-ish
"""

import sys, time
import numpy as np

sys.path.insert(0, ".")   # run from project root

from phase1_basic.circuit import Circuit
from phase1_basic.simulators import StateVectorSimulator
from phase1_basic.backends import (NativeSVBackend, NativeCliffordBackend,
                                   AerBackend, StimBackend, auto_backend)
from phase3_scale.mps_canonical import MPSCanonical
from phase3_scale.ipe import run_ipea, diagonal_unitary_round_fn
from phase3_scale.vqe_sparse import (SparsePauliHamiltonian, vqe_gradient,
                                     vqe_minimize)

PASS = True
def check(name, cond, detail=""):
    global PASS
    s = "PASS" if cond else "FAIL"
    if not cond: PASS = False
    print(f"  [{s}] {name} {detail}")

# ------------------------------------------------------------------
print("=" * 70); print("[1] Backend adapters"); print("=" * 70)

rng = np.random.default_rng(0)
c = Circuit(4)
for _ in range(20):
    q = int(rng.integers(4))
    g = ['h', 'x', 'y', 'z', 's'][int(rng.integers(5))]
    getattr(c, g)(q)
a, b = int(rng.integers(4)), int(rng.integers(4))
if a != b:
    c.cnot(a, b)
for q in range(4):
    c.rz(float(rng.uniform(0, 2 * np.pi)), q)

native = NativeSVBackend().run(c)
aer = AerBackend().run(c)
check("Aer statevector fidelity vs native",
      float(np.abs(np.vdot(native, aer)) ** 2) > 1 - 1e-6)

cc = Circuit(3).h(0).cnot(0, 1).cnot(1, 2)
for q in range(3):
    cc.measure(q, q)
ours = NativeCliffordBackend().run(cc, shots=4000, seed=7)
theirs = StimBackend().run(cc, shots=4000, seed=7)
check("Stim marginal q0 vs native",
      abs(float(np.mean(ours[:, 0])) - float(np.mean(theirs[:, 0]))) < 0.04,
      f"native={np.mean(ours[:,0]):.3f} stim={np.mean(theirs[:,0]):.3f}")

check("auto_backend picks Clifford sampler",
      auto_backend(cc) == "native-clifford-sampler")
check("auto_backend picks SV for non-Clifford",
      auto_backend(c) in ("native-statevector", "torch-statevector"),
      f"-> {auto_backend(c)}")  # GPU hosts route to the torch SV backend
try:
    big = Circuit(30).rx(0.1, 0)
    auto_backend(big)
    check("auto_backend rejects oversized", False)
except ValueError:
    check("auto_backend rejects oversized", True)

# ------------------------------------------------------------------
print("=" * 70); print("[2] MPS canonical + adaptive chi"); print("=" * 70)

for n in (6, 10, 14):
    mps = MPSCanonical(n, chi_max=64, eps=1e-14)
    sv = StateVectorSimulator(n)
    rng = np.random.default_rng(n)
    for _ in range(4 * n):
        q = int(rng.integers(n))
        g = ['h', 'x', 'y', 'z', 's'][int(rng.integers(5))]
        getattr(mps, g)(q); getattr(sv, g)(q)
        if q < n - 1 and rng.random() < 0.5:
            mps.cnot(q, q + 1); sv.cnot(q, q + 1)
    sv._flush_all()
    fid = mps.fidelity(sv.state)
    check(f"MPS-SV fidelity n={n} (adaptive chi)",
          fid > 1 - 1e-8, f"fid={fid:.10f} max_chi={mps.max_bond()}")
    mps.canonicalize(n // 2)
    e = mps.entanglement_entropy(n // 2)
    check(f"entropy finite n={n}", 0 <= e <= n, f"S={e:.3f}")

# truncation actually engages at small chi_max
mps = MPSCanonical(12, chi_max=4, eps=1e-14)
sv = StateVectorSimulator(12)
rng = np.random.default_rng(99)
for _ in range(60):
    q = int(rng.integers(12))
    getattr(mps, 'h', None)(q) if False else mps.h(q)
    sv.h(q)
    if q < 11:
        mps.cnot(q, q + 1); sv.cnot(q, q + 1)
sv._flush_all()  # unused below, but keep parity
check("chi_max=4 keeps bond bounded", mps.max_bond() <= 4,
      f"max_chi={mps.max_bond()}")

# ------------------------------------------------------------------
print("=" * 70); print("[3] IPEA"); print("=" * 70)

for k in (6, 10, 16):     # exactly representable phase: strict bound
    est, bits = run_ipea(1, diagonal_unitary_round_fn(0.25), k)
    check(f"IPEA exact phi=0.25 k={k}", abs(est - 0.25) < 1e-12,
          f"est={est:.12f}")

from collections import Counter
for phi_true in (0.3, 0.777):   # arbitrary phase: majority vote over shots
    ests = [run_ipea(1, diagonal_unitary_round_fn(phi_true), 10)[0]
            for _ in range(31)]
    est = Counter(ests).most_common(1)[0][0]
    check(f"IPEA majority phi={phi_true}", abs(est - phi_true) < 2 ** -9,
          f"est={est:.6f}")

# memory comparison: QPE needs 1+k qubits, IPEA needs 2
k = 20
print(f"  QPE would need {2 ** (1 + k) / 1e9:.1f} GB amplitudes; "
      f"IPEA uses 4 (n_system=1) at k={k}")

# ------------------------------------------------------------------
print("=" * 70); print("[4] Sparse VQE + parameter-shift"); print("=" * 70)

# expectation vs direct: H = ZZ + 0.5 X I - 0.3 I Y on |++>
H = SparsePauliHamiltonian.from_list([("ZZ", 1.0), ("XI", 0.5), ("IY", -0.3)])
sv = StateVectorSimulator(2); sv.h(0); sv.h(1)
sv._flush_all()
e_sparse = H.expectation(sv.state, 2)
e_direct = sv.get_expectation('ZZ') + 0.5 * sv.get_expectation('XI') \
    - 0.3 * sv.get_expectation('IY')
check("sparse expectation vs direct", abs(e_sparse - e_direct) < 1e-9,
      f"sparse={e_sparse:.9f} direct={e_direct:.9f}")

# parameter-shift gradient vs finite differences (2-qubit ansatz)
def circuit_fn(p):
    sim = StateVectorSimulator(2)
    sim.ry(p[0], 0); sim.ry(p[1], 1); sim.cnot(0, 1); sim.ry(p[2], 1)
    return sim.state

H2 = SparsePauliHamiltonian.from_list([("ZZ", 1.0), ("XI", 0.7), ("IZ", 0.4)])
p0 = np.array([0.3, -0.6, 1.1])
g_ps, e0 = vqe_gradient(circuit_fn, p0, H2, 2)
g_fd = np.zeros(3)
for i in range(3):
    dp = np.zeros(3); dp[i] = 1e-5
    g_fd[i] = (H2.expectation(circuit_fn(p0 + dp), 2)
               - H2.expectation(circuit_fn(p0 - dp), 2)) / 2e-5
check("parameter-shift vs finite-diff",
      np.allclose(g_ps, g_fd, atol=1e-4),
      f"ps={np.round(g_ps,5)} fd={np.round(g_fd,5)}")

# minimize <ZZ> (ground state of -ZZ is Bell pair, energy -1)
Hzz = SparsePauliHamiltonian.from_list([("ZZ", -1.0)])
best_p, best_e = vqe_minimize(circuit_fn, np.zeros(3), Hzz, 2,
                              iters=150, lr=0.15)
check("VQE converges toward -1", best_e < -0.95, f"E={best_e:.5f}")

# ------------------------------------------------------------------
print("=" * 70); print("[5] Torch GPU-ready backend"); print("=" * 70)

try:
    from phase1_basic.gpu_backend import TorchSVSimulator, TorchSVBackend, default_device
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    print("  (torch not installed -- skipped)")

if HAVE_TORCH:
    check("torch device resolves (cpu here)", str(default_device()) in
          ("cpu", "cuda"))
    for n in (6, 10, 14):
        t = TorchSVSimulator(n)
        s = StateVectorSimulator(n)
        rng = np.random.default_rng(100 + n)
        for _ in range(4 * n):
            q = int(rng.integers(n))
            r = rng.random()
            if r < 0.5:
                g = ['h', 'x', 'z', 's'][int(rng.integers(4))]
                getattr(t, g)(q); getattr(s, g)(q)
            elif r < 0.75:
                a = float(rng.uniform(0, 2 * np.pi))
                t.rx(a, q); s.rx(a, q)
            else:
                ang = float(rng.uniform(0, 2 * np.pi))
                t.rz(ang, q)
                s.rz(ang, q)
            if q < n - 1 and rng.random() < 0.5:
                t.cnot(q, q + 1); s.cnot(q, q + 1)
        s._flush_all()
        fid = t.fidelity(s.state)
        check(f"torch fidelity n={n}", fid > 1 - 1e-9, f"fid={fid:.10f}")

    from phase1_basic.backends import BACKENDS
    check("torch registered in backend registry",
          "torch-statevector" in BACKENDS)

# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("ALL PHASE C VALIDATION PASSED" if PASS else "SOME CHECKS FAILED")
print("=" * 70)
