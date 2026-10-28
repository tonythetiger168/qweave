"""
Phase B validation suite.

[1] Circuit IR + OpenQASM roundtrip & backend equivalence
[2] Trajectory sampler vs exact density matrix (analytic + numeric)
[3] Detector error model + PyMatching pipeline
[4] Benchmarks: trajectory memory scaling vs density matrix
"""

import sys, time
import numpy as np

sys.path.insert(0, ".")   # run from project root

from phase1_basic.circuit import Circuit
from phase1_basic.simulators import StateVectorSimulator, CliffordSimulator
from phase2_noise.trajectory import TrajectorySimulator, exact_noisy_expectation
from phase2_noise.detector_model import (
    RepetitionCodeMemory, DemBuilder, Matcher, run_memory_experiment)

PASS = True
def check(name, cond, detail=""):
    global PASS
    s = "PASS" if cond else "FAIL"
    if not cond: PASS = False
    print(f"  [{s}] {name} {detail}")

# ------------------------------------------------------------------
print("=" * 70); print("[1] Circuit IR + OpenQASM"); print("=" * 70)

c = Circuit(3)
c.h(0).cnot(0, 1).cnot(1, 2)
c.rz(np.pi / 3, 2).rx(np.pi / 2, 1).u3(np.pi / 2, 0.3, -0.7, 0)
c.measure(0, 0).measure(1, 1).measure(2, 2)
qasm = c.to_qasm()
c2 = Circuit.from_qasm(qasm)
check("QASM roundtrip gate names", [g.name for g in c.gates] == [g.name for g in c2.gates])
check("QASM roundtrip targets", [g.targets for g in c.gates] == [g.targets for g in c2.gates])
check("QASM roundtrip params", all(np.allclose(g.params, h.params) for g, h in zip(c.gates, c2.gates) if g.params))

# IR -> statevector equals direct API (measurement-free subcircuit)
c_nomeas = Circuit(3)
c_nomeas.h(0).cnot(0, 1).cnot(1, 2)
c_nomeas.rz(np.pi / 3, 2).rx(np.pi / 2, 1).u3(np.pi / 2, 0.3, -0.7, 0)
direct = StateVectorSimulator(3)
direct.h(0); direct.cnot(0, 1); direct.cnot(1, 2); direct.rz(np.pi / 3, 2); direct.rx(np.pi / 2, 1)
direct.u3(np.pi / 2, 0.3, -0.7, 0)
direct._flush_all()          # u3 is buffered; flush before raw-state compare
via_ir = c_nomeas.run_statevector()
check("IR->SV fidelity vs direct API", via_ir.fidelity(direct.state) > 1 - 1e-9)

# IR -> Clifford shots vs SV statistics (Clifford subcircuit)
cc = Circuit(2).h(0).cnot(0, 1)
cc.measure(0, 0)
oc = cc.run_clifford_shots(3000, seed=2)
sv = StateVectorSimulator(2); sv.h(0); sv.cnot(0, 1)
sv_counts = sv.measure_all(shots=3000)
frac_sv = sum(v for k, v in sv_counts.items() if k[0] == '1') / 3000
check("IR->Clifford marginal vs SV", abs(float(np.mean(oc[:, 0])) - frac_sv) < 0.04,
      f"cliff={np.mean(oc[:,0]):.3f} sv={frac_sv:.3f}")

# ------------------------------------------------------------------
print("=" * 70); print("[2] Trajectory vs exact density matrix"); print("=" * 70)

def traj_expectation(ops, pauli, n, shots=4000, seed=3):
    t = TrajectorySimulator(n)
    _, states = t.sample(ops, shots=shots, seed=seed, keep_states=True)
    P = np.array([[1.0]])
    for ch in pauli:
        M = {'I': np.eye(2), 'X': np.array([[0, 1], [1, 0]]),
             'Y': np.array([[0, -1j], [1j, 0]]), 'Z': np.diag([1, -1])}[ch]
        P = np.kron(P, M)
    vals = np.einsum('i,ij,j->', states[0], P, states[0])  # placeholder
    # compute per shot: <psi|P|psi>
    exps = np.array([np.vdot(s, P @ s).real for s in states])
    return float(exps.mean())

# analytic: |+> phase damping -> <X> = 1 - lam
for lam in (0.1, 0.5, 0.9):
    ops = [('h', 0), ('phase_damp', 0, lam)]
    got = traj_expectation(ops, 'X', 1, seed=4)
    want = np.sqrt(1 - lam)   # K1=diag(0,sqrt(lam)) convention
    check(f"phase_damp |+> <X> lam={lam}", abs(got - want) < 0.03,
          f"got={got:.3f} want={want:.3f}")

# analytic: |1> amplitude damping -> <Z> = 2*gamma - 1
for gm in (0.1, 0.5, 1.0):
    ops = [('x', 0), ('amp_damp', 0, gm)]
    got = traj_expectation(ops, 'Z', 1, seed=5)
    check(f"amp_damp |1> <Z> gamma={gm}", abs(got - (2 * gm - 1)) < 0.03,
          f"got={got:.3f} want={2*gm-1:.3f}")

# depolarizing |+> -> <X> = 1 - 4p/3  (exact numeric cross-check too)
for p in (0.05, 0.2):
    ops = [('h', 0), ('depolarize', 0, p)]
    got = traj_expectation(ops, 'X', 1, seed=6)
    exact = exact_noisy_expectation(ops, 'X', 1)
    check(f"depolarize |+> <X> p={p}", abs(got - (1 - 4 * p / 3)) < 0.03 and abs(exact - (1 - 4 * p / 3)) < 1e-9,
          f"traj={got:.4f} exact={exact:.6f}")

# 3-qubit GHZ with local depolarizing: ZZ correlation matches exact DM
p = 0.1
ops = [('h', 0), ('cnot', 0, 1), ('cnot', 1, 2), ('depolarize', 0, p)]
got = traj_expectation(ops, 'ZZI', 3, shots=3000, seed=7)
exact = exact_noisy_expectation(ops, 'ZZI', 3)
check("GHZ depolarized <ZZI> vs exact DM", abs(got - exact) < 0.04,
      f"traj={got:.4f} exact={exact:.4f}")

# measurement statistics from trajectories (noisy Bell)
ops = [('h', 0), ('cnot', 0, 1), ('depolarize', 0, 0.1), ('mz', 0), ('mz', 1)]
t = TrajectorySimulator(2)
outc = t.sample(ops, shots=4000, seed=8)
corr = float(np.mean(outc[:, 0] == outc[:, 1]))
check("noisy Bell measurement correlation > 0.9", corr > 0.9, f"corr={corr:.4f}")

# ------------------------------------------------------------------
print("=" * 70); print("[3] Detector error model + PyMatching"); print("=" * 70)

exp = RepetitionCodeMemory(distance=3, rounds=3)
circ = exp.build_circuit()

# noiseless: zero detections, zero logical errors
ops = circ.to_sampler_ops()
out0 = CliffordSimulator(exp.n).sample(ops, shots=500, seed=9)
det0 = exp.detector_matrix(out0)
check("noiseless: no detection events", det0.sum() == 0)
check("noiseless: no logical errors", exp.actual_logical(out0).sum() == 0)

# fault enumeration sanity
builder = DemBuilder(exp, circ)
faults = builder.enumerate_faults()
check("fault count > 0", len(faults) > 0, f"n={len(faults)}")
check("fault detector weights bounded (star-decomposed)",
      all(len(f[1]) <= 8 for f in faults), f"max={max(len(f[1]) for f in faults)}")
log_flags = {f[0]: f[2] for f in faults}
# any single data-qubit X must commute or anti-commute consistently:
# at least some faults must flip the logical (boundary data errors)
check("some faults flip logical", any(f[2] for f in faults))
check("some faults are detected", any(len(f[1]) > 0 for f in faults))

# single-fault decodability: for each detectable fault, feeding its
# detector pattern yields predicted logical == actual logical flag
matcher = Matcher(faults, exp.detector_count())
ok, tot = 0, 0
for label, dets, logical, _p in faults:
    if not dets:
        continue
    events = np.zeros((1, exp.detector_count()), dtype=np.uint8)
    for d in dets:
        events[0, d] = 1
    pred = matcher.decode(events)[0]
    tot += 1
    ok += int(pred == logical)
check("single-fault decode logical consistency", ok / tot > 0.95,
      f"{ok}/{tot}")

# p sweep: decoded beats undecoded, monotonic-ish
res = {}
for p in (0.001, 0.005, 0.02):
    r = run_memory_experiment(distance=3, rounds=3, shots=2000,
                              p_gate=p, p_meas=p, seed=11 + int(p * 1000))
    res[p] = r
    print(f"    p={p:<6g} decoded={r['logical_error_decoded']:.4f} "
          f"undecoded={r['logical_error_undecoded']:.4f} faults={r['n_faults']}")
check("decoded < undecoded at low p",
      res[0.001]['logical_error_decoded'] < res[0.001]['logical_error_undecoded'])
check("logical error increases with p",
      res[0.02]['logical_error_decoded'] > res[0.001]['logical_error_decoded'])

# ------------------------------------------------------------------
print("=" * 70); print("[4] Benchmarks: trajectory vs density matrix scaling"); print("=" * 70)

for n in (12, 16, 20):
    ops = [('h', 0)] + [('cnot', i, i + 1) for i in range(n - 1)] \
          + [('depolarize', 0, 0.01), ('mz', 0)]
    t0 = time.perf_counter()
    TrajectorySimulator(n).sample(ops, shots=50, seed=1)
    dt = time.perf_counter() - t0
    dm_bytes = 16 * (4 ** n) / 1e9   # complex128 density matrix
    sv_bytes = 16 * (2 ** n) / 1e9
    print(f"  n={n:2d}: trajectory 50 shots in {dt:6.2f}s | "
          f"SV mem {sv_bytes:7.3f} GB | DM mem {dm_bytes:8.1f} GB (infeasible)")
check("trajectory runs at n=20 (DM would need 16 TB)", True)

print("\n" + "=" * 70)
print("ALL PHASE B VALIDATION PASSED" if PASS else "SOME CHECKS FAILED")
print("=" * 70)
