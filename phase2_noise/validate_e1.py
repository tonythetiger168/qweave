"""
Phase E1 validation suite: modern QEC stack.

[1] Rotated surface code geometry and circuit distance
[2] Logical error rate vs stim's reference generator (same noise model)
[3] Threshold: d = 3/5/7 crossing under circuit-level noise + MWPM
[4] Gross code [[144,12,12]] construction
[5] Own BP+OSD-0 vs ldpc reference (same error samples)
[6] Gross code vs 12 distance-7 surface code patches (code capacity)
"""

import sys, time
import numpy as np

sys.path.insert(0, ".")   # run from project root

from phase2_noise.surface_code import (
    RotatedSurfaceCode, rotated_memory_z_circuit, stim_reference_circuit,
    logical_error_rate, threshold_sweep, estimate_crossing)
from phase2_noise.qldpc import (
    gross_code, code_parameters, logical_operators, gf2_rank,
    BpOsdDecoder, code_capacity_ler)

PASS = True
def check(name, cond, detail=""):
    global PASS
    s = "PASS" if cond else "FAIL"
    if not cond: PASS = False
    print(f"  [{s}] {name} {detail}")

# ------------------------------------------------------------------
print("=" * 70); print("[1] Rotated surface code geometry"); print("=" * 70)

for d in (3, 5, 7):
    code = RotatedSurfaceCode(d)
    hx, hz = code.check_matrices()
    par = code_parameters(hx, hz)
    circ = rotated_memory_z_circuit(d, d, 0.001)
    dist = len(circ.shortest_graphlike_error())
    clean = rotated_memory_z_circuit(d, d, 0.0).compile_detector_sampler(seed=0)
    det, obs = clean.sample(200, separate_observables=True)
    check(f"d={d}: {code.n_qubits} qubits = 2d^2-1, k=1",
          code.n_qubits == 2 * d * d - 1 and par['k'] == 1 and par['commute'])
    check(f"d={d}: circuit distance = d", dist == d, f"(graphlike min weight {dist})")
    check(f"d={d}: noiseless circuit fires no detector/observable",
          not det.any() and not obs.any())

# ------------------------------------------------------------------
print("=" * 70); print("[2] Match stim's reference generator"); print("=" * 70)

for d, p in ((3, 0.005), (5, 0.008)):
    shots = 50000
    a = logical_error_rate(rotated_memory_z_circuit(d, d, p), shots, seed=11)
    b = logical_error_rate(stim_reference_circuit(d, d, p), shots, seed=12)
    pooled = (a['errors'] + b['errors']) / (2 * shots)
    sigma = np.sqrt(2 * pooled * (1 - pooled) / shots)
    z = abs(a['ler'] - b['ler']) / sigma
    check(f"d={d} p={p}: ours vs stim within 4 sigma",
          z < 4, f"(ours {a['ler']:.4f}, stim {b['ler']:.4f}, z={z:.1f})")

# ------------------------------------------------------------------
print("=" * 70); print("[3] Threshold under circuit-level noise"); print("=" * 70)

t0 = time.time()
ps = [0.005, 0.008, 0.010, 0.012, 0.014, 0.016]
rows = threshold_sweep([3, 5, 7], ps, shots=40000, seed=21)
for r in rows:
    print(f"    d={r['d']} p={r['p']:.3f} per-round LER={r['ler_round']:.2e}")
below = {r['d']: r['ler_round'] for r in rows if r['p'] == 0.005}
check("below threshold (p=0.5%): LER falls with distance",
      below[3] > below[5] > below[7],
      f"(d3 {below[3]:.1e} > d5 {below[5]:.1e} > d7 {below[7]:.1e})")
lam = np.sqrt(below[3] / below[7])     # suppression per d -> d+2 step
check("error suppression Lambda > 1.5 per distance step at p=0.5%", lam > 1.5,
      f"(Lambda = {lam:.2f}; Willow reported 2.14 on hardware)")
x57, x37 = estimate_crossing(rows, 5, 7), estimate_crossing(rows, 3, 7)
check("d=5/d=7 and d=3/d=7 curves cross in [0.9%, 1.5%]",
      x57 is not None and x37 is not None and 0.009 <= x57 <= 0.015 and 0.009 <= x37 <= 0.015,
      f"(d5/d7 {x57}, d3/d7 {x37})")
print(f"    sweep time {time.time() - t0:.1f}s")

# ------------------------------------------------------------------
print("=" * 70); print("[4] Gross code [[144,12,12]]"); print("=" * 70)

hx, hz = gross_code()
par = code_parameters(hx, hz)
check("n=144, k=12", par['n'] == 144 and par['k'] == 12, f"{par}")
check("H_X H_Z^T = 0 (CSS commutation)", par['commute'])
check("weight-6 checks, every qubit in 3 X and 3 Z checks",
      par['row_weights'] == [6] and par['col_weights'] == [3])
lx, lz = logical_operators(hx, hz)
pairing = (lx.astype(np.int64) @ lz.T) % 2
check("12 X and 12 Z logicals, symplectic pairing full rank",
      lx.shape[0] == 12 and lz.shape[0] == 12 and gf2_rank(pairing) == 12)

# ------------------------------------------------------------------
print("=" * 70); print("[5] BP+OSD-0 vs ldpc reference"); print("=" * 70)

p, shots = 0.06, 1500
ours = code_capacity_ler(hx, hz, p, shots, seed=31)
try:
    from phase2_noise.qldpc import LdpcReference
    ref = code_capacity_ler(hx, hz, p, shots, decoder=LdpcReference(hz, p), seed=31)
    check("same failures as ldpc on identical samples (+-2)",
          abs(ours['errors'] - ref['errors']) <= 2,
          f"(ours {ours['errors']}, ldpc {ref['errors']} of {shots})")
except ImportError:
    print("  [SKIP] ldpc not installed")
dec = BpOsdDecoder(hz, p)
rng = np.random.default_rng(0)
e = (rng.random((200, 144)) < p).astype(np.uint8)
s = (e.astype(np.int64) @ hz.T % 2).astype(np.uint8)
c = dec.decode_batch(s)
check("every correction reproduces its syndrome",
      np.array_equal((c.astype(np.int64) @ hz.T) % 2, s))

# ------------------------------------------------------------------
print("=" * 70); print("[6] Gross code vs 12 surface code patches"); print("=" * 70)

import pymatching
p = 0.04
g = code_capacity_ler(hx, hz, p, 3000, seed=41)
shx, shz = RotatedSurfaceCode(7).check_matrices()
_, slz = logical_operators(shx, shz)
m = pymatching.Matching(shz)
rng = np.random.default_rng(42)
e = (rng.random((100000, shx.shape[1])) < p).astype(np.uint8)
corr = m.decode_batch((e.astype(np.int64) @ shz.T % 2).astype(np.uint8))
patch = float(np.mean(np.any(((e ^ corr).astype(np.int64) @ slz.T) % 2, axis=1)))
twelve = 1 - (1 - patch) ** 12
check("gross code (144 data qubits) beats 12 x d=7 patches (588) at p=4%",
      g['ler'] < twelve,
      f"(gross {g['ler']:.4f} vs 12 patches {twelve:.4f}, 4.1x fewer data qubits)")

print("=" * 70)
print("ALL CHECKS PASS" if PASS else "SOME CHECKS FAILED")
sys.exit(0 if PASS else 1)
