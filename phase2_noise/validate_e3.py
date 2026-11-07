"""
Phase E3 validation suite: per-platform noise profiles.

[1] A uniform profile builds exactly the same circuit as the float-p API
[2] Every profile is physical (rates in (0, 5%), valid idle Pauli channel)
[3] Google Willow profile reproduces the published surface-code Lambda = 2.14
[4] Lambda falls when every error rate is doubled
"""

import sys, time
import numpy as np

sys.path.insert(0, ".")   # run from project root

from phase2_noise.noise_profiles import NoiseProfile, PROFILES
from phase2_noise.surface_code import rotated_memory_z_circuit, logical_error_rate, per_round

PASS = True
def check(name, cond, detail=""):
    global PASS
    s = "PASS" if cond else "FAIL"
    if not cond: PASS = False
    print(f"  [{s}] {name} {detail}")


def lam(prof, shots=(20_000, 60_000, 200_000), seed=3):
    eps = {}
    for d, n in zip((3, 5, 7), shots):
        r = 3 * d
        eps[d] = per_round(logical_error_rate(rotated_memory_z_circuit(d, r, prof), n,
                                              seed=seed + d)['ler'], r)
    return float(np.sqrt(eps[3] / eps[7])), eps

# ------------------------------------------------------------------
print("=" * 70); print("[1] Uniform profile == float p"); print("=" * 70)

same = all(str(rotated_memory_z_circuit(d, d, p)) ==
           str(rotated_memory_z_circuit(d, d, NoiseProfile.uniform(p)))
           for d in (3, 5) for p in (0.001, 0.005))
check("identical stim circuits for d=3,5 and p=0.1%, 0.5%", same)

# ------------------------------------------------------------------
print("=" * 70); print("[2] Profiles are physical"); print("=" * 70)

ok = True
for key, prof in PROFILES.items():
    rates = (prof.p1, prof.p2, prof.p_meas, prof.p_reset)
    idle = prof.idle_pauli()
    ok &= all(0 < r < 0.05 for r in rates)
    ok &= idle is None or (all(v >= 0 for v in idle) and sum(idle) < 0.05)
    ok &= bool(prof.source)
    print(f"    {key:15s} p2={prof.p2:.4f} meas={prof.p_meas:.4f} idle={idle}")
check(f"{len(PROFILES)} profiles: rates in (0, 5%), idle channel valid, source cited", ok)

# ------------------------------------------------------------------
print("=" * 70); print("[3] Google Willow surface-code Lambda"); print("=" * 70)

t0 = time.time()
lw, eps = lam(PROFILES["google_willow"])
check("Willow profile: Lambda within 0.25 of the published 2.14", abs(lw - 2.14) < 0.25,
      f"(Lambda = {lw:.2f}; eps_7 = {eps[7]*100:.3f}%/cycle vs published 0.143%; "
      f"{time.time() - t0:.0f}s)")

# ------------------------------------------------------------------
print("=" * 70); print("[4] Twice the noise"); print("=" * 70)

l2, _ = lam(PROFILES["google_willow"].scaled(2.0), shots=(20_000, 40_000, 60_000))
check("Willow x2: Lambda drops and stays near or below 1", l2 < lw and l2 < 1.3,
      f"(Lambda {lw:.2f} -> {l2:.2f})")

print("=" * 70)
print("ALL CHECKS PASS" if PASS else "SOME CHECKS FAILED")
sys.exit(0 if PASS else 1)
