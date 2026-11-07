"""
E3: surface-code memory under each platform's noise profile.

For d = 3, 5, 7 (rounds = 3d) it reports the per-cycle logical error rate and
Lambda = eps_d / eps_{d+2} (geometric mean over 3->5 and 5->7). The reference
is Google Willow: Lambda = 2.14 +- 0.02 and 0.143% per cycle at d = 7
(Nature 638, 920, 2025).

Run from the project root:  python scripts/e3_platform_lambda.py [out.json]
"""

import json, sys, time
import numpy as np

sys.path.insert(0, ".")
from phase2_noise.noise_profiles import PROFILES
from phase2_noise.surface_code import rotated_memory_z_circuit, logical_error_rate, per_round


def platform_row(key, prof, shots=(200_000, 400_000, 1_000_000), seed=7):
    eps = {}
    for d, n in zip((3, 5, 7), shots):
        r = 3 * d
        res = logical_error_rate(rotated_memory_z_circuit(d, r, prof), n, seed=seed + d)
        eps[d] = (per_round(res['ler'], r), res['errors'], n)
    lam = [eps[3][0] / eps[5][0] if eps[5][0] > 0 else float('inf'),
           eps[5][0] / eps[7][0] if eps[7][0] > 0 else float('inf')]
    return dict(platform=key, name=prof.name,
                eps={d: v[0] for d, v in eps.items()},
                errors={d: v[1] for d, v in eps.items()},
                shots={d: v[2] for d, v in eps.items()},
                lambda_35=lam[0], lambda_57=lam[1],
                lambda_geo=float(np.sqrt(lam[0] * lam[1])))


if __name__ == "__main__":
    out = []
    for key, prof in PROFILES.items():
        t0 = time.time()
        row = platform_row(key, prof)
        row['seconds'] = time.time() - t0
        print(json.dumps(row), flush=True)
        out.append(row)
    if len(sys.argv) > 1:
        json.dump(out, open(sys.argv[1], "w"), indent=1)
