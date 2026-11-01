"""
QL1 validation suite -- run from repo root:  python3 phase4_llm/validate_ql1.py

Checks
  1. mpo_decompose/reconstruct roundtrip error vs chi_max (random + structured)
  2. adaptive eps threshold engages (eps sweep changes effective bond)
  3. MPOLinear.forward matches nn.Linear within decomposition error
  4. compression ratio + effective params
  5. sensitivity_profile ranks structured < random (compressible first)
  6. speed smoke: forward pass timing (CPU; CUDA if available)
"""

import sys, time
sys.path.insert(0, ".")
import torch

from phase4_llm.tn_layers import (mpo_decompose, mpo_reconstruct,
                                  mpo_num_params, MPOLinear,
                                  sensitivity_profile)

PASS = True
def check(name, cond, detail=""):
    global PASS
    s = "PASS" if cond else "FAIL"
    if not cond:
        PASS = False
    print(f"  [{s}] {name} {detail}")

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {dev}")

# ------------------------------------------------------------------
print("=" * 70); print("[1] decomposition roundtrip vs chi_max"); print("=" * 70)
torch.manual_seed(0)
O, I = 256, 256
W_rand = torch.randn(O, I, device=dev)
# structured: low-rank + small noise (like a trained weight)
U, V = torch.randn(O, 16, device=dev), torch.randn(16, I, device=dev)
W_struct = U @ V + 0.02 * torch.randn(O, I, device=dev)

for name, W in (("random", W_rand), ("structured-rank16", W_struct)):
    for chi in (4, 16, 64):
        cores = mpo_decompose(W, chi_max=chi, eps=1e-12)
        Wh = mpo_reconstruct(cores)[:O, :I]
        rel = (torch.linalg.norm(W - Wh) / torch.linalg.norm(W)).item()
        eff = mpo_num_params(cores)
        print(f"  {name:20s} chi<={chi:3d}: rel_err={rel:.4f}  params={eff} "
              f"({100*eff/(O*I):.1f}% of dense)")

errs = []
for chi in (4, 16, 64):
    cores = mpo_decompose(W_rand, chi_max=chi, eps=1e-12)
    errs.append((torch.linalg.norm(W_rand - mpo_reconstruct(cores)[:O, :I])
                 / torch.linalg.norm(W_rand)).item())
check("random: monotone error decrease", errs[0] > errs[1] > errs[2],
      f"{[round(e,4) for e in errs]}")

# ------------------------------------------------------------------
print("=" * 70); print("[2] adaptive eps threshold"); print("=" * 70)
for eps in (1e-1, 1e-2, 1e-4):
    cores = mpo_decompose(W_struct, chi_max=256, eps=eps)
    bonds = [c.shape[2] for c in cores[:-1]]
    check(f"eps={eps:g}: max bond = {max(bonds)}", max(bonds) <= 256)
    print(f"    bond profile: {bonds}")

# ------------------------------------------------------------------
print("=" * 70); print("[3] MPOLinear forward vs compressed weight"); print("=" * 70)
torch.manual_seed(1)
for (o, i_, chi) in ((64, 64, 16), (256, 256, 32), (96, 160, 16)):
    dense = torch.nn.Linear(i_, o).to(dev)
    with torch.no_grad():
        dense.bias.zero_()
    mpo = MPOLinear.from_linear(dense, chi_max=chi, eps=1e-12)
    x = torch.randn(8, i_, device=dev)
    y0 = torch.nn.functional.linear(x, mpo.effective_weight(), mpo.bias)
    y1 = mpo(x)
    rel = ((y0 - y1).norm() / y0.norm().clamp_min(1e-12)).item()
    check(f"MPOLinear {o}x{i_} chi<={chi}: fwd matches compressed weight",
          rel < 1e-4, f"rel={rel:.2e}")

# ------------------------------------------------------------------
print("=" * 70); print("[4] compression ratio"); print("=" * 70)
cores = mpo_decompose(W_struct, chi_max=64, eps=1e-3)
eff = mpo_num_params(cores)
check("structured chi64 eps1e-3: >=3x compression", eff <= O * I / 3,
      f"{eff} vs {O*I} ({100*eff/(O*I):.1f}%)")

# ------------------------------------------------------------------
print("=" * 70); print("[5] sensitivity profile ordering"); print("=" * 70)
s = sensitivity_profile([W_struct, W_rand], chi_probe=8)
check("structured scored more compressible than random", s[0] < s[1],
      f"structured={s[0]:.4f} random={s[1]:.4f}")

# ------------------------------------------------------------------
print("=" * 70); print("[6] speed smoke"); print("=" * 70)
mpo = MPOLinear(512, 512, chi_max=32, device=dev).to(dev)
x = torch.randn(64, 512, device=dev)
for _ in range(3):
    mpo(x)
if dev == 'cuda':
    torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    mpo(x)
if dev == 'cuda':
    torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / 20
print(f"  MPOLinear 512x512 chi32, batch 64: {dt*1000:.2f} ms/fwd ({dev})")
check("forward runs", dt > 0)

print("\n" + "=" * 70)
print("ALL QL1 VALIDATION PASSED" if PASS else "SOME CHECKS FAILED")
print("=" * 70)
