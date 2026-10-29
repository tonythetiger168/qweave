"""Run by validate_ql6.py [10] under TRITON_INTERPRET=1 (CPU). Prints one
'CHECK <PASS|FAIL> <name>' line per check."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
import torch.nn.functional as F
from phase4_llm import ql6_decode as q6, ql6_kernels as K
from phase4_llm.ql5_compress import tiny_qwen3


def check(name, ok, detail=""):
    print(f"CHECK {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


torch.manual_seed(0)
worst = 0.0
for M, Kd, R, N in [(1, 1024, 640, 2048), (5, 100, 37, 77), (16, 64, 16, 16), (3, 300, 200, 50)]:
    x = torch.randn(M, Kd); B = torch.randn(R, Kd) / Kd ** .5
    A = torch.randn(N, R) / R ** .5; b = torch.randn(N)
    for split in (0, 1, 3):
        y = K.lowrank_a(K.lowrank_b(x, B, split), A, b)
        worst = max(worst, float((y - F.linear(F.linear(x, B), A, b)).abs().max()))
check("lowrank_b + lowrank_a == torch (odd shapes, M 1-16, split 1/3/auto)", worst < 1e-4, f"{worst:.1e}")
x = torch.randn(4, 96); Ag = torch.randn(80, 40) / 40 ** .5; Au = torch.randn(80, 24) / 24 ** .5
B2 = torch.randn(64, 96) / 96 ** .5
t = x @ B2.T
h = K.lowrank_a_glu(K.lowrank_b(x, B2), Ag, Au)
err = float((h - F.silu(t[:, :40] @ Ag.T) * (t[:, 40:] @ Au.T)).abs().max())
check("lowrank_a_glu == silu(A_g t_g) * (A_u t_u)", err < 1e-4, f"{err:.1e}")

m = tiny_qwen3(256, tie=False, seed=3).eval()
q6.synthetic_lowrank_(m, 0.5, 0, align=16)
q6.fuse_siblings_(m)
pr = torch.randint(1, 256, (1, 9), generator=torch.Generator().manual_seed(5))
with torch.no_grad():
    l0 = m(pr).logits
    ref = q6.greedy_generate(m, pr, 16)
    n = K.use_triton_(m)
    l1 = m(pr).logits
check("use_triton_ routes linears and SwiGLU MLPs", n == {'linear': 10, 'glu_mlp': 2}, str(n))
check("model logits unchanged", float((l0 - l1).abs().max()) < 1e-4, f"{float((l0 - l1).abs().max()):.1e}")
check("greedy decode unchanged", torch.equal(q6.greedy_generate(m, pr, 16), ref))
g, s, _ = q6.spec_engine(m, m, 'static', 48, (4,))
check("static greedy + speculative unchanged", torch.equal(g(pr, 16), ref) and torch.equal(s(pr, 16, 4)[0], ref))
with torch.no_grad():
    big = torch.randint(1, 256, (2, 20))                      # 40 rows > 16: torch fallback
    check("> 16 rows falls back to torch", torch.isfinite(m(big).logits).all())
