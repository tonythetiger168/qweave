"""
QL6/K2 -- Triton decode kernels for low-rank (2-site MPO) layers
================================================================
Stage S (H100, CUDA graphs) measured low-rank layers at 0.84-0.95x dense at
batch 1, although they read 16-45 % fewer bytes. Two causes, both visible in
the numbers:

  * launch-bound decode: dense Qwen3-0.6B ran at 301 tok/s = 3.3 ms/token,
    ~11 % of H100 bandwidth (the byte floor is ~0.36 ms). Time scales with
    the NUMBER of kernels, and low rank doubles the GEMMs per Linear.
  * skinny GEMVs: t = x B^T has only r = 384-704 outputs, so a plain GEMV
    puts work on a few dozen SMs out of 132.

So the kernels here (M <= 16 rows, i.e. decode and speculative verify):

  lowrank_b     t-partials = x B^T, split along K so ~4x more programs run;
                the partial sums are NOT reduced here: they are tiny
                (SPLIT x M x r fp32) and every program of the next kernel
                sums them while loading t (no atomics, no memset kernel)
  lowrank_a     y = A t (+ bias), t summed from the partials
  lowrank_a_glu h = silu(A_g t_g) * (A_u t_u): gate/up's two A factors and
                the SwiGLU in one kernel

Per MLP: torch low rank = B_gu, A_g, A_u, silu*mul, B_d, A_d (6 kernels);
here = 4, dense fused = 3. Per attention Linear: 2 vs dense 1. So the
kernels cut launches and fill the GPU, but low rank still launches more
kernels than dense: expect a gain where bandwidth matters (4B: ~36 % of the
byte floor at batch 1), little at 0.6B where launches dominate.

CPU testing: TRITON_INTERPRET=1 runs the same kernels in NumPy
(validate_ql6.py does this in a subprocess).
"""

import os
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                   # CPU-only installs
    HAVE_TRITON = False

INTERPRET = os.environ.get('TRITON_INTERPRET') == '1'
MAX_M = 16          # rows handled by the kernels (one tl.dot tile)
BLOCK_N = 16        # output rows per program (tl.dot minimum)
BLOCK_K = 64


def available(x: torch.Tensor) -> bool:
    return HAVE_TRITON and (x.is_cuda or INTERPRET)


if HAVE_TRITON:
    @triton.jit
    def _lowrank_b_kernel(x_ptr, b_ptr, t_ptr, M, K, R, K_PER_SPLIT,
                          stride_xm, stride_br, stride_ts, stride_tm,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """t_part[s, m, n] = sum_{k in split s} x[m, k] * B[n, k]."""
        pid_n, pid_s = tl.program_id(0), tl.program_id(1)
        rm = tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        k0 = pid_s * K_PER_SPLIT
        for kk in range(0, K_PER_SPLIT, BK):
            rk = k0 + kk + tl.arange(0, BK)
            km = (rk < K) & (rk < k0 + K_PER_SPLIT)
            xt = tl.load(x_ptr + rm[:, None] * stride_xm + rk[None, :],
                         mask=(rm[:, None] < M) & km[None, :], other=0.0)
            bt = tl.load(b_ptr + rn[:, None] * stride_br + rk[None, :],
                         mask=(rn[:, None] < R) & km[None, :], other=0.0)
            acc += tl.dot(xt, tl.trans(bt).to(xt.dtype))
        tl.store(t_ptr + pid_s * stride_ts + rm[:, None] * stride_tm + rn[None, :], acc,
                 mask=(rm[:, None] < M) & (rn[None, :] < R))

    @triton.jit
    def _load_t(t_ptr, rm, rk, M, r_lo, r_hi, SPLIT, stride_ts, stride_tm,
                BM: tl.constexpr, BK: tl.constexpr):
        """Sum the SPLIT partials of t[:, r_lo + rk] (zero outside [r_lo, r_hi))."""
        col = r_lo + rk
        msk = (rm[:, None] < M) & (col[None, :] < r_hi)
        acc = tl.zeros((BM, BK), dtype=tl.float32)
        for s in range(0, SPLIT):
            acc += tl.load(t_ptr + s * stride_ts + rm[:, None] * stride_tm + col[None, :],
                           mask=msk, other=0.0)
        return acc

    @triton.jit
    def _lowrank_a_kernel(t_ptr, a_ptr, bias_ptr, y_ptr, M, R, N, SPLIT,
                          stride_ts, stride_tm, stride_an, stride_ym,
                          HAS_BIAS: tl.constexpr,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """y[m, n] = sum_r t[m, r] A[n, r] (+ bias[n])."""
        pid = tl.program_id(0)
        rm = tl.arange(0, BM)
        rn = pid * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, R, BK):
            rk = tl.arange(0, BK)
            at = tl.load(a_ptr + rn[:, None] * stride_an + (k0 + rk)[None, :],
                         mask=(rn[:, None] < N) & ((k0 + rk)[None, :] < R), other=0.0)
            tt = _load_t(t_ptr, rm, rk, M, k0, R, SPLIT, stride_ts, stride_tm, BM, BK)
            acc += tl.dot(tt.to(at.dtype), tl.trans(at))
        if HAS_BIAS:
            acc += tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
        tl.store(y_ptr + rm[:, None] * stride_ym + rn[None, :], acc.to(y_ptr.dtype.element_ty),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))

    @triton.jit
    def _lowrank_a_glu_kernel(t_ptr, ag_ptr, au_ptr, h_ptr, M, RG, RU, N, SPLIT,
                              stride_ts, stride_tm, stride_agn, stride_aun, stride_hm,
                              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """h = silu(t[:, :RG] A_g^T) * (t[:, RG:RG+RU] A_u^T)."""
        pid = tl.program_id(0)
        rm = tl.arange(0, BM)
        rn = pid * BN + tl.arange(0, BN)
        g = tl.zeros((BM, BN), dtype=tl.float32)
        u = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, RG, BK):
            rk = tl.arange(0, BK)
            at = tl.load(ag_ptr + rn[:, None] * stride_agn + (k0 + rk)[None, :],
                         mask=(rn[:, None] < N) & ((k0 + rk)[None, :] < RG), other=0.0)
            tt = _load_t(t_ptr, rm, rk, M, k0, RG, SPLIT, stride_ts, stride_tm, BM, BK)
            g += tl.dot(tt.to(at.dtype), tl.trans(at))
        for k0 in range(0, RU, BK):
            rk = tl.arange(0, BK)
            at = tl.load(au_ptr + rn[:, None] * stride_aun + (k0 + rk)[None, :],
                         mask=(rn[:, None] < N) & ((k0 + rk)[None, :] < RU), other=0.0)
            tt = _load_t(t_ptr, rm, rk, M, RG + k0, RG + RU, SPLIT, stride_ts, stride_tm, BM, BK)
            u += tl.dot(tt.to(at.dtype), tl.trans(at))
        h = g * tl.sigmoid(g) * u
        tl.store(h_ptr + rm[:, None] * stride_hm + rn[None, :], h.to(h_ptr.dtype.element_ty),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))


def _split(K: int, R: int, sms: int = 132) -> int:
    """Split K so that (R / BLOCK_N) * split >= ~2 programs per SM."""
    progs = triton.cdiv(R, BLOCK_N)
    s = max(1, min(8, (2 * sms + progs - 1) // progs))
    while s > 1 and K // s < BLOCK_K:
        s //= 2
    return s


def lowrank_b(x2: torch.Tensor, B: torch.Tensor, split: int = 0) -> torch.Tensor:
    """x2 (M <= 16, K) contiguous -> fp32 partials (split, M, R)."""
    M, K = x2.shape
    R = B.shape[0]
    split = split or _split(K, R)
    kps = triton.cdiv(triton.cdiv(K, split), BLOCK_K) * BLOCK_K
    split = triton.cdiv(K, kps)
    t = torch.empty(split, M, R, device=x2.device, dtype=torch.float32)
    _lowrank_b_kernel[(triton.cdiv(R, BLOCK_N), split)](
        x2, B, t, M, K, R, kps, x2.stride(0), B.stride(0), t.stride(0), t.stride(1),
        BM=MAX_M, BN=BLOCK_N, BK=BLOCK_K)
    return t


def lowrank_a(t: torch.Tensor, A: torch.Tensor, bias=None, out_dtype=None) -> torch.Tensor:
    S, M, R = t.shape
    N = A.shape[0]
    y = torch.empty(M, N, device=t.device, dtype=out_dtype or A.dtype)
    _lowrank_a_kernel[(triton.cdiv(N, BLOCK_N),)](
        t, A, bias if bias is not None else A, y, M, R, N, S,
        t.stride(0), t.stride(1), A.stride(0), y.stride(0),
        HAS_BIAS=bias is not None, BM=MAX_M, BN=BLOCK_N, BK=BLOCK_K)
    return y


def lowrank_a_glu(t: torch.Tensor, Ag: torch.Tensor, Au: torch.Tensor, out_dtype=None):
    S, M, R = t.shape
    N, RG, RU = Ag.shape[0], Ag.shape[1], Au.shape[1]
    assert RG + RU == R and Au.shape[0] == N
    h = torch.empty(M, N, device=t.device, dtype=out_dtype or Ag.dtype)
    _lowrank_a_glu_kernel[(triton.cdiv(N, BLOCK_N),)](
        t, Ag, Au, h, M, RG, RU, N, S, t.stride(0), t.stride(1),
        Ag.stride(0), Au.stride(0), h.stride(0), BM=MAX_M, BN=BLOCK_N, BK=BLOCK_K)
    return h


# ----------------------------------------------------------------------
# module integration
# ----------------------------------------------------------------------
def lowrank_linear_forward(self, x):
    """Drop-in forward for LowRankLinear: Triton for <= 16 rows, else torch."""
    lead = x.shape[:-1]
    M = x.numel() // x.shape[-1]
    if M > MAX_M or not available(x):
        y = F.linear(F.linear(x, self.B), self.A)
        return y + self.bias if self.bias is not None else y
    x2 = x.reshape(M, x.shape[-1]).contiguous()
    y = lowrank_a(lowrank_b(x2, self.B), self.A, self.bias, out_dtype=x.dtype)
    return y.reshape(*lead, y.shape[-1])


def glu_mlp_forward(self, x):
    """MLP whose gate/up are a low-rank FusedGateUp (no bias, SiLU) and
    whose down_proj is LowRankLinear: 4 kernels at decode."""
    lead = x.shape[:-1]
    M = x.numel() // x.shape[-1]
    gu = self.gate_up
    if M > MAX_M or not available(x):
        g, u = gu(x)
        return self.down_proj(self.act_fn(g) * u)
    x2 = x.reshape(M, x.shape[-1]).contiguous()
    h = lowrank_a_glu(lowrank_b(x2, gu.B), gu.Ag, gu.Au, out_dtype=x.dtype)
    d = self.down_proj
    y = lowrank_a(lowrank_b(h, d.B), d.A, d.bias, out_dtype=x.dtype)
    return y.reshape(*lead, y.shape[-1])


def _is_silu(act) -> bool:
    z = torch.linspace(-6, 6, 97)
    try:
        return bool(torch.allclose(act(z), F.silu(z), atol=1e-6))
    except Exception:
        return False


@torch.no_grad()
def use_triton_(model) -> dict:
    """Route every LowRankLinear and every low-rank SwiGLU MLP (after
    fuse_siblings_) through the kernels above. Returns counts."""
    import types
    from phase4_llm.tn_layers import LowRankLinear
    n_lin = n_mlp = 0
    for m in model.modules():
        gu = getattr(m, 'gate_up', None)
        # duck-typed: ql6_decode may run as __main__, a different class object
        if type(gu).__name__ == 'FusedGateUp' and gu.lowrank and gu.bias is None and \
                isinstance(getattr(m, 'down_proj', None), LowRankLinear) and \
                _is_silu(m.act_fn):
            m.forward = types.MethodType(glu_mlp_forward, m)
            n_mlp += 1
    for m in model.modules():
        if isinstance(m, LowRankLinear):
            m.forward = types.MethodType(lowrank_linear_forward, m)
            n_lin += 1
    return {'linear': n_lin, 'glu_mlp': n_mlp}


# ----------------------------------------------------------------------
# per-layer decode latency (GPU): dense vs torch low rank vs Triton
# ----------------------------------------------------------------------
@torch.no_grad()
def bench_layers(shapes, keeps=(0.9, 0.6), M=1, align=64, iters=200, graphs=True):
    """For each (in, out) shape: microseconds per call of dense F.linear,
    torch low rank (two F.linear) and Triton low rank, each captured in a
    CUDA graph of 20 calls (so launch cost is what CUDA graphs leave)."""
    dev, dt = 'cuda', torch.bfloat16
    rows = []

    def timed(fn, x):
        fn(x); torch.cuda.synchronize()
        if graphs:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(20):
                    fn(x)
            run = g.replay
            per = 20
        else:
            run, per = (lambda: fn(x)), 1
        for _ in range(5):
            run()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters // per):
            run()
        e1.record(); torch.cuda.synchronize()
        return 1e3 * e0.elapsed_time(e1) / (iters // per * per)

    for fin, fout, name in shapes:
        x = torch.randn(M, fin, device=dev, dtype=dt)
        W = torch.randn(fout, fin, device=dev, dtype=dt) * fin ** -0.5
        t_dense = timed(lambda x: F.linear(x, W), x)
        for keep in keeps:
            r = max(align, int(round(keep * fin * fout / (fin + fout) / align)) * align)
            A = torch.randn(fout, r, device=dev, dtype=dt) * r ** -0.5
            B = torch.randn(r, fin, device=dev, dtype=dt) * fin ** -0.5
            t_torch = timed(lambda x: F.linear(F.linear(x, B), A), x)
            t_tri = timed(lambda x: lowrank_a(lowrank_b(x, B), A), x)
            err = float((lowrank_a(lowrank_b(x, B), A).float() -
                         F.linear(F.linear(x, B), A).float()).abs().max())
            rows.append({'layer': name, 'in': fin, 'out': fout, 'rank': r, 'M': M,
                         'bytes_ratio': round(fin * fout / (r * (fin + fout)), 3),
                         'us_dense': round(t_dense, 2), 'us_torch_lowrank': round(t_torch, 2),
                         'us_triton_lowrank': round(t_tri, 2),
                         'speedup_triton_vs_dense': round(t_dense / t_tri, 3),
                         'speedup_torch_vs_dense': round(t_dense / t_torch, 3),
                         'max_abs_err': err})
            print(rows[-1], flush=True)
    return rows
