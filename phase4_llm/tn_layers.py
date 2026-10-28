"""
QL1: Tensor-Network (MPO/TT) compression for Linear layers
===========================================================
Ports the adaptive-truncation SVD logic from MPSCanonical (phase3_scale)
to weight-matrix compression, following the CompactifAI / Minima line of
prior art (see docs/QUANTUM_LLM_ROADMAP.md).

A Linear weight W (out, in) is viewed as a 2^(ko+ki) tensor with one
binary index per bit of (row | col); TT-SVD sweeps it into MPO cores
(chi_l, 2, chi_r) with ADAPTIVE bond dimension: keep singular values up
to a discarded-weight threshold eps, capped by chi_max.

Engine: torch -- CUDA when available (same code trains on GPU).
"""

from typing import List, Optional
import torch


# ----------------------------------------------------------------------
def _adaptive_cut(S: torch.Tensor, eps: float, chi_max: int) -> int:
    """First chi keeping cumulative squared-weight >= (1-eps) of total."""
    w = S ** 2
    total = w.sum()
    if total <= 0:
        return 1
    csum = torch.cumsum(w, 0)
    keep = int(torch.searchsorted(csum, (1 - eps) * total).item()) + 1
    return max(1, min(keep, S.numel(), chi_max))


def mpo_decompose(W: torch.Tensor, chi_max: int = 64, eps: float = 1e-3,
                  row_bits: Optional[int] = None,
                  col_bits: Optional[int] = None) -> List[torch.Tensor]:
    """
    TT-SVD of a 2D weight into 3D MPO cores [(chi_l, 2, chi_r), ...],
    ordered row-bits first then col-bits. out/in need not be powers of
    two (zero-padded internally).
    """
    out, in_ = W.shape
    if row_bits is None:
        row_bits = max(1, (out - 1).bit_length())
    if col_bits is None:
        col_bits = max(1, (in_ - 1).bit_length())
    T = torch.zeros(2 ** row_bits, 2 ** col_bits, dtype=W.dtype, device=W.device)
    T[:out, :in_] = W
    k = row_bits + col_bits
    t = T.reshape(*([2] * k))

    cores: List[torch.Tensor] = []
    chi_l = 1
    for i in range(k - 1):
        t = t.reshape(chi_l * 2, -1)
        U, S, Vh = torch.linalg.svd(t, full_matrices=False)
        chi = _adaptive_cut(S, eps, chi_max)
        U, S, Vh = U[:, :chi], S[:chi], Vh[:chi, :]
        cores.append(U.reshape(chi_l, 2, chi))
        t = (S[:, None] * Vh).reshape(chi, *([2] * (k - 1 - i)))
        chi_l = chi
    cores.append(t.reshape(chi_l, 2, 1))
    return cores


def mpo_num_params(cores: List[torch.Tensor]) -> int:
    return int(sum(c.numel() for c in cores))


def mpo_reconstruct(cores: List[torch.Tensor],
                    row_bits: Optional[int] = None) -> torch.Tensor:
    """Back to dense (2^row_bits, 2^col_bits) -- for validation only."""
    t = cores[0]                                   # (1, p0, b0)
    for c in cores[1:]:                            # (a, p, b)
        t = torch.einsum('...b,bqc->...qc', t, c)  # contract trailing bond
    t = t[0, ..., 0]                               # (2,)*k
    k = t.dim()
    if row_bits is None:
        row_bits = k // 2
    return t.reshape(2 ** row_bits, 2 ** (k - row_bits))


# ----------------------------------------------------------------------
class MPOLinear(torch.nn.Module):
    """Drop-in replacement for nn.Linear with MPO-compressed weight."""

    def __init__(self, in_features: int, out_features: int,
                 chi_max: int = 64, eps: float = 1e-3,
                 bias: bool = True, device=None, dtype=None):
        super().__init__()
        factory = {'device': device, 'dtype': dtype or torch.float32}
        W = torch.nn.Linear(in_features, out_features, bias=False,
                            **factory).weight.data
        self.row_bits = max(1, (out_features - 1).bit_length())
        self.col_bits = max(1, (in_features - 1).bit_length())
        self.cores = torch.nn.ParameterList(
            [torch.nn.Parameter(c.contiguous()) for c in
             mpo_decompose(W, chi_max, eps, self.row_bits, self.col_bits)])
        self.in_features = in_features
        self.out_features = out_features
        self.bias = torch.nn.Parameter(torch.zeros(out_features, **factory)) \
            if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in) -> (..., out)
        lead = x.shape[:-1]
        ki, ko = self.col_bits, self.row_bits
        xin = x.to(self.cores[0].dtype)
        if xin.shape[-1] < 2 ** ki:          # pad non-power-of-two inputs
            xpad = torch.zeros(*lead, 2 ** ki, dtype=xin.dtype, device=xin.device)
            xpad[..., :xin.shape[-1]] = xin
            xin = xpad
        t = xin.reshape(*lead, *([2] * ki))

        # --- contract col cores, right to left; result bond = a of leftmost
        for i in range(ki - 1, -1, -1):
            c = self.cores[ko + i]                   # (a, p, b)
            if i == ki - 1:
                t = torch.einsum('...p,qpx->...qx', t, c)[..., 0]
            else:
                t = torch.einsum('...px,qpx->...q', t, c)

        # --- contract row cores, right to left; bond consumed first at the
        #     right edge; phys bits accumulate in reverse order
        for j in range(ko - 1, -1, -1):
            c = self.cores[j]                        # (a, p, b)
            t = torch.einsum('...x,qpx->...qp', t, c)
            t = t.movedim(-2, -1)                    # keep bond last
        t = t[..., 0]                                # (..., p_{ko-1}..p_0)

        # reverse row-bit axes (they accumulated right-to-left)
        y = t.reshape(*lead, *([2] * ko))
        axes = list(range(y.dim()))
        axes = axes[:len(lead)] + axes[len(lead):][::-1]
        y = y.permute(*axes).reshape(*lead, 2 ** ko)
        y = y[..., :self.out_features]
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y

    def effective_weight(self) -> torch.Tensor:
        return mpo_reconstruct(list(self.cores), row_bits=self.row_bits)[
            :self.out_features, :self.in_features]


# ----------------------------------------------------------------------
def sensitivity_profile(weights: List[torch.Tensor], chi_probe: int = 8,
                        eps: float = 1e-6) -> List[float]:
    """
    Cheap per-layer compressibility score: relative Frobenius error of the
    MPO approximation at a small probe bond dimension. LOWER = more
    compressible (allocate chi budget there first).
    """
    scores = []
    for W in weights:
        rb = max(1, (W.shape[0] - 1).bit_length())
        cores = mpo_decompose(W.detach().float(), chi_max=chi_probe, eps=eps,
                              row_bits=rb)
        Wh = mpo_reconstruct(cores, row_bits=rb)[:W.shape[0], :W.shape[1]].to(W.dtype)
        err = torch.linalg.norm(W - Wh) / (torch.linalg.norm(W) + 1e-12)
        scores.append(float(err))
    return scores


# ======================================================================
# QL2: interleaved TT-matrix (true MPO) layers
# ----------------------------------------------------------------------
# The row-bits-then-col-bits ordering above puts the whole (out | in) cut
# on ONE bond, so chi caps the matrix rank of W -- at chi=32 a 1024x1024
# weight becomes rank-32. The standard MPO (CompactifAI, TT-matrix) pairs
# out-mode k with in-mode k on the same site: cores (r, o_k, i_k, r'),
# and no single bond bounds the matrix rank.
# ======================================================================
def tt_factorize(n: int, d: int) -> List[int]:
    """Split n into d balanced factors (pads n up when it factors badly)."""
    def split(m):
        fs, p, x = [], 2, m
        while p * p <= x:
            while x % p == 0:
                fs.append(p); x //= p
            p += 1
        if x > 1:
            fs.append(x)
        out = [1] * d
        for f in sorted(fs, reverse=True):     # greedy: largest into smallest bin
            out[out.index(min(out))] *= f
        return sorted(out)
    ideal = n ** (1.0 / d)
    m = n
    while True:
        fs = split(m)
        if max(fs) <= 2.5 * ideal:
            return fs
        m += 1


class TTLinear(torch.nn.Module):
    """nn.Linear replacement with TT-matrix (interleaved MPO) weight."""

    def __init__(self, in_features: int, out_features: int, d: int = 4,
                 rank: int = 64, eps: float = 1e-3, bias: bool = True,
                 weight: Optional[torch.Tensor] = None, device=None, dtype=None):
        super().__init__()
        self.in_features, self.out_features, self.d = in_features, out_features, d
        self.in_modes = tt_factorize(in_features, d)
        self.out_modes = tt_factorize(out_features, d)
        if weight is None:
            weight = torch.nn.Linear(in_features, out_features, bias=False,
                                     device=device).weight.data
        cores = tt_matrix_decompose(weight.float(), self.out_modes, self.in_modes,
                                    rank, eps)
        dt = dtype or weight.dtype
        self.cores = torch.nn.ParameterList(
            [torch.nn.Parameter(c.to(device=device, dtype=dt).contiguous())
             for c in cores])
        self.bias = torch.nn.Parameter(torch.zeros(out_features, device=device,
                                                   dtype=dt)) if bias else None

    @property
    def ranks(self) -> List[int]:
        return [c.shape[3] for c in self.cores[:-1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        I = 1
        for m in self.in_modes:
            I *= m
        xin = x.reshape(-1, x.shape[-1]).to(self.cores[0].dtype)
        if xin.shape[-1] < I:
            xin = torch.nn.functional.pad(xin, (0, I - xin.shape[-1]))
        B = xin.shape[0]
        t = xin.reshape(B, 1, 1, I)                 # (B, Oacc, r, Irest)
        oacc = 1
        for c in self.cores:                        # (r, o, i, s)
            r, o, i, s = c.shape
            t = t.reshape(B, oacc, r, i, -1)
            t = torch.einsum('barix,rois->baosx', t, c)
            oacc *= o
            t = t.reshape(B, oacc, s, -1)
        y = t.reshape(B, oacc)[:, :self.out_features]
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*lead, self.out_features).to(x.dtype)

    def effective_weight(self) -> torch.Tensor:
        return tt_matrix_reconstruct(list(self.cores), self.out_modes,
                                     self.in_modes)[:self.out_features,
                                                    :self.in_features]


def tt_matrix_decompose(W: torch.Tensor, out_modes: List[int],
                        in_modes: List[int], rank: int = 64,
                        eps: float = 1e-3) -> List[torch.Tensor]:
    """TT-SVD of W (out, in) into cores (r, o_k, i_k, r'), adaptive eps."""
    d = len(out_modes)
    O, I = 1, 1
    for m in out_modes:
        O *= m
    for m in in_modes:
        I *= m
    T = torch.zeros(O, I, dtype=W.dtype, device=W.device)
    T[:W.shape[0], :W.shape[1]] = W
    t = T.reshape(*out_modes, *in_modes)
    perm = [p for k in range(d) for p in (k, d + k)]     # (o1,i1,o2,i2,...)
    t = t.permute(*perm).contiguous()
    # per-bond eps so the total discarded weight stays ~eps
    eps_b = eps / max(d - 1, 1)
    cores, r = [], 1
    for k in range(d - 1):
        t = t.reshape(r * out_modes[k] * in_modes[k], -1)
        U, S, Vh = torch.linalg.svd(t, full_matrices=False)
        chi = _adaptive_cut(S, eps_b, rank)
        cores.append(U[:, :chi].reshape(r, out_modes[k], in_modes[k], chi))
        t = S[:chi, None] * Vh[:chi]
        r = chi
    cores.append(t.reshape(r, out_modes[-1], in_modes[-1], 1))
    return cores


def tt_matrix_reconstruct(cores: List[torch.Tensor], out_modes: List[int],
                          in_modes: List[int]) -> torch.Tensor:
    t = cores[0]
    for c in cores[1:]:
        t = torch.einsum('...a,aoib->...oib', t, c)
    d = len(cores)
    t = t.reshape(*[m for k in range(d) for m in (out_modes[k], in_modes[k])])
    t = t.permute(*range(0, 2 * d, 2), *range(1, 2 * d, 2))
    O, I = 1, 1
    for m in out_modes:
        O *= m
    for m in in_modes:
        I *= m
    return t.reshape(O, I)


def tt_rel_error(W: torch.Tensor, d: int, rank: int, eps: float = 0.0):
    """(relative Frobenius error, #params) of a TT-matrix fit -- for budgeting."""
    om, im = tt_factorize(W.shape[0], d), tt_factorize(W.shape[1], d)
    cores = tt_matrix_decompose(W.float(), om, im, rank, eps)
    Wh = tt_matrix_reconstruct(cores, om, im)[:W.shape[0], :W.shape[1]]
    err = torch.linalg.norm(W.float() - Wh) / (torch.linalg.norm(W.float()) + 1e-12)
    return float(err), mpo_num_params(cores)


# ======================================================================
# QL2b: activation-whitened 2-site MPO (the practical default)
# ----------------------------------------------------------------------
# Measured on Qwen3-0.6B: raw-norm TT-matrix compression of trained
# weights at -30% params sends WikiText-2 PPL 21.7 -> 7.5e5 and the
# einsum chain trains at 0.12 it/s. Trained weights are NOT low-
# entanglement in the raw Frobenius norm -- but they are in the norm the
# network actually sees, ||W X||. Whitening with the calibration Gram
# matrix (SVD-LLM, arXiv 2403.07378):
#     X X^T = S S^T,   W S = U Sigma V^T,   W ~ (U_r Sigma_r)(V_r^T S^-1)
# makes the truncation error EXACTLY the singular-value tail in the
# output norm. The result is a 2-site MPO (out-site, in-site, bond r):
# two dense GEMMs, faster than the dense layer when r < in*out/(in+out).
# ======================================================================
class LowRankLinear(torch.nn.Module):
    """y = (x B^T) A^T + b  -- 2-site MPO with bond dimension `rank`."""

    def __init__(self, A: torch.Tensor, B: torch.Tensor,
                 bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.A = torch.nn.Parameter(A.contiguous())        # (out, r)
        self.B = torch.nn.Parameter(B.contiguous())        # (r, in)
        self.bias = torch.nn.Parameter(bias.clone()) if bias is not None else None
        self.in_features, self.out_features = B.shape[1], A.shape[0]

    @property
    def rank(self) -> int:
        return self.B.shape[0]

    def forward(self, x):
        y = torch.nn.functional.linear(torch.nn.functional.linear(x, self.B), self.A)
        return y + self.bias if self.bias is not None else y


def whitened_svd(W: torch.Tensor, gram: torch.Tensor, damp: float = 1e-4):
    """Returns (U*Sigma, Sigma, V^T S^-1) of W S where gram = X X^T = S S^T.
    Output-norm error at rank r is sqrt(sum_{i>=r} Sigma_i^2)."""
    W = W.double()
    G = gram.double()
    G = G + damp * G.diagonal().mean() * torch.eye(G.shape[0], dtype=G.dtype,
                                                   device=G.device)
    S = torch.linalg.cholesky(G)                            # lower, G = S S^T
    U, Sig, Vh = torch.linalg.svd(W @ S, full_matrices=False)
    Bfull = torch.linalg.solve_triangular(S, Vh, upper=False, left=False)  # Vh S^-1
    return U * Sig, Sig, Bfull


def lowrank_from_svd(US, Bfull, rank, bias=None, dtype=torch.float32):
    return LowRankLinear(US[:, :rank].to(dtype), Bfull[:rank].to(dtype),
                         None if bias is None else bias.to(dtype))
