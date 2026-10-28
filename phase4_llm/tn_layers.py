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


def mpo_reconstruct(cores: List[torch.Tensor]) -> torch.Tensor:
    """Back to dense (2^row_bits, 2^col_bits) -- for validation only."""
    t = cores[0]                                   # (1, p0, b0)
    for c in cores[1:]:                            # (a, p, b)
        t = torch.einsum('...b,bqc->...qc', t, c)  # contract trailing bond
    t = t[0, ..., 0]                               # (2,)*k
    return t.reshape(-1, t.numel() // (2 ** 0)).reshape(
        int(2 ** (len(cores) // 1)), -1) if False else _flat(t)


def _flat(t: torch.Tensor) -> torch.Tensor:
    k = t.dim()
    half = k // 2
    return t.reshape(2 ** half, 2 ** (k - half))


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
        return mpo_reconstruct(list(self.cores))[
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
        cores = mpo_decompose(W.detach().float(), chi_max=chi_probe, eps=eps)
        Wh = mpo_reconstruct(cores)[:W.shape[0], :W.shape[1]].to(W.dtype)
        err = torch.linalg.norm(W - Wh) / (torch.linalg.norm(W) + 1e-12)
        scores.append(float(err))
    return scores
