"""
QL3: entanglement diagnostics + QSA (quantum-inspired sparse attention)
======================================================================
1. Diagnostics: an attention map A (N x N) is an operator between the
   query and key "subsystems". Its operator-Schmidt spectrum across the
   query|key cut is just its singular values; the entanglement entropy
       S = -sum p_i ln p_i,   p_i = s_i^2 / sum s^2
   and effective rank exp(S) measure how many "modes" the head mixes.
   Low-entanglement heads are compressible -> they get fewer key blocks.

2. QSA: block-sparse causal attention. Each query block attends to
   sink block(s) + a local window + the top-k_h key blocks scored with
   mean-pooled q.k (a rank-1-per-block estimate). k_h is set per head
   from the measured effective rank. CUDA path: torch flex_attention
   with a BlockMask built from the selected kv blocks; reference path:
   dense masked softmax (CPU / validation).
"""

import math
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 1. entanglement diagnostics
# ----------------------------------------------------------------------
def attention_entanglement(A: torch.Tensor) -> Dict[str, torch.Tensor]:
    """A: (..., N, N) attention probabilities -> per-map entropy stats."""
    s = torch.linalg.svdvals(A.float())
    p = s ** 2 / (s ** 2).sum(-1, keepdim=True).clamp_min(1e-30)
    S = -(p * torch.log(p.clamp_min(1e-30))).sum(-1)
    row_H = -(A.float() * torch.log(A.float().clamp_min(1e-30))).sum(-1).mean(-1)
    return {'entropy': S, 'eff_rank': torch.exp(S), 'row_entropy': row_H}


def head_budgets(eff_rank: torch.Tensor, base_topk: int, kmin: int = 2,
                 kmax: Optional[int] = None) -> List[int]:
    """Scale the per-head top-k by effective rank relative to the median."""
    med = eff_rank.median().clamp_min(1e-6)
    k = torch.round(base_topk * eff_rank / med).long().clamp(min=kmin)
    if kmax is not None:
        k = k.clamp(max=kmax)
    return k.tolist()


# ----------------------------------------------------------------------
# 2. block selection
# ----------------------------------------------------------------------
def qsa_select(q: torch.Tensor, k: torch.Tensor, block: int = 128,
               topk=8, local: int = 2, sink: int = 1) -> torch.Tensor:
    """
    q: (B, H, N, D), k: (B, Hkv, N, D). Returns bool (B, H, nq, nk):
    which key blocks each query block attends to (causal, incl. diagonal).
    topk: int or per-head list.
    """
    B, H, N, D = q.shape
    Hkv = k.shape[1]
    nb = (N + block - 1) // block
    pad = nb * block - N
    qp = F.pad(q, (0, 0, 0, pad)).reshape(B, H, nb, block, D).float().mean(3)
    kp = F.pad(k, (0, 0, 0, pad)).reshape(B, Hkv, nb, block, D).float().mean(3)
    kp = kp.repeat_interleave(H // Hkv, dim=1)
    sc = qp @ kp.transpose(-1, -2)                               # (B,H,nb,nb)
    i = torch.arange(nb, device=q.device)
    causal = i[None, :] <= i[:, None]
    forced = causal & ((i[:, None] - i[None, :] < local) | (i[None, :] < sink))
    sc = sc.masked_fill(~causal | forced, float('-inf'))
    rank = sc.argsort(-1, descending=True).argsort(-1)          # 0 = best
    kh = torch.as_tensor(topk if isinstance(topk, (list, tuple)) else [topk] * H,
                         device=q.device).view(1, H, 1, 1)
    sel = (rank < kh) & torch.isfinite(sc)
    return sel | forced


# ----------------------------------------------------------------------
# 3. attention kernels
# ----------------------------------------------------------------------
def qsa_reference(q, k, v, sel, block=128, scale=None):
    """Dense masked softmax with the block selection (validation path)."""
    B, H, N, D = q.shape
    g = H // k.shape[1]
    k = k.repeat_interleave(g, 1); v = v.repeat_interleave(g, 1)
    m = sel.repeat_interleave(block, 2).repeat_interleave(block, 3)[..., :N, :N]
    m = m & torch.ones(N, N, dtype=torch.bool, device=q.device).tril()
    s = (q.float() @ k.float().transpose(-1, -2)) * (scale or D ** -0.5)
    s = s.masked_fill(~m, float('-inf'))
    return (torch.softmax(s, -1) @ v.float()).to(q.dtype)


_flex = None


def _get_flex():
    global _flex
    if _flex is None:
        from torch.nn.attention.flex_attention import flex_attention
        _flex = torch.compile(flex_attention, dynamic=False)
    return _flex


def qsa_block_mask(sel: torch.Tensor, N: int, block: int = 128):
    from torch.nn.attention.flex_attention import BlockMask
    nb = sel.shape[-1]
    diag = torch.eye(nb, dtype=torch.bool, device=sel.device)
    full = sel & ~diag
    part = sel & diag

    def pack(m):
        num = m.sum(-1).to(torch.int32)
        idx = (~m).to(torch.int8).argsort(-1, stable=True).to(torch.int32)
        return num, idx

    kn, ki = pack(part)
    fn, fi = pack(full)

    def causal(b, h, qi, kvi):
        return qi >= kvi

    return BlockMask.from_kv_blocks(kn, ki, fn, fi, BLOCK_SIZE=block,
                                    mask_mod=causal, seq_lengths=(N, N))


def qsa_attention(q, k, v, topk=8, block=128, local=2, sink=1, scale=None):
    """Block-sparse causal attention. q (B,H,N,D); k,v (B,Hkv,N,D)."""
    sel = qsa_select(q, k, block, topk, local, sink)
    if q.is_cuda:
        bm = qsa_block_mask(sel, q.shape[2], block)
        return _get_flex()(q, k, v, block_mask=bm, scale=scale,
                           enable_gqa=k.shape[1] != q.shape[1])
    return qsa_reference(q, k, v, sel, block, scale)


def density(sel: torch.Tensor) -> float:
    """Fraction of causal blocks actually computed."""
    nb = sel.shape[-1]
    return float(sel.float().sum() / (sel.shape[0] * sel.shape[1] * nb * (nb + 1) / 2))


# ----------------------------------------------------------------------
# 4. transformers integration (prefill / PPL eval)
# ----------------------------------------------------------------------
QSA_CFG = {'topk': 8, 'block': 128, 'local': 2, 'sink': 1,
           'per_layer_topk': None, 'min_len': 1024}


def qsa_hf_attention(module, query, key, value, attention_mask, scaling=None,
                     dropout=0.0, **kw):
    """AttentionInterface-compatible: returns (B, N, H, D), None."""
    N = query.shape[2]
    if N < QSA_CFG['min_len']:          # short sequences -> dense SDPA
        out = F.scaled_dot_product_attention(
            query, key, value, is_causal=True, scale=scaling,
            enable_gqa=key.shape[1] != query.shape[1])
        return out.transpose(1, 2).contiguous(), None
    topk = QSA_CFG['topk']
    plt = QSA_CFG['per_layer_topk']
    if plt is not None and getattr(module, 'layer_idx', None) is not None:
        topk = plt[module.layer_idx]
    out = qsa_attention(query, key, value, topk=topk, block=QSA_CFG['block'],
                        local=QSA_CFG['local'], sink=QSA_CFG['sink'], scale=scaling)
    return out.transpose(1, 2).contiguous(), None


def register_hf():
    from transformers import AttentionInterface
    AttentionInterface.register("qsa", qsa_hf_attention)
    try:                    # transformers >= 4.53 also needs a mask builder
        from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
        AttentionMaskInterface.register("qsa", sdpa_mask)
    except ImportError:
        pass
