"""
QL6 -- make the compressed model actually faster at batch-1 decode
==================================================================
Gate (docs/FERMIQLLM_SURPASS_PLAN.md, QL6): batch-1 decode >= 1.3x the
dense model on the same engine (FermiQLLM claims "+15 % inference" with no
baseline), and 32K-token prefill >= 2x end to end with QSA top-16.

Batch-1 decode is memory-bandwidth bound: every generated token reads every
weight once, so the speed limit is (bytes read per token)^-1. That gives a
CPU-computable ceiling before any GPU money is spent:

  R1  byte roofline        decode_bytes(): weights read per token by module
                           (embedding = one row, LM head = full matrix, KV
                           cache at a given context). Ceiling speedup =
                           dense bytes / compressed bytes. At QL5's -20 % the
                           ceiling on Qwen3-0.6B is ~1.25x, i.e. low rank alone
                           CANNOT reach 1.3x there; it needs more compression
                           or to stack with INT8, or the lossless route L2.
  F6  rank alignment       align_ranks_(): re-truncate / zero-pad every bond to
                           a multiple of 64 (measured: r=384 1.14x dense,
                           r=477 0.39x). QL5 already allocates aligned ranks;
                           this fixes models built elsewhere (QL1.5).
  K1  sibling fusion       fuse_siblings_(): gate_proj and up_proj read the
                           same x, so their B factors (or dense weights) are
                           concatenated into ONE GEMM. Exact. Applied to the
                           dense baseline too, so the comparison stays fair.
      (not a single Triton kernel for x B^T A^T: at decode the r-vector
      t = x B^T is needed by every output tile, so one kernel either
      recomputes it per tile or needs a grid-wide sync; the launch overhead
      it would save is removed by CUDA graphs instead -- mode "compiled")
  L1  lossy route          compressed model alone (QL5 quality gate applies)
  L2  lossless route       speculative_generate(): dense target verifies
                           gamma tokens drafted by a compressed model in one
                           batched forward; greedy output is the target's own
                           greedy output, so quality is the dense model's.
                           Named drafts: off-the-shelf Qwen3-0.6B (for 4B) vs
                           QL-compressed drafts.

Decode SPEED depends only on shapes, not weight values, so --synthetic-keep
builds shape-matched random low-rank copies: Route L1's speed can be
measured without waiting for QL5's heal. Those copies are speed-only.

  python phase4_llm/ql6_decode.py --smoke                     # CPU, no downloads
  python phase4_llm/ql6_decode.py --roofline                  # 0.6B/4B ceilings, CPU
  python phase4_llm/ql6_decode.py --model Qwen/Qwen3-0.6B-Base \\
      --synthetic-keep 0.8,0.6 --compressed /workspace/ql5/A1/model.pt \\
      --out /workspace/ql6/s06
"""

import argparse, json, math, os, sys, time, types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import torch
import torch.nn.functional as F

from phase4_llm.tn_layers import (LowRankLinear, LowRankEmbedding, LowRankHead,
                                  regauge_)

DECODE_GATE = 1.3        # x dense, batch-1 decode, same engine
PREFILL_GATE = 2.0       # x dense at 32K tokens, end to end
TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')

# Published Qwen3 configs (config.json on the Hub), so the roofline runs
# without downloads on a meta device.
QWEN3_CONFIGS = {
    'Qwen3-0.6B': dict(vocab_size=151936, hidden_size=1024, intermediate_size=3072,
                       num_hidden_layers=28, num_attention_heads=16,
                       num_key_value_heads=8, head_dim=128, tie_word_embeddings=True),
    'Qwen3-1.7B': dict(vocab_size=151936, hidden_size=2048, intermediate_size=6144,
                       num_hidden_layers=28, num_attention_heads=16,
                       num_key_value_heads=8, head_dim=128, tie_word_embeddings=True),
    'Qwen3-4B': dict(vocab_size=151936, hidden_size=2560, intermediate_size=9728,
                     num_hidden_layers=36, num_attention_heads=32,
                     num_key_value_heads=8, head_dim=128, tie_word_embeddings=True),
}

# Fixed prompt set (no downloads): math, knowledge, code, chat, Chinese.
# Acceptance rate in L2 depends on the text, so the set is part of the report.
PROMPTS = [
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer: Let's think step by step.",
    "Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?\nAnswer: Let's think step by step.",
    "Question: Solve for x: 3x + 7 = 25. Show each step.\nAnswer:",
    "Prove that the sum of the first n odd numbers is n^2.\nProof:",
    "The derivative of f(x) = x^3 sin(x) is",
    "The French Revolution began in 1789 and",
    "Photosynthesis is the process by which plants",
    "The main causes of the First World War were",
    "In quantum mechanics, the uncertainty principle states that",
    "The capital of Australia is Canberra, which was chosen because",
    "Explain the difference between TCP and UDP in networking.",
    "A tensor network is a way to represent",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "# Python function that checks whether a string is a palindrome\ndef is_palindrome(s):",
    "SELECT name, COUNT(*) FROM orders JOIN customers ON",
    "Write a short email to a colleague asking to move tomorrow's meeting to Friday.",
    "Here are five tips for staying productive while working from home:\n1.",
    "Once upon a time, in a small village by the sea,",
    "The recipe for a simple tomato pasta sauce is:",
    "Summary of the plot of Romeo and Juliet:",
    "量子計算的基本原理是",
    "請用三句話介紹台灣的地理位置：",
    "大型語言模型的推理速度主要受限於",
    "The best way to learn a new programming language is",
]


# ----------------------------------------------------------------------
# R1. byte roofline for batch-1 decode
# ----------------------------------------------------------------------
def decode_bytes(model, ctx: int = 0, wbytes: float = 2.0, kv_bytes: float = 2.0) -> dict:
    """Bytes read from HBM per generated token at batch 1.

    Counted per MODULE (not per parameter) so tying is right: a tied
    embedding is read one row per token, the LM head reads the full matrix
    even though it is the same tensor. LowRankHead tied to a
    LowRankEmbedding reads A (vocab x r) and B. KV cache: K and V for
    `ctx` previous tokens in every layer."""
    out = {'linear': 0.0, 'lowrank': 0.0, 'embed': 0.0, 'head': 0.0, 'norm': 0.0}
    head = model.get_output_embeddings() if hasattr(model, 'get_output_embeddings') else None
    for name, m in model.named_modules():
        if isinstance(m, LowRankHead) or (head is not None and m is head):
            A, B = m.factors() if isinstance(m, LowRankHead) else (m.weight, None)
            out['head'] += A.numel() + (B.numel() if B is not None else 0)
        elif isinstance(m, LowRankLinear):
            out['lowrank'] += m.A.numel() + m.B.numel()
        elif isinstance(m, FusedGateUp):
            out['lowrank' if m.lowrank else 'linear'] += sum(
                p.numel() for n, p in m.named_parameters() if n != 'bias')
        elif isinstance(m, LowRankEmbedding):
            out['embed'] += m.rank + m.B.numel()           # one row of A, all of B
        elif isinstance(m, torch.nn.Embedding):
            out['embed'] += m.embedding_dim
        elif isinstance(m, torch.nn.Linear):
            out['linear'] += m.weight.numel()
        elif 'norm' in type(m).__name__.lower() and getattr(m, 'weight', None) is not None:
            out['norm'] += m.weight.numel()
    out = {k: v * wbytes for k, v in out.items()}
    cfg = getattr(model, 'config', None)
    kv = 0.0
    if cfg is not None and ctx:
        hd = getattr(cfg, 'head_dim', None) or cfg.hidden_size // cfg.num_attention_heads
        kv = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * hd * ctx * kv_bytes
    out['kv_cache'] = kv
    out['total'] = sum(out.values())
    return out


def meta_qwen3(name_or_cfg):
    """Qwen3ForCausalLM on the meta device: shapes only, no memory, no download."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = QWEN3_CONFIGS[name_or_cfg] if isinstance(name_or_cfg, str) else name_or_cfg
    with torch.device('meta'):
        return Qwen3ForCausalLM(Qwen3Config(**cfg))


def _align(r, a):
    return max(a, int(round(r / a)) * a) if a > 1 else max(1, int(r))


def _replace(root, target, repl):
    for name, child in root.named_children():
        if child is target:
            setattr(root, name, repl)
            return True
        if _replace(child, target, repl):
            return True
    return False


@torch.no_grad()
def synthetic_lowrank_(model, keep: float, embed_rank: int = 0, align: int = 64,
                       targets=TARGETS, seed: int = 0) -> dict:
    """Replace every target Linear by a LowRankLinear keeping ~`keep` of its
    params (rank aligned), and the (tied) embedding / head by rank
    `embed_rank` (0 = untouched). Random factors at a sane scale: SPEED
    ONLY, the outputs are meaningless. Returns {name: rank}."""
    g = torch.Generator().manual_seed(seed)
    ranks = {}
    for name, m in list(model.named_modules()):
        if isinstance(m, torch.nn.Linear) and name.rsplit('.', 1)[-1] in targets:
            o, i = m.out_features, m.in_features
            r = min(_align(keep * o * i / (o + i), align), min(o, i))
            dev, dt = m.weight.device, m.weight.dtype
            mk = (lambda *s: torch.empty(*s, device=dev, dtype=dt)) if dev.type == 'meta' else \
                 (lambda *s: (torch.randn(*s, generator=g) * s[-1] ** -0.5).to(dev, dt))
            new = LowRankLinear(mk(o, r), mk(r, i),
                                None if m.bias is None else m.bias.detach().clone())
            _replace(model, m, new)
            ranks[name] = r
    if embed_rank:
        emb = model.get_input_embeddings()
        V, D = emb.weight.shape
        dev, dt = emb.weight.device, emb.weight.dtype
        mk = (lambda *s: torch.empty(*s, device=dev, dtype=dt)) if dev.type == 'meta' else \
             (lambda *s: (torch.randn(*s, generator=g) * s[-1] ** -0.5).to(dev, dt))
        tied = bool(getattr(model.config, 'tie_word_embeddings', False))
        le = LowRankEmbedding(mk(V, embed_rank), mk(embed_rank, D))
        model.set_input_embeddings(le)
        model.set_output_embeddings(LowRankHead(tied=le) if tied else
                                    LowRankHead(mk(V, embed_rank), mk(embed_rank, D)))
        if tied:
            model.config.tie_word_embeddings = False     # factors already shared
        ranks['embed'] = embed_rank
    return ranks


def roofline_table(models=('Qwen3-0.6B', 'Qwen3-4B'), keeps=(1.0, 0.9, 0.8, 0.6, 0.5, 0.4),
                   embed_keep=(1.0, 0.5), ctx=(0, 2048, 8192), align=64,
                   int8=(False, True)):
    """Ceiling decode speedups vs dense bf16 (weights at 2 B, INT8 at 1 B).
    INT8 here means weight-only INT8 of the compressed model; the fair INT8
    comparison is against dense INT8, also listed (ratio = 'vs_dense_int8')."""
    rows = []
    for name in models:
        dense = meta_qwen3(name)
        n_dense = sum(p.numel() for p in dense.parameters())
        D = dense.config.hidden_size
        for ek in embed_keep:
            for keep in keeps:
                m = meta_qwen3(name)
                er = 0 if ek >= 1 else _align(ek * m.config.vocab_size * D /
                                               (m.config.vocab_size + D), align)
                if keep < 1 or er:
                    synthetic_lowrank_(m, keep, er, align, targets=TARGETS if keep < 1 else ())
                n = sum(p.numel() for p in m.parameters())
                for c in ctx:
                    bd = decode_bytes(dense, c)['total']
                    bd8 = decode_bytes(dense, c, wbytes=1.0)['total']
                    for q in int8:
                        bc = decode_bytes(m, c, wbytes=1.0 if q else 2.0)['total']
                        rows.append({'model': name, 'layer_keep': keep, 'embed_keep': ek,
                                     'embed_rank': er, 'int8': q, 'ctx': c,
                                     'param_saving_pct': round(100 * (1 - n / n_dense), 1),
                                     'ceiling_vs_dense_bf16': round(bd / bc, 3),
                                     'ceiling_vs_dense_int8': round(bd8 / bc, 3) if q else None})
    return rows


# ----------------------------------------------------------------------
# F6. rank alignment of an existing compressed model
# ----------------------------------------------------------------------
@torch.no_grad()
def align_ranks_(model, align: int = 64, mode: str = 'nearest') -> dict:
    """Re-truncate (or zero-pad) every LowRankLinear / LowRankEmbedding /
    untied LowRankHead bond to a multiple of `align`.

    Each bond is first put in balanced canonical form (regauge_), so A B =
    Qa U s Vh Qb^T with s sorted: dropping trailing bond indices is the
    optimal rank-r truncation of the current A B (Eckart-Young). Padding
    adds zero columns/rows, which is exact. mode: nearest | down | up.
    Returns {name: (old_rank, new_rank)}."""
    changes = {}
    for name, m in list(model.named_modules()):
        if isinstance(m, (LowRankLinear, LowRankEmbedding)) or \
                (isinstance(m, LowRankHead) and m.tied is None):
            r = m.rank
            if mode == 'up':
                nr = int(math.ceil(r / align)) * align
            elif mode == 'down':
                nr = max(align, (r // align) * align)
            else:
                nr = _align(r, align)
            nr = min(nr, min(m.A.shape[0], m.B.shape[1]))
            if nr == r:
                continue
            regauge_(m.A.data, m.B.data)
            if nr < r:
                A, B = m.A.data[:, :nr], m.B.data[:nr]
            else:
                A = F.pad(m.A.data, (0, nr - r))
                B = F.pad(m.B.data, (0, 0, 0, nr - r))
            m.A = torch.nn.Parameter(A.contiguous())
            m.B = torch.nn.Parameter(B.contiguous())
            changes[name] = (r, nr)
    return changes


def rank_audit(model, align: int = 64) -> dict:
    ranks = {n: m.rank for n, m in model.named_modules()
             if isinstance(m, (LowRankLinear, LowRankEmbedding))
             or (isinstance(m, LowRankHead) and m.tied is None)}
    bad = {n: r for n, r in ranks.items() if r % align}
    return {'n_bonds': len(ranks), 'misaligned': bad, 'align': align}


# ----------------------------------------------------------------------
# K1. sibling fusion: gate_proj + up_proj in one GEMM
# ----------------------------------------------------------------------
class FusedGateUp(torch.nn.Module):
    """(gate(x), up(x)) with one GEMM on x. Both dense: W = [Wg; Wu].
    Both low rank: t = x [Bg; Bu]^T, then the two A factors."""

    def __init__(self, gate, up):
        super().__init__()
        self.lowrank = isinstance(gate, LowRankLinear)
        if self.lowrank:
            self.rg = gate.rank
            self.B = torch.nn.Parameter(torch.cat([gate.B.data, up.B.data]).contiguous())
            self.Ag = torch.nn.Parameter(gate.A.data.contiguous())
            self.Au = torch.nn.Parameter(up.A.data.contiguous())
        else:
            self.og = gate.out_features
            self.W = torch.nn.Parameter(torch.cat([gate.weight.data, up.weight.data]).contiguous())
        bg, bu = gate.bias, up.bias
        self.bias = None if bg is None else torch.nn.Parameter(torch.cat([bg.data, bu.data]))

    def forward(self, x):
        if self.lowrank:
            t = F.linear(x, self.B)
            g, u = F.linear(t[..., :self.rg], self.Ag), F.linear(t[..., self.rg:], self.Au)
            if self.bias is not None:
                g, u = g + self.bias[:g.shape[-1]], u + self.bias[g.shape[-1]:]
            return g, u
        gu = F.linear(x, self.W, self.bias)
        return gu[..., :self.og], gu[..., self.og:]


def _fused_mlp_forward(self, x):
    g, u = self.gate_up(x)
    return self.down_proj(self.act_fn(g) * u)


@torch.no_grad()
def fuse_siblings_(model) -> int:
    """Fuse gate_proj/up_proj in every Qwen3/Llama-style MLP whose two
    projections are the same kind (both dense or both low rank). Returns the
    number of fused MLPs. Exact up to float summation order."""
    n = 0
    for m in model.modules():
        g, u = getattr(m, 'gate_proj', None), getattr(m, 'up_proj', None)
        if g is None or u is None or not hasattr(m, 'down_proj') or not hasattr(m, 'act_fn'):
            continue
        same = (isinstance(g, LowRankLinear) and isinstance(u, LowRankLinear)) or \
               (type(g) is torch.nn.Linear and type(u) is torch.nn.Linear)
        if not same:
            continue
        m.gate_up = FusedGateUp(g, u)
        del m.gate_proj, m.up_proj
        m.forward = types.MethodType(_fused_mlp_forward, m)
        n += 1
    return n


@torch.no_grad()
def as_linear_pairs_(model) -> int:
    """LowRankLinear -> Sequential(Linear(in, r), Linear(r, out)): same math,
    but tools that only know nn.Linear (torchao INT8, vLLM loaders) see it."""
    n = 0
    for name, m in list(model.named_modules()):
        if isinstance(m, LowRankLinear):
            l1 = torch.nn.Linear(m.in_features, m.rank, bias=False,
                                 device=m.B.device, dtype=m.B.dtype)
            l2 = torch.nn.Linear(m.rank, m.out_features, bias=m.bias is not None,
                                 device=m.A.device, dtype=m.A.dtype)
            l1.weight.copy_(m.B); l2.weight.copy_(m.A)
            if m.bias is not None:
                l2.bias.copy_(m.bias)
            _replace(model, m, torch.nn.Sequential(l1, l2))
            n += 1
    return n


# ----------------------------------------------------------------------
# L2. lossless speculative decoding (greedy)
# ----------------------------------------------------------------------
def spec_speedup(alpha: float, gamma: int, c: float) -> float:
    """Expected speedup of speculative decoding (Leviathan et al. 2023):
    alpha = per-token acceptance, gamma = drafted tokens, c = draft step
    cost / target step cost. Verifying gamma+1 tokens at batch 1 costs about
    one target step because decode is bandwidth bound."""
    if alpha >= 1:
        return (gamma + 1) / (gamma * c + 1)
    return (1 - alpha ** (gamma + 1)) / ((1 - alpha) * (gamma * c + 1))


def _crop(cache, keep):
    """Keep the first `keep` tokens. crop(-n) removes n tokens in both
    transformers 4.x and 5.x (5.x rejects a positive length)."""
    n = cache.get_seq_length() - keep
    if n > 0:
        cache.crop(-n)


@torch.no_grad()
def speculative_generate(target, draft, input_ids, max_new: int, gamma: int = 4,
                         eos_token_id=None):
    """Greedy speculative decoding at batch 1 with KV caches on both models.

    Invariant: each cache holds every committed token except the last one.
    Each round the draft proposes `gamma` tokens; the target scores
    [uncached tokens + proposals] in ONE forward; the longest prefix where
    proposal == target argmax is kept, plus the target's own next token.
    The output is therefore exactly the target's greedy continuation (up to
    the float differences between batched and one-token forwards)."""
    from transformers import DynamicCache
    assert input_ids.shape[0] == 1, "batch 1 only"
    dev = input_ids.device
    seq = input_ids
    tc, dc = DynamicCache(), DynamicCache()
    if seq.shape[1] > 1:
        target(seq[:, :-1], past_key_values=tc, use_cache=True)
        draft(seq[:, :-1], past_key_values=dc, use_cache=True)
    L0 = seq.shape[1]
    stats = {'rounds': 0, 'proposed': 0, 'accepted': 0, 'target_calls': 0, 'draft_calls': 0}
    while seq.shape[1] - L0 < max_new:
        g = min(gamma, max_new - (seq.shape[1] - L0))
        props = []
        inp = seq[:, dc.get_seq_length():]
        for _ in range(g):
            lg = draft(inp, past_key_values=dc, use_cache=True).logits[:, -1]
            nxt = lg.argmax(-1, keepdim=True)
            props.append(nxt)
            inp = nxt
            stats['draft_calls'] += 1
        props = torch.cat(props, 1)                                   # (1, g)
        old = seq.shape[1]
        t_inp = torch.cat([seq[:, tc.get_seq_length():], props], 1)
        lg = target(t_inp, past_key_values=tc, use_cache=True).logits[:, -(g + 1):]
        stats['target_calls'] += 1
        pred = lg.argmax(-1)                                          # (1, g+1)
        match = (props == pred[:, :g])[0].long()
        n = int(match.cumprod(0).sum())
        new = torch.cat([props[:, :n], pred[:, n:n + 1]], 1)
        if eos_token_id is not None:
            hit = (new[0] == eos_token_id).nonzero()
            if len(hit):
                new = new[:, :int(hit[0]) + 1]
        seq = torch.cat([seq, new], 1)
        stats['rounds'] += 1; stats['proposed'] += g; stats['accepted'] += n
        _crop(tc, seq.shape[1] - 1)
        _crop(dc, seq.shape[1] - 1)
        if eos_token_id is not None and int(seq[0, -1]) == eos_token_id:
            break
    seq = seq[:, :L0 + max_new]
    stats['alpha'] = stats['accepted'] / max(1, stats['proposed'])
    stats['tokens_per_target_call'] = (seq.shape[1] - L0) / max(1, stats['target_calls'])
    return seq, stats


@torch.no_grad()
def greedy_generate(model, input_ids, max_new: int, eos_token_id=None):
    """Plain KV-cached greedy loop: the reference L2 must reproduce, and the
    'eager' engine for decode timing."""
    from transformers import DynamicCache
    cache = DynamicCache()
    lg = model(input_ids, past_key_values=cache, use_cache=True).logits[:, -1]
    out = [lg.argmax(-1, keepdim=True)]
    for _ in range(max_new - 1):
        if eos_token_id is not None and int(out[-1]) == eos_token_id:
            break
        lg = model(out[-1], past_key_values=cache, use_cache=True).logits[:, -1]
        out.append(lg.argmax(-1, keepdim=True))
    return torch.cat([input_ids] + out, 1)


class StaticDecoder:
    """A model with a StaticCache and fixed-shape steps. With compile=True
    (CUDA) each shape is captured once as a CUDA graph (torch.compile
    'reduce-overhead'), as HF generate does for its static cache. Rolling
    back after a rejected draft needs no crop: stale cache slots sit at
    positions after the current query, so the causal mask hides them until
    they are overwritten."""

    def __init__(self, model, max_len: int, compile: bool = False):
        from transformers import StaticCache
        p = next(model.parameters())
        try:                                              # transformers >= 5
            self.cache = StaticCache(config=model.config, max_cache_len=max_len)
        except TypeError:                                 # 4.x
            self.cache = StaticCache(config=model.config, max_batch_size=1,
                                     max_cache_len=max_len, device=p.device, dtype=p.dtype)
        self.model, self.dev, self.max_len, self.len = model, p.device, max_len, 0
        self.fwd = torch.compile(model, mode='reduce-overhead', fullgraph=True) if compile else model

    def reset(self):
        self.cache.reset()
        self.len = 0

    def _seek(self, start):
        """transformers 5.x writes at an internal per-layer counter, not at
        cache_position: move it back after a rejection (outside the graph,
        and only when it differs). 4.x writes at cache_position already."""
        if start != self.len:
            for layer in getattr(self.cache, 'layers', []):
                if getattr(layer, 'is_initialized', False) and hasattr(layer, 'cumulative_length'):
                    layer.cumulative_length.fill_(start)

    def _call(self, fn, ids, start):
        self._seek(start)
        self.len = start + ids.shape[1]
        pos = torch.arange(start, start + ids.shape[1], device=self.dev)
        return fn(ids, past_key_values=self.cache, cache_position=pos,
                  position_ids=pos[None], use_cache=True).logits

    def prefill(self, ids):                       # variable length: not compiled
        return self._call(self.model, ids, 0)

    def step(self, ids, start):                   # fixed shape: CUDA graph
        return self._call(self.fwd, ids, start)


@torch.no_grad()
def greedy_static(dec: StaticDecoder, input_ids, max_new: int):
    dec.reset()
    L0 = input_ids.shape[1]
    nxt = dec.prefill(input_ids)[:, -1].argmax(-1, keepdim=True)
    out = [nxt]
    for i in range(max_new - 1):
        nxt = dec.step(nxt, L0 + i)[:, -1].argmax(-1, keepdim=True)
        out.append(nxt)
    return torch.cat([input_ids] + out, 1)


@torch.no_grad()
def speculative_static(T: StaticDecoder, Dd: StaticDecoder, input_ids, max_new: int,
                       gamma: int = 4):
    """speculative_generate with static caches and only two step shapes:
    draft (1, 1) and target verify (1, gamma + 1), so both replay CUDA
    graphs. Always drafts `gamma` tokens (the last round may overshoot and
    is trimmed) to keep the verify shape fixed."""
    T.reset(); Dd.reset()
    seq, L0 = input_ids, input_ids.shape[1]
    if L0 > 1:
        T.prefill(seq[:, :-1]); Dd.prefill(seq[:, :-1])
    d_len = L0 - 1
    stats = {'rounds': 0, 'proposed': 0, 'accepted': 0, 'target_calls': 0, 'draft_calls': 0}
    while seq.shape[1] - L0 < max_new:
        cur = seq.shape[1]
        for j in range(d_len, cur):                       # catch up (1 or 2 tokens)
            lg = Dd.step(seq[:, j:j + 1], j)
            stats['draft_calls'] += 1
        props = [lg[:, -1].argmax(-1, keepdim=True)]
        for i in range(1, gamma):
            props.append(Dd.step(props[-1], cur + i - 1)[:, -1].argmax(-1, keepdim=True))
            stats['draft_calls'] += 1
        props = torch.cat(props, 1)
        pred = T.step(torch.cat([seq[:, cur - 1:cur], props], 1), cur - 1).argmax(-1)
        stats['target_calls'] += 1
        n = int((props == pred[:, :gamma])[0].long().cumprod(0).sum())
        seq = torch.cat([seq, props[:, :n], pred[:, n:n + 1]], 1)
        d_len = min(cur + gamma - 1, cur + n)
        stats['rounds'] += 1; stats['proposed'] += gamma; stats['accepted'] += n
    seq = seq[:, :L0 + max_new]
    stats['alpha'] = stats['accepted'] / max(1, stats['proposed'])
    stats['tokens_per_target_call'] = max_new / max(1, stats['target_calls'])
    return seq, stats


def spec_engine(target, draft, engine: str, max_len: int, gammas=(4,)):
    """(greedy(ids, n), spec(ids, n, gamma), draft_greedy(ids, n)) on one
    engine: eager = DynamicCache loops; static = StaticCache, no compile
    (CPU check of the graph path); compiled = static + CUDA graphs."""
    if engine == 'eager':
        return (lambda ids, n: greedy_generate(target, ids, n),
                lambda ids, n, g: speculative_generate(target, draft, ids, n, g),
                lambda ids, n: greedy_generate(draft, ids, n))
    comp = engine == 'compiled'
    T1 = StaticDecoder(target, max_len, comp)                  # (1, 1) steps
    T2 = T1 if not comp else StaticDecoder(target, max_len + max(gammas) + 2, comp)
    Dd = StaticDecoder(draft, max_len + max(gammas) + 2, comp)
    return (lambda ids, n: greedy_static(T1, ids, n),
            lambda ids, n, g: speculative_static(T2, Dd, ids, n, g),
            lambda ids, n: greedy_static(Dd, ids, n))


# ----------------------------------------------------------------------
# timing with paired bootstrap CIs
# ----------------------------------------------------------------------
def _sync(dev):
    if dev.type == 'cuda':
        torch.cuda.synchronize()


def ratio_ci(t_base, t_new, n=10_000, seed=0):
    """Speedup = mean(base time) / mean(new time) over the same prompts, with
    a 95 % paired-bootstrap percentile CI (prompts resampled jointly)."""
    a, b = np.asarray(t_base, float), np.asarray(t_new, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    bs = a[idx].mean(1) / b[idx].mean(1)
    return float(a.mean() / b.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def make_engine(model, mode: str, max_len: int):
    """Returns gen(ids, n) -> ids. 'eager': DynamicCache Python loop.
    'compiled': HF static cache + torch.compile(mode='reduce-overhead'),
    i.e. CUDA graphs, the standard way to remove launch overhead at batch 1."""
    if mode == 'eager':
        return lambda ids, n: greedy_generate(model, ids, n)
    if mode == 'compiled':
        # generate() auto-compiles the forward (reduce-overhead, fullgraph)
        # when the cache is static and the device is CUDA
        model.generation_config.cache_implementation = 'static'
        pad = getattr(model.generation_config, 'pad_token_id', None)
        if pad is None:
            model.generation_config.pad_token_id = 0

        def gen(ids, n):
            return model.generate(ids, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                                  attention_mask=torch.ones_like(ids))
        return gen
    raise ValueError(mode)


def time_decode(gen, prompts, new_tokens, dev, repeats=1, warm=2):
    """Per prompt: decode seconds per token = (t(N) - t(1)) / (N - 1), which
    removes prefill. Returns a list of per-prompt seconds/token."""
    for ids in prompts[:warm]:
        gen(ids, new_tokens); gen(ids, 1)
    per = []
    for ids in prompts:
        best = float('inf')
        for _ in range(repeats):
            _sync(dev); t0 = time.perf_counter(); gen(ids, 1); _sync(dev)
            t1 = time.perf_counter(); gen(ids, new_tokens); _sync(dev)
            t2 = time.perf_counter()
            best = min(best, ((t2 - t1) - (t1 - t0)) / (new_tokens - 1))
        per.append(max(best, 1e-9))
    return per


def time_full(fn, prompts, new_tokens, dev, warm=2):
    """Per prompt seconds/token INCLUDING prefill (same for both sides of
    an L2 comparison). fn(ids, n) may return (ids, stats)."""
    for ids in prompts[:warm]:
        fn(ids, new_tokens)
    per, stats = [], []
    for ids in prompts:
        _sync(dev); t0 = time.perf_counter()
        r = fn(ids, new_tokens)
        _sync(dev)
        per.append((time.perf_counter() - t0) / new_tokens)
        if isinstance(r, tuple):
            stats.append(r[1])
    return per, stats


def lossless_check(greedy_fn, spec_fn, prompts, new_tokens):
    """Fraction of generated tokens (identical prefix) that match the
    target's own greedy decode on the same engine. 1.0 in fp32; bf16 can
    differ after a near-tie because the batched verify forward rounds
    differently from one-token forwards."""
    same = tot = 0
    for ids in prompts:
        ref = greedy_fn(ids, new_tokens)[0, ids.shape[1]:]
        out = spec_fn(ids, new_tokens)[0][0, ids.shape[1]:]
        k = min(len(ref), len(out))
        first = (ref[:k] != out[:k]).nonzero()
        same += k if len(first) == 0 else int(first[0])
        tot += k
    return same / max(1, tot)


@torch.no_grad()
def teacher_forced_gaps(model, seq, prompt_len: int) -> torch.Tensor:
    """For every generated token: (max logit - logit of that token) under ONE
    batched forward of `model` over the whole sequence, in fp32. 0 means the
    token is the model's argmax; a small positive gap is a near-tie that bf16
    rounding can flip either way."""
    lg = model(seq[:, :-1]).logits[0, prompt_len - 1:].float()
    tok = seq[0, prompt_len:]
    return lg.max(-1).values - lg.gather(-1, tok[:, None])[:, 0]


def lossless_report(target, draft, greedy_fn, spec_fn, prompts, new_tokens, engine,
                    tol: float = 0.25) -> dict:
    """Three views of "speculative output == target greedy output":

    prefix_bf16  identical-prefix fraction vs greedy on the same engine. In
                 bf16 one near-tie flip (the batched verify forward rounds
                 differently from one-token forwards) changes everything
                 after it, so this is low even with correct code: QL6 stage S
                 measured 0.40-0.76 on H100, and a 4-layer random Qwen3 gives
                 0.3-0.5 in bf16 vs 1.0 in fp32 on CPU.
    fp32         the same comparison with both models upcast to fp32 (exact,
                 then cast back): tests the LOGIC; must be 1.0.
    tf_within    fraction of emitted tokens whose teacher-forced gap to the
                 target's argmax is <= tol logits, for spec AND for plain
                 greedy (the noise floor of greedy itself)."""
    ps = prompts
    out = {'prefix_bf16': lossless_check(greedy_fn, spec_fn, ps, new_tokens)}
    gs, gg = [], []
    for ids in ps:
        L = ids.shape[1]
        gs.append(teacher_forced_gaps(target, spec_fn(ids, new_tokens)[0], L))
        gg.append(teacher_forced_gaps(target, greedy_fn(ids, new_tokens), L))
    gs, gg = torch.cat(gs), torch.cat(gg)
    out.update(tf_within_spec=float((gs <= tol).float().mean()),
               tf_within_greedy=float((gg <= tol).float().mean()),
               tf_max_gap_spec=round(float(gs.max()), 4),
               tf_max_gap_greedy=round(float(gg.max()), 4), tol=tol)
    dt = next(target.parameters()).dtype
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        # "fp32" must mean fp32: main() enables TF32 (10-bit mantissa), which
        # brings the near-tie flips back (stage L: 0.89 with TF32 on)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        if dt != torch.float32:
            target.float(); draft.float()
        g32, s32, _ = spec_engine(target, draft, 'eager' if engine == 'eager' else 'static',
                                  max(p.shape[1] for p in ps) + new_tokens + 16, (8,))
        gm = getattr(spec_fn, 'gamma', 4)
        same = tot = 0
        div = []
        for ids in ps:
            L = ids.shape[1]
            ref = g32(ids, new_tokens)
            got = s32(ids, new_tokens, gm)[0]
            k = min(ref.shape[1], got.shape[1]) - L
            bad = (ref[0, L:L + k] != got[0, L:L + k]).nonzero()
            j = k if len(bad) == 0 else int(bad[0])
            same += j; tot += k
            if j < k:     # fp32 logit gap between the two candidate tokens at the split
                lg = target(ref[:, :L + j]).logits[0, -1].float()
                div.append(round(abs(float(lg[ref[0, L + j]] - lg[got[0, L + j]])), 6))
        out['fp32'] = same / max(1, tot)
        out['fp32_divergence_gaps'] = div
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
        target.to(dt); draft.to(dt)
    return out


# ----------------------------------------------------------------------
# prefill: dense SDPA vs QSA top-16, end to end
# ----------------------------------------------------------------------
@torch.no_grad()
def time_prefill(model, n_tokens, dev, vocab, iters=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(10, vocab, (1, n_tokens), generator=g).to(dev)
    model(ids[:, :min(n_tokens, 2048)], use_cache=False, logits_to_keep=1)
    ts = []
    for _ in range(iters):
        _sync(dev); t0 = time.perf_counter()
        model(ids, use_cache=False, logits_to_keep=1)
        _sync(dev); ts.append(time.perf_counter() - t0)
    return ts


def set_attn(model, impl):
    if impl == 'qsa':
        from phase4_llm.qsa_attention import register_hf
        register_hf()
    model.set_attn_implementation(impl) if hasattr(model, 'set_attn_implementation') \
        else setattr(model.config, '_attn_implementation', impl)


# ----------------------------------------------------------------------
# gate
# ----------------------------------------------------------------------
def gate(report: dict) -> dict:
    """Decode: best variant's speedup vs dense on the SAME engine; PASS on
    the point estimate, PASS_at_CI_bound on the lower 95 % bound. A variant
    with fusion / INT8 is judged by the SMALLER of its speedup vs plain dense
    and vs dense with the same fusion / INT8 (so INT8 cannot pass on INT8's
    merit alone). Lossy
    variants (route L1) only count with a QL5 quality PASS. Lossless ones
    (L2) need exact agreement with dense greedy in fp32 (the logic) and, in
    bf16, >= 99 % of emitted tokens within `tol` logits of the target's
    argmax under teacher forcing, or no worse than plain greedy's own rate
    (bf16 near-ties make exact bf16 agreement meaningless, see
    lossless_report). Prefill: 32K."""
    out = {'decode_gate': DECODE_GATE, 'prefill_gate': PREFILL_GATE, 'decode': {}}
    best = None
    for row in report.get('decode', []):
        if row.get('variant') == 'dense' or 'speedup' not in row:
            continue
        lossless = row.get('route') == 'L2'
        if lossless and 'lossless' in row:
            ll = row['lossless']                     # see lossless_report
            fp32_ok = ll['fp32'] >= 0.99 or all(g < 1e-3 for g in ll.get('fp32_divergence_gaps', [1]))
            quality_ok = fp32_ok and \
                ll['tf_within_spec'] >= min(0.99, ll['tf_within_greedy'] - 0.005)
        elif lossless:
            quality_ok = row.get('agreement', 0) >= 0.99
        else:
            quality_ok = bool(row.get('quality_pass'))
        s, ci = row['speedup'], row['ci95']
        if 'speedup_vs_matched' in row and row['speedup_vs_matched'] < s:
            s, ci = row['speedup_vs_matched'], row['ci95_vs_matched']
        r = {'engine': row['engine'], 'route': row.get('route'), 'speedup': s,
             'ci95': ci, 'quality_ok': quality_ok,
             'pass_point': s >= DECODE_GATE and quality_ok,
             'pass_ci': ci[0] >= DECODE_GATE and quality_ok,
             'speed_only': bool(row.get('synthetic'))}
        out['decode'][f"{row['variant']}@{row['engine']}"] = r
        rank = (r['pass_ci'], r['pass_point'], r['quality_ok'], r['speedup'])
        if not r['speed_only'] and (best is None or rank > best[0]):
            best = (rank, r)
    best = best[1] if best else None
    out['decode_PASS'] = bool(best and best['pass_point'])
    out['decode_PASS_at_CI_bound'] = bool(best and best['pass_ci'])
    pf = [p for p in report.get('prefill', []) if p.get('tokens') == 32768 and 'speedup' in p]
    out['prefill_32k'] = pf[0] if pf else None
    out['prefill_PASS'] = bool(pf and pf[0]['speedup'] >= PREFILL_GATE)
    return out


# ----------------------------------------------------------------------
def _load(spec, dev, dtype):
    """HF id / local HF dir, or a QL5 model.pt (save_compressed)."""
    if spec.endswith('.pt'):
        from phase4_llm.tn_layers import load_compressed
        return load_compressed(spec, device=dev, dtype=dtype)
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(spec, dtype=dtype,
                                                attn_implementation='sdpa').to(dev).eval()


def _int8(model):
    """Real weight-only INT8 kernels via torchao (fake_quant_ gives no speed)."""
    from torchao.quantization import quantize_
    try:
        from torchao.quantization import Int8WeightOnlyConfig
        quantize_(model, Int8WeightOnlyConfig())
    except ImportError:
        from torchao.quantization import int8_weight_only
        quantize_(model, int8_weight_only())
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
    ap.add_argument('--out', default='/workspace/ql6')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--roofline', action='store_true', help='print ceilings and exit (CPU)')
    ap.add_argument('--compressed', default='',
                    help='comma list of QL5 model.pt files (route L1, real quality)')
    ap.add_argument('--quality-pass', default='',
                    help='comma list of 0/1 per --compressed: QL5 gate PASS (from gate.json)')
    ap.add_argument('--synthetic-keep', default='0.8,0.6',
                    help='layer keep fractions for speed-only random low-rank copies')
    ap.add_argument('--embed-keep', type=float, default=0.5,
                    help='synthetic copies: kept fraction of embedding/head params')
    ap.add_argument('--rank-align', type=int, default=64)
    ap.add_argument('--drafts', default='',
                    help='comma list of draft models for route L2 (HF id or model.pt)')
    ap.add_argument('--gamma', default='3,5')
    ap.add_argument('--engines', default='eager,compiled')
    ap.add_argument('--new-tokens', type=int, default=256)
    ap.add_argument('--n-prompts', type=int, default=24)
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--no-fuse', action='store_true')
    ap.add_argument('--int8', action='store_true', help='also time torchao INT8 variants')
    ap.add_argument('--triton', action='store_true',
                    help='also time low-rank variants on the ql6_kernels Triton path '
                         '(and use it for .pt drafts)')
    ap.add_argument('--layer-bench', action='store_true',
                    help='per-layer decode latency: dense vs torch vs Triton low rank (CUDA)')
    ap.add_argument('--prefill', default='8192,16384,32768')
    ap.add_argument('--qsa-topk', type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.roofline:
        rows = roofline_table()
        json.dump(rows, open(os.path.join(args.out, 'ql6_roofline.json'), 'w'), indent=2)
        for r in rows:
            if r['ctx'] == 2048:
                print(r)
        return

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if dev.type == 'cuda' else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.smoke:
        from phase4_llm.ql5_compress import tiny_qwen3
        V = 512
        load_base = lambda: tiny_qwen3(V, tie=False).to(dev, dtype).eval()   # tied random = copy input
        tok = None
        args.new_tokens, args.n_prompts, args.repeats = min(args.new_tokens, 16), min(args.n_prompts, 4), 1
        args.rank_align, args.engines = 8, 'eager'
        args.prefill = '512' if dev.type == 'cuda' else '256'   # flex: >= 4 blocks of 128
        prompts = [torch.randint(1, V, (1, 12 + 3 * i), generator=torch.Generator().manual_seed(i)).to(dev)
                   for i in range(args.n_prompts)]
        print("[ql6] smoke: tiny random Qwen3 (no downloads)")
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        load_base = lambda: _load(args.model, dev, dtype)
        prompts = [tok(p, return_tensors='pt').input_ids.to(dev) for p in PROMPTS[:args.n_prompts]]
    base = load_base()
    V = base.config.vocab_size
    max_len = max(p.shape[1] for p in prompts) + args.new_tokens + 8
    report = {'model': args.model, 'device': str(dev), 'dtype': str(dtype),
              'gpu': torch.cuda.get_device_name() if dev.type == 'cuda' else None,
              'torch': torch.__version__, 'new_tokens': args.new_tokens,
              'n_prompts': len(prompts), 'decode': [], 'spec': [], 'prefill': [],
              'roofline': {}}
    del base

    # ---- variants for decode timing ----
    variants = [('dense', load_base, {'route': None})]
    for keep in [float(k) for k in args.synthetic_keep.split(',') if k]:
        def mk(keep=keep):
            m = load_base()
            D = m.config.hidden_size
            er = 0 if args.embed_keep >= 1 else _align(args.embed_keep * V * D / (V + D), args.rank_align)
            synthetic_lowrank_(m, keep, er, args.rank_align)
            return m.to(dev, dtype).eval()
        variants.append((f'synthetic_keep{keep}_emb{args.embed_keep}', mk,
                         {'route': 'L1', 'synthetic': True}))
    qp = [q == '1' for q in args.quality_pass.split(',')] if args.quality_pass else []
    for i, path in enumerate([p for p in args.compressed.split(',') if p]):
        variants.append((f'ql5:{os.path.basename(os.path.dirname(path)) or path}',
                         lambda path=path: _load(path, dev, dtype),
                         {'route': 'L1', 'quality_pass': qp[i] if i < len(qp) else None}))

    dense_bytes = None
    for vname, mk, meta in variants:
        m = mk()
        rb = decode_bytes(m, ctx=max_len // 2, wbytes=2.0 if dtype == torch.bfloat16 else 4.0)
        dense_bytes = dense_bytes or rb['total']
        report['roofline'][vname] = {'bytes_per_token': rb, 'ceiling': round(dense_bytes / rb['total'], 3),
                                     'params': sum(p.numel() for p in m.parameters()),
                                     'audit': rank_audit(m, args.rank_align)}
        del m
    from phase4_llm.ql6_kernels import available as _tri_available, use_triton_, bench_layers
    tri_ok = _tri_available(torch.zeros(1, device=dev))
    if args.layer_bench and dev.type == 'cuda':
        c = load_base().config
        hd = getattr(c, 'head_dim', None) or c.hidden_size // c.num_attention_heads
        H, I = c.hidden_size, c.intermediate_size
        shapes = [(H, c.num_attention_heads * hd, 'q_proj'), (H, c.num_key_value_heads * hd, 'k_proj'),
                  (c.num_attention_heads * hd, H, 'o_proj'), (H, I, 'up_proj'), (I, H, 'down_proj')]
        report['layers'] = bench_layers(shapes, keeps=(0.9, 0.6), M=1) + \
            bench_layers(shapes[3:4], keeps=(0.9,), M=6)            # speculative verify size
    for engine in args.engines.split(','):
        times = {}
        for vname, mk, meta in variants:
            for fuse in ([False] if args.no_fuse else [False, True]):
                extras = ['']
                if args.int8 and dev.type == 'cuda' and engine == 'compiled':
                    extras.append('int8')
                if args.triton and fuse and vname != 'dense' and tri_ok:
                    extras.append('triton')
                for ex in extras:
                    q8, tri = ex == 'int8', ex == 'triton'
                    key = vname + ('+fuse' if fuse else '') + ('+' + ex if ex else '')
                    try:
                        m = mk()
                        if fuse:
                            fuse_siblings_(m)
                        if q8:
                            as_linear_pairs_(m); _int8(m)
                        if tri:
                            meta = {**meta, 'triton': use_triton_(m)}
                        gen = make_engine(m, engine, max_len)
                        times[key] = time_decode(gen, prompts, args.new_tokens, dev, args.repeats)
                        row = {'variant': key, 'engine': engine, **meta,
                               'tok_s': round(1 / float(np.mean(times[key])), 1)}
                        base_key = 'dense' + ('+fuse' if fuse else '') + ('+int8' if q8 else '')
                        if key != 'dense':
                            s, lo, hi = ratio_ci(times['dense'], times[key])
                            row.update(speedup=round(s, 3), ci95=[round(lo, 3), round(hi, 3)])
                            if base_key in times and base_key != key:
                                s2, lo2, hi2 = ratio_ci(times[base_key], times[key])
                                row.update(vs_matched_dense=base_key, speedup_vs_matched=round(s2, 3),
                                           ci95_vs_matched=[round(lo2, 3), round(hi2, 3)])
                        report['decode'].append(row)
                    except Exception as e:                           # torchao/compile failures etc.
                        report['decode'].append({'variant': key, 'engine': engine,
                                                 'error': f"{type(e).__name__}: {str(e)[:300]}"})
                    print("[ql6] decode", report['decode'][-1], flush=True)
                    m = None
                    if dev.type == 'cuda':
                        torch._dynamo.reset(); torch.cuda.empty_cache()

    # ---- route L2: lossless speculative decoding, greedy and spec on the SAME engine ----
    drafts = [d for d in args.drafts.split(',') if d]
    gammas = [int(x) for x in args.gamma.split(',')]
    l2_engines = ['eager', 'static'] if args.smoke else \
        [e for e in args.engines.split(',') if dev.type == 'cuda' or e != 'compiled']
    if args.smoke:
        drafts = ['self', 'noisy']
    if drafts:
        target = load_base()
        for d in drafts:
            if d in ('self', 'noisy'):
                dm = load_base()
                if d == 'noisy':                     # smoke: imperfect draft
                    with torch.no_grad():
                        for p in dm.parameters():
                            p.add_(0.05 * p.std() * torch.randn_like(p))
            else:
                dm = _load(d, dev, dtype)
                if args.triton and d.endswith('.pt') and tri_ok:
                    fuse_siblings_(dm); use_triton_(dm)
            for engine in l2_engines:
                for gm in gammas:
                    # fresh engine per gamma: greedy reference, draft cost and spec are
                    # timed in the same compiled session; lossless_report runs last
                    # because its fp32 cast invalidates the CUDA graphs
                    try:
                        greedy_fn, spec_fn, dgreedy = spec_engine(
                            target, dm, engine, max_len + gm + 2, (gm,))
                        sp = lambda ids, n, gm=gm: spec_fn(ids, n, gm)
                        sp.gamma = gm
                        ref, _ = time_full(greedy_fn, prompts, args.new_tokens, dev)
                        dref, _ = time_full(dgreedy, prompts, args.new_tokens, dev)
                        per, st = time_full(sp, prompts, args.new_tokens, dev)
                        c = float(np.mean(dref)) / float(np.mean(ref))
                        ll = lossless_report(target, dm, greedy_fn, sp,
                                             prompts[:min(8, len(prompts))],
                                             min(args.new_tokens, 128), engine)
                        alpha = float(np.mean([x['alpha'] for x in st]))
                        tpc = float(np.mean([x['tokens_per_target_call'] for x in st]))
                        s_, lo, hi = ratio_ci(ref, per)
                        row = {'variant': f'spec:{d}:g{gm}', 'engine': engine, 'route': 'L2',
                               'draft': d, 'gamma': gm, 'alpha': round(alpha, 3),
                               'tokens_per_target_call': round(tpc, 2),
                               'draft_cost_c': round(c, 3),
                               'dense_tok_s': round(1 / float(np.mean(ref)), 1),
                               'draft_tok_s': round(1 / float(np.mean(dref)), 1),
                               'predicted': round(spec_speedup(alpha, gm, c), 3),
                               'agreement': round(ll['prefix_bf16'], 4), 'lossless': ll,
                               'speedup': round(s_, 3),
                               'ci95': [round(lo, 3), round(hi, 3)]}
                    except Exception as e:
                        row = {'variant': f'spec:{d}:g{gm}', 'engine': engine, 'route': 'L2',
                               'error': f"{type(e).__name__}: {str(e)[:300]}"}
                    report['spec'].append(row)
                    if 'error' not in row:
                        report['decode'].append(row)
                    print("[ql6] spec", row, flush=True)
                    if dev.type == 'cuda':
                        torch._dynamo.reset(); torch.cuda.empty_cache()
            del dm
        # HF built-in assisted generation (eager): named baseline for our loop
        if not args.smoke:
            try:
                dm = _load(drafts[0], dev, dtype)
                ref, _ = time_full(lambda ids, n: greedy_generate(target, ids, n),
                                   prompts, args.new_tokens, dev)
                per, _ = time_full(lambda ids, n: target.generate(
                    ids, assistant_model=dm, max_new_tokens=n, min_new_tokens=n,
                    do_sample=False, attention_mask=torch.ones_like(ids)),
                    prompts, args.new_tokens, dev)
                s_, lo, hi = ratio_ci(ref, per)
                report['spec'].append({'variant': f'hf_assisted:{drafts[0]}', 'engine': 'eager',
                                       'speedup': round(s_, 3), 'ci95': [round(lo, 3), round(hi, 3)]})
            except Exception as e:
                report['spec'].append({'variant': 'hf_assisted', 'error': str(e)[:300]})
            print("[ql6] spec", report['spec'][-1], flush=True)
        del target

    # ---- prefill: SDPA vs QSA top-16, whole model ----
    if args.prefill:
        from phase4_llm.qsa_attention import QSA_CFG
        QSA_CFG['topk'] = args.qsa_topk
        if args.smoke:      # flex_attention (CUDA) needs block sizes of 128
            QSA_CFG.update(block=128 if dev.type == 'cuda' else 32, min_len=64)
        m = load_base()
        for n in [int(x) for x in args.prefill.split(',')]:
            res = {'tokens': n}
            try:
                set_attn(m, 'sdpa')
                res['dense_s'] = time_prefill(m, n, dev, V)
                set_attn(m, 'qsa')
                res['qsa_s'] = time_prefill(m, n, dev, V)
                s = float(np.median(res['dense_s']) / np.median(res['qsa_s']))
                res['speedup'] = round(s, 3)
            except Exception as e:
                res['error'] = f"{type(e).__name__}: {str(e)[:300]}"
            report['prefill'].append(res)
            print("[ql6] prefill", res, flush=True)
        del m

    report['gate'] = gate(report)
    json.dump(report, open(os.path.join(args.out, 'ql6_report.json'), 'w'), indent=2)
    print(json.dumps(report['gate'], indent=2))
    print("QL6 DECODE:", "PASS" if report['gate']['decode_PASS'] else "FAIL",
          "| at 95% CI bound:", "PASS" if report['gate']['decode_PASS_at_CI_bound'] else "FAIL",
          "| PREFILL 32K:", "PASS" if report['gate']['prefill_PASS'] else "FAIL/not run")


if __name__ == '__main__':
    main()
