"""
QL5 -- compress Qwen3 while keeping knowledge and reasoning
===========================================================
Gate (docs/FERMIQLLM_SURPASS_PLAN.md, QL5): -20 % of TOTAL params with
MMLU -<=2 pt and GSM8K -<=5 pt vs the uncompressed *Base* model, on
Qwen3-0.6B-Base, then Qwen3-4B-Base. QL1.5 (heal_finetune.py) lost
13 / 34 pt at only -11 %. Every change below targets one measured failure:

  F3  PPL reference was the chat model     -> default Qwen3-*-Base, and a
                                              same-recipe dense control run
  F4  knowledge/math collapse first        -> calibration + heal on a
      (calibrated on web text only)           knowledge/reasoning mixture
                                              (ql5_data.py, 13-gram decontam)
      every layer treated as equally       -> rank allocation minimises
      important                               activation error x layer Fisher
                                              importance (first-order Taylor)
      embeddings untouched, so the 11 %    -> D1: frequency x hidden-state
      all came from the transformer           weighted low rank of the tied
                                              embedding / LM head (26 % of 0.6B)
      6 M heal tokens                      -> >=100 M tokens, multi-GPU,
                                              chunked KD (no full-vocab logits
                                              tensor), per-source loss weights
  G1  free gauge on every bond             -> optional periodic re-gauging to
                                              the balanced canonical form
      quantisation never combined          -> --quant int8|fp8 after heal

The QL4 harness evaluates the saved model:
  python phase4_llm/eval/run_eval.py --dense Qwen/Qwen3-0.6B-Base \\
      --compressed OUT/model.pt --limit 0 --log-samples --out OUT/eval
  python phase4_llm/ql5_gate.py --dense OUT/eval/ql4_dense.json \\
      --compressed OUT/eval/ql4_compressed.json

CPU smoke (tiny random Qwen3, no downloads):
  python phase4_llm/ql5_compress.py --smoke
Multi-GPU:
  torchrun --nproc_per_node 4 phase4_llm/ql5_compress.py --model ... --out ...
"""

import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from phase4_llm import heal_finetune as hf
from phase4_llm.tn_layers import (LowRankLinear, LowRankEmbedding, LowRankHead,
                                  whitened_svd, lowrank_from_svd,
                                  weighted_embedding_svd, regauge_,
                                  quantize_model_, save_compressed)
from phase4_llm.anneal_alloc import allocate
from phase4_llm.ql5_data import MixturePacker, Decontam, parse_mix, DEFAULT_MIX

_align = hf._align


# ----------------------------------------------------------------------
# distributed helpers (plain all-reduce of grads; no DDP wrapper because
# the loss calls the backbone and the head separately)
# ----------------------------------------------------------------------
def dist_setup():
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local = int(os.environ.get('LOCAL_RANK', 0))
    if world > 1:
        dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        return rank, world, torch.device('cuda', local)
    return rank, world, torch.device('cpu')


def allreduce_grads(params, world, bucket=2 ** 26):
    if world == 1:
        return
    grads = [p.grad for p in params if p.grad is not None]
    i = 0
    while i < len(grads):
        j, n = i, 0
        while j < len(grads) and (n == 0 or n + grads[j].numel() <= bucket):
            n += grads[j].numel(); j += 1
        flat = torch.cat([g.reshape(-1) for g in grads[i:j]])
        dist.all_reduce(flat)
        flat /= world
        o = 0
        for g in grads[i:j]:
            g.copy_(flat[o:o + g.numel()].view_as(g)); o += g.numel()
        i = j


def broadcast_params(params, world):
    if world > 1:
        for p in params:
            dist.broadcast(p.data, 0)


# ----------------------------------------------------------------------
# loss: CE + KD on chunks of positions, each chunk checkpointed, so the
# (tokens x 151,936) logits never exist for the whole batch at once
# ----------------------------------------------------------------------
def _chunk_loss(hs, ht, y, w, head_s, head_t, alpha, T):
    ls = head_s(hs).float()
    ce = F.cross_entropy(ls, y, reduction='none')
    if ht is None:
        return (w * ce).sum()
    with torch.no_grad():
        lpt = F.log_softmax(head_t(ht).float() / T, -1)
    lps = F.log_softmax(ls / T, -1)
    kl = (lpt.exp() * (lpt - lps)).sum(-1) * T * T
    return (w * ((1 - alpha) * ce + alpha * kl)).sum()


def chunked_kd_loss(h_s, h_t, labels, tok_w, head_s, head_t,
                    alpha=0.5, T=1.0, chunk=4096):
    """h_s/h_t: (B, L, d) final hidden states (after the last norm) of
    student / teacher (h_t None -> CE only); labels (B, L) token ids;
    tok_w (B,) per-window loss weight. Returns the weighted mean over the
    B * (L - 1) next-token predictions."""
    B, L, d = h_s.shape
    hs = h_s[:, :-1].reshape(-1, d)
    ht = h_t[:, :-1].reshape(-1, d) if h_t is not None else None
    y = labels[:, 1:].reshape(-1)
    w = tok_w[:, None].expand(B, L - 1).reshape(-1).to(hs.dtype)
    tot = hs.new_zeros((), dtype=torch.float32)
    for i in range(0, y.numel(), chunk):
        sl = slice(i, i + chunk)
        args = (hs[sl], None if ht is None else ht[sl], y[sl], w[sl],
                head_s, head_t, alpha, T)
        if hs.requires_grad:
            tot = tot + checkpoint(_chunk_loss, *args, use_reentrant=False)
        else:
            tot = tot + _chunk_loss(*args)
    return tot / y.numel()


# ----------------------------------------------------------------------
def tiny_qwen3(vocab=512, tie=True, seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=512, tie_word_embeddings=tie)
    return Qwen3ForCausalLM(cfg)


class _SmokeTok:
    """Stand-in tokenizer for --smoke: random ids, deterministic per text."""

    def __init__(self, vocab):
        self.vocab, self.eos_token_id = vocab, 0

    def __call__(self, text, return_tensors=None):
        g = torch.Generator().manual_seed(len(text))
        ids = torch.randint(1, self.vocab, (1, 2048), generator=g)
        return type('T', (), {'input_ids': ids if return_tensors else ids[0].tolist()})()


def count_params(m):
    return sum(p.numel() for p in m.parameters())      # tied params counted once


def _replace(root, target, repl):
    for name, child in root.named_children():
        if child is target:
            setattr(root, name, repl)
            return True
        if _replace(child, target, repl):
            return True
    return False


# ----------------------------------------------------------------------
def calibrate(model, cands, head, rows, device, fisher=True, batch=4):
    """One pass over `rows` (list of LongTensor[seq+1]):
       * input Gram X^T X per candidate Linear and for the LM head input
       * Fisher importance per candidate: sum over batches of
         ||dL/dW (.) W||^2 (first-order Taylor estimate of the loss change
         when the layer is perturbed in proportion to its own weights)
       * token frequencies (for the embedding weighting)"""
    mods = [m for _, m in cands] + [head]
    grams = [None] * len(mods)

    def hook(i):
        def f(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).double()
            g = x.T @ x
            grams[i] = g if grams[i] is None else grams[i] + g
        return f

    hs = [m.register_forward_hook(hook(i)) for i, m in enumerate(mods)]
    imp = torch.zeros(len(cands), dtype=torch.float64)
    req = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    ws = [m.weight for _, m in cands]
    if fisher:
        for w in ws:
            w.requires_grad_(True)
    model.eval()
    freq = None
    for j in range(0, len(rows), batch):
        x = torch.stack(rows[j:j + batch]).to(device)
        c = torch.bincount(x.reshape(-1).cpu(), minlength=model.config.vocab_size)
        freq = c if freq is None else freq + c
        with torch.set_grad_enabled(fisher), torch.autocast(
                device_type='cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            out = model(x, labels=x)
        if fisher:
            grads = torch.autograd.grad(out.loss, ws)
            for l, (g, w) in enumerate(zip(grads, ws)):
                imp[l] += float((g.double() * w.detach().double()).pow(2).sum())
    for h in hs:
        h.remove()
    for n, p in model.named_parameters():
        p.requires_grad_(req[n])
    model.train()
    return grams[:-1], grams[-1], imp, freq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
    ap.add_argument('--out', default='/workspace/ql5')
    ap.add_argument('--smoke', action='store_true')
    # compression
    ap.add_argument('--total-saving', type=float, default=0.2,
                    help='fraction of TOTAL model params to remove (0 = dense control)')
    ap.add_argument('--embed-keep', type=float, default=0.5,
                    help='kept fraction of embedding/LM-head params (1 = untouched)')
    ap.add_argument('--embed-floor', type=float, default=1e-6,
                    help='frequency floor so unseen tokens keep a direction')
    ap.add_argument('--freq-windows', type=int, default=2048,
                    help='extra windows tokenised only to count token frequency')
    ap.add_argument('--targets',
                    default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
    ap.add_argument('--ratios', default='0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85')
    ap.add_argument('--rank-align', type=int, default=64)
    ap.add_argument('--alloc', choices=['greedy', 'sa', 'sqa'], default='greedy',
                    help='greedy: near-optimal on this separable knapsack (F5)')
    ap.add_argument('--alloc-weight', choices=['fisher', 'none'], default='fisher')
    ap.add_argument('--calib', type=int, default=256, help='calibration windows')
    ap.add_argument('--calib-mix', default=DEFAULT_MIX)
    ap.add_argument('--no-decontam', action='store_true')
    # heal
    ap.add_argument('--heal-tokens', type=float, default=100e6)
    ap.add_argument('--heal-mix', default=DEFAULT_MIX)
    ap.add_argument('--source-weight', default='metamath:1.5,gsm8k:1.5,owm:1.2',
                    help='per-source loss multipliers (others 1.0)')
    ap.add_argument('--seq', type=int, default=1024)
    ap.add_argument('--batch', type=int, default=16, help='windows per GPU per step')
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--warmup', type=float, default=0.03)
    ap.add_argument('--kd-alpha', type=float, default=0.7)
    ap.add_argument('--kd-t', type=float, default=1.0)
    ap.add_argument('--kd-chunk', type=int, default=4096)
    ap.add_argument('--heal-scope', choices=['tn', 'all'], default='tn',
                    help='tn: low-rank factors + norms + biases; all: every param')
    ap.add_argument('--regauge-every', type=int, default=0,
                    help='G1: re-gauge every N steps (0 = off)')
    ap.add_argument('--eval-every', type=int, default=0)
    ap.add_argument('--quant', default='none',
                    help='comma list of int8,fp8: fake-quantised copies after heal')
    ap.add_argument('--save-model', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rank, world, device = dist_setup()
    main_rank = rank == 0
    log = (lambda *a, **k: print(*a, **{"flush": True, **k})) if main_rank else (lambda *a, **k: None)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    t_start = time.time()

    # ---------------- model + data ----------------
    if args.smoke:
        V = 512
        model = tiny_qwen3(V).to(device)
        tok = _SmokeTok(V)
        hf.ALLOW_SYNTHETIC = True
        if args.heal_tokens == 100e6:
            args.heal_tokens = 3 * 4 * 64 * world
        args.seq, args.batch = min(args.seq, 64), min(args.batch, 4)
        args.calib, args.freq_windows = min(args.calib, 16), min(args.freq_windows, 16)
        args.rank_align = min(args.rank_align, 8)
        args.kd_chunk = min(args.kd_chunk, 37)        # odd size: exercises the tail
        model_id = os.path.join(args.out, 'tiny_base')
        if main_rank:
            os.makedirs(model_id, exist_ok=True)
            model.save_pretrained(model_id)
        log("[ql5] smoke: tiny random Qwen3 (no downloads)")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32, attn_implementation="sdpa").to(device)
        model_id = args.model
    V = model.config.vocab_size
    tied = bool(getattr(model.config, 'tie_word_embeddings', False))
    log(f"[ql5] world={world} device={device} model={args.model} tied={tied}")

    decontam = None
    if not args.smoke and not args.no_decontam:
        decontam = Decontam.from_eval_sets()
        log(f"[ql5] decontam: {len(decontam.h):,} test-set 13-grams")

    def packer(spec, seed, r=0, w=1):
        return MixturePacker(tok, parse_mix(spec), args.seq, seed=seed, rank=r,
                             world=w, decontam=decontam, synthetic=args.smoke, vocab=V)

    ppl = lambda: hf.ppl_eval(model, tok, device)
    ppl_dense = ppl()
    log(f"[ql5] dense PPL = {ppl_dense:.3f}")

    teacher = None
    if args.heal_tokens > 0 and args.kd_alpha > 0:
        import copy
        teacher = copy.deepcopy(model).to(torch.bfloat16 if device.type == 'cuda'
                                          else torch.float32).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    # ---------------- calibrate (identical on every rank) ----------------
    layers = model.model.layers
    cands = []
    for i, layer in enumerate(layers):
        for name in args.targets.split(','):
            mod = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name)
            cands.append((f"L{i}.{name}", mod))
    total_before = count_params(model)
    cand_params = sum(m.weight.numel() for _, m in cands)
    emb = model.model.embed_tokens
    head = model.lm_head
    emb_params = emb.weight.numel() * (1 if tied else 2)
    report = {'args': vars(args), 'world': world, 'tied': tied,
              'params_total_before': total_before, 'candidate_params': cand_params,
              'embedding_params': emb_params, 'ppl_dense': round(ppl_dense, 4)}

    compress = args.total_saving > 0
    t0 = time.time()
    if compress:
        cp = packer(args.calib_mix, seed=10_000)
        rows = [torch.tensor(next(cp)[0]) for _ in range(args.calib)]
        grams, hgram, imp, freq = calibrate(model, cands, head, rows, device,
                                            fisher=args.alloc_weight == 'fisher')
        for _ in range(args.freq_windows):          # frequency only, no forward
            freq += torch.bincount(torch.tensor(next(cp)[0]), minlength=V)
        report['calib_sources'] = dict(cp.counts)
        log(f"[ql5] calibration: {args.calib} windows, Fisher "
            f"{'on' if args.alloc_weight == 'fisher' else 'off'} ({time.time() - t0:.0f}s)")

        # ---- D1: embedding / LM head ----
        emb_saved, plan_emb = 0, None
        if args.embed_keep < 1:
            E = emb.weight.detach()
            Vv, d = E.shape
            r_e = _align(args.embed_keep * Vv * d / (Vv + d), args.rank_align)
            if r_e < Vv * d / (Vv + d):
                A, Sig, B = weighted_embedding_svd(E, freq, hgram if tied else None,
                                                   floor=args.embed_floor)
                tot = (Sig ** 2).sum()
                rel = float((Sig[r_e:] ** 2).sum() / tot)
                plan_emb = (r_e, A[:, :r_e].float(), B[:r_e].float(), rel)
                emb_saved = emb.weight.numel() - r_e * (Vv + d)
                if not tied:          # head: own weighted factorisation
                    Ah, Sh, Bh = weighted_embedding_svd(head.weight.detach(), freq,
                                                        hgram, floor=args.embed_floor)
                    plan_emb += (Ah[:, :r_e].float(), Bh[:r_e].float())
                    emb_saved *= 2
                log(f"[ql5] D1 embedding rank {r_e}: weighted rel. err {rel:.4f}, "
                    f"saves {emb_saved:,}")
            report['embed_rank'] = r_e

        # ---- layer ranks under the remaining budget ----
        need = args.total_saving * total_before - emb_saved
        layer_saving = max(0.0, need / cand_params)
        ratios = [float(r) for r in args.ratios.split(',')]
        L, K = len(cands), len(ratios) + 1
        err = torch.zeros(L, K); par = torch.zeros(L, K)
        factors = {}
        if layer_saving > 0:
            for l, (_, m) in enumerate(cands):
                US, Sig, Bf = whitened_svd(m.weight.detach(), grams[l])
                factors[l] = (US.float(), Bf.float())
                tot = (Sig ** 2).sum()
                tail = torch.flip(torch.cumsum(torch.flip(Sig ** 2, [0]), 0), [0])
                o, i = m.weight.shape
                for k, rt in enumerate(ratios):
                    r = _align(rt * o * i / (o + i), args.rank_align)
                    err[l, k] = float(tail[r] / tot) if r < len(Sig) else 0.0
                    par[l, k] = min(r * (o + i), o * i)
                err[l, -1], par[l, -1] = 0.0, o * i
            w = imp / imp.mean() if args.alloc_weight == 'fisher' and imp.sum() > 0 \
                else torch.ones(L, dtype=torch.float64)
            errw = err * w[:, None].float()
            ar = allocate(errw, par, cand_params * (1 - layer_saving),
                          method=args.alloc)
            choice = ar[ar['best']]['choice']
            plan = {l: _align(ratios[k] * cands[l][1].weight.shape[0] *
                              cands[l][1].weight.shape[1] /
                              sum(cands[l][1].weight.shape), args.rank_align)
                    for l, k in enumerate(choice) if k < len(ratios)}
            report['layer_saving_target'] = round(layer_saving, 4)
            report['fisher_importance'] = {cands[l][0]: round(float(w[l]), 4)
                                           for l in range(L)}
            report['layer_ranks'] = {cands[l][0]: r for l, r in plan.items()}
            report['alloc_weighted_err'] = round(float(ar[ar['best']]['err']), 5)
        else:
            plan = {}
        del grams

        # ---- apply ----
        with torch.no_grad():
            for l, r in plan.items():
                name, mod = cands[l]
                US, Bf = factors[l]
                new = lowrank_from_svd(US.to(device), Bf.to(device), r,
                                       bias=mod.bias).to(device)
                assert _replace(model, mod, new), name
            if plan_emb is not None:
                r_e, A, B = plan_emb[:3]
                new_e = LowRankEmbedding(A.to(device), B.to(device))
                model.model.embed_tokens = new_e
                if tied:
                    model.lm_head = LowRankHead(tied=new_e)
                else:
                    model.lm_head = LowRankHead(plan_emb[4].to(device),
                                                plan_emb[5].to(device))
                model.config.tie_word_embeddings = False   # shared via module
                report['embed_weighted_rel_err'] = round(plan_emb[3], 5)
        del factors
        log(f"[ql5] compressed {len(plan)}/{len(cands)} layers"
            f"{' + embedding/head' if plan_emb is not None else ''} "
            f"({time.time() - t0:.0f}s)")

    total_after = count_params(model)
    saved = 1 - total_after / total_before
    report.update(params_total_after=total_after,
                  param_saving_pct=round(100 * saved, 2),
                  compress_time_s=round(time.time() - t0, 1))
    log(f"[ql5] params {total_before:,} -> {total_after:,} (-{100 * saved:.2f}% total)")
    report['ppl_compressed'] = round(ppl(), 4) if compress else report['ppl_dense']
    log(f"[ql5] PPL before heal = {report['ppl_compressed']}")

    # ---------------- heal ----------------
    tokens_per_step = args.batch * args.seq * world
    steps = int(args.heal_tokens // tokens_per_step)
    report['heal_steps'] = steps
    report['heal_tokens'] = steps * tokens_per_step
    curve = []
    if steps > 0:
        if args.heal_scope == 'tn':
            for n_, p in model.named_parameters():
                p.requires_grad_(n_.endswith(('.A', '.B', 'bias')) or 'norm' in n_)
        else:
            for p in model.parameters():
                p.requires_grad_(True)
        params = [p for p in model.parameters() if p.requires_grad]
        if world > 1 and device.type == 'cuda':
            from torch.distributed.optim import ZeroRedundancyOptimizer
            opt = ZeroRedundancyOptimizer(params, optimizer_class=torch.optim.AdamW,
                                          lr=args.lr, weight_decay=0.0,
                                          betas=(0.9, 0.95))
        else:
            opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0,
                                    betas=(0.9, 0.95))
        warm = max(1, int(steps * args.warmup))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) *
            (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))))
        if hasattr(model, 'gradient_checkpointing_enable') and device.type == 'cuda':
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False})
        sw = {k: 1.0 for k in parse_mix(args.heal_mix)}
        for part in filter(None, args.source_weight.split(',')):
            k, v = part.split(':')
            if k in sw:
                sw[k] = float(v)
        hp = packer(args.heal_mix, seed=args.seed + 1, r=rank, w=world)
        gauged = [m for m in model.modules()
                  if isinstance(m, (LowRankLinear, LowRankEmbedding))]
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        ac = dict(device_type='cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda')
        t0 = time.time()
        for step in range(steps):
            wins, srcs = zip(*(next(hp) for _ in range(args.batch)))
            x = torch.tensor(wins, device=device)
            wt = torch.tensor([sw[s] for s in srcs], device=device)
            with torch.autocast(**ac):
                h_s = model.model(input_ids=x).last_hidden_state
                h_t = None
                if teacher is not None:
                    with torch.no_grad():
                        h_t = teacher.model(input_ids=x).last_hidden_state
                loss = chunked_kd_loss(h_s, h_t, x, wt, model.lm_head,
                                       teacher.lm_head if teacher is not None else None,
                                       alpha=args.kd_alpha, T=args.kd_t,
                                       chunk=args.kd_chunk)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            allreduce_grads(params, world)
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            if args.regauge_every and (step + 1) % args.regauge_every == 0:
                inner = getattr(opt, 'optim', opt)
                for m in gauged:
                    regauge_(m.A, m.B)
                    for p in (m.A, m.B):      # moments live in the old gauge
                        inner.state.pop(p, None)
                broadcast_params([p for m in gauged for p in (m.A, m.B)], world)
            if step % 50 == 0 or step == steps - 1:
                el = time.time() - t0
                log(f"  step {step:5d}/{steps} loss {loss.item():.4f} gnorm {float(gn):.2f} "
                    f"lr {sched.get_last_lr()[0]:.2e} "
                    f"{(step + 1) * tokens_per_step / el:,.0f} tok/s")
            if args.eval_every and (step + 1) % args.eval_every == 0 and step < steps - 1:
                if main_rank:
                    curve.append({'step': step + 1, 'ppl': round(ppl(), 4)})
                    log(f"  eval step {step + 1}: PPL {curve[-1]['ppl']}")
                if world > 1:
                    dist.barrier()
        report['train_time_s'] = round(time.time() - t0, 1)
        report['train_tokens_per_s'] = round(report['heal_tokens'] / (time.time() - t0), 1)
        report['heal_sources'] = dict(hp.counts)
        if device.type == 'cuda':
            report['peak_vram_gb'] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
        del teacher
    if decontam is not None:
        report['decontam'] = {'checked_docs': decontam.checked,
                              'dropped_docs': decontam.dropped}

    report['heal_curve'] = curve
    report['ppl_healed'] = round(ppl(), 4)
    report['ppl_vs_dense_pct'] = round(100 * (report['ppl_healed'] - ppl_dense) / ppl_dense, 2)
    log(f"[ql5] PPL healed = {report['ppl_healed']} "
        f"({report['ppl_vs_dense_pct']:+.2f}% vs dense)")

    if main_rank:
        os.makedirs(args.out, exist_ok=True)
        if args.save_model:
            save_compressed(model, model_id, f'{args.out}/model.pt')
            if not args.smoke:
                tok.save_pretrained(f'{args.out}/tokenizer')
        # quantised copies (weights only; PPL measured on the dequantised model)
        report['quant'] = {}
        fmts = [q for q in args.quant.split(',') if q and q != 'none']
        if fmts:
            base = {k: v.detach().clone() for k, v in model.state_dict().items()}
            for q in fmts:
                n = quantize_model_(model, q)
                report['quant'][q] = {'quantised_params': n, 'ppl': round(ppl(), 4)}
                log(f"[ql5] {q}: PPL {report['quant'][q]['ppl']}")
                if args.save_model:
                    save_compressed(model, model_id, f'{args.out}/model_{q}.pt')
                model.load_state_dict(base)
        report['wall_s'] = round(time.time() - t_start, 1)
        with open(f'{args.out}/ql5_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        log("[ql5] report ->", f'{args.out}/ql5_report.json')
        log(json.dumps({k: v for k, v in report.items()
                        if k not in ('fisher_importance', 'layer_ranks', 'args')}, indent=2))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
    # Streaming `datasets` iterators leave HTTP worker threads alive; on some torch
    # builds interpreter shutdown then aborts (PyGILState_Release, rc=134) after all
    # outputs are written. Exit directly so a finished run reports rc=0.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
