"""
QL1.5 — Heal fine-tune for TN-compressed LLM layers (CUDA)

Pipeline (CompactifAI-style):
  1. load a small open model (HF transformers), bf16 on CUDA
  2. sensitivity-profile candidate Linear layers, compress the most
     compressible ones into MPOLinear (adaptive chi)
  3. baseline: WikiText-2 test PPL
  4. heal: AdamW on MPO cores + biases (low LR, cosine, grad ckpt)
  5. report: PPL delta, params saved, peak VRAM, tokens/s

QL2 upgrades (2026-10-28):
  * --format tt (default): interleaved TT-matrix/MPO cores (TTLinear);
    the old row|col ordering (--format mpo) caps matrix rank at chi.
  * --target-saving: per-layer TT rank chosen under a global parameter
    budget by simulated quantum annealing (anneal_alloc.allocate).
  * acceptance compares the healed model against the UNCOMPRESSED
    dense model (ppl_dense), not against the compressed starting point.
  * --distill: heal with KL to the frozen dense teacher + CE.

Smoke mode (--smoke) runs the same pipeline on a tiny randomly
initialized GPT without any downloads — use it to verify a fresh GPU
pod before the real run.

RunPod quick start (see docs/QL1.5_RUNPOD.md):
  python phase4_llm/heal_finetune.py --smoke
  python phase4_llm/heal_finetune.py --model Qwen/Qwen3-0.6B \
      --chi 32 --eps 1e-3 --steps 300 --out /workspace/ql15
"""

import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
import torch.nn.functional as F

from phase4_llm.tn_layers import (MPOLinear, TTLinear, sensitivity_profile,
                                  mpo_num_params, tt_rel_error)
from phase4_llm.anneal_alloc import allocate


ALLOW_SYNTHETIC = False      # only --smoke may fall back to synthetic text


# ----------------------------------------------------------------------
def ppl_eval(model, tokenizer, device, max_sents=64, stride=1024, win=1024):
    """WikiText-2 test perplexity (non-overlapping windows)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"][:max_sents * 4])
    except Exception:
        if not ALLOW_SYNTHETIC:
            raise
        print("  [warn] wikitext unavailable, using synthetic text")
        text = "the quick brown fox jumps over the lazy dog. " * 4000
    enc = tokenizer(text, return_tensors="pt").input_ids[0].to(device)
    nll_sum, n_tok = 0.0, 0
    was_training = model.training
    model.eval()
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                         enabled=str(device).startswith('cuda')):
        for i in range(0, min(enc.numel(), win * max_sents), stride):
            chunk = enc[i:i + win]
            if chunk.numel() < 8:
                break
            out = model(chunk.unsqueeze(0), labels=chunk.unsqueeze(0))
            nll_sum += out.loss.item() * (chunk.numel() - 1)
            n_tok += chunk.numel() - 1
    if was_training:
        model.train()
    return math.exp(nll_sum / max(n_tok, 1))


def heal_data(tokenizer, n=512, seq=256):
    """n random windows of seq+1 tokens from WikiText-103 train
    (falls back to WikiText-2, then synthetic)."""
    text = None
    for cfg, rows in (("wikitext-103-raw-v1", 200_000), ("wikitext-2-raw-v1", None)):
        try:
            from datasets import load_dataset
            ds = load_dataset("Salesforce/wikitext", cfg, split="train")
            text = "\n\n".join(ds["text"][:rows] if rows else ds["text"])
            break
        except Exception:
            continue
    if text is None:
        if not ALLOW_SYNTHETIC:
            raise RuntimeError("WikiText unavailable -- refusing synthetic heal data")
        text = "quantum tensor networks compress large language models. " * 4000
    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(1)
    hi = max(enc.numel() - seq - 1, 1)
    starts = torch.randint(0, hi, (n,), generator=g).tolist()
    return [enc[j:j + seq + 1] for j in starts]


def count_params(m):
    return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-0.6B')
    ap.add_argument('--chi', type=int, default=32)
    ap.add_argument('--eps', type=float, default=1e-3)
    ap.add_argument('--compress-frac', type=float, default=0.5,
                    help='fraction of candidate layers to compress')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--out', default='/workspace/ql15')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--format', choices=['tt', 'mpo'], default='tt')
    ap.add_argument('--d', type=int, default=4, help='TT sites per layer')
    ap.add_argument('--ranks', default='16,32,64,128,256',
                    help='TT rank options for the allocator')
    ap.add_argument('--target-saving', type=float, default=0.3,
                    help='fraction of candidate-layer params to remove')
    ap.add_argument('--alloc', choices=['greedy', 'sa', 'sqa'], default='sqa')
    ap.add_argument('--alloc-sweeps', type=int, default=300)
    ap.add_argument('--targets',
                    default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
    ap.add_argument('--seq', type=int, default=512)
    ap.add_argument('--distill', action='store_true')
    ap.add_argument('--kd-alpha', type=float, default=0.5)
    ap.add_argument('--kd-t', type=float, default=1.0)
    ap.add_argument('--heal-scope', choices=['all', 'tn'], default='all')
    ap.add_argument('--gate', type=float, default=3.0)
    ap.add_argument('--save-model', action='store_true')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32  # autocast
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    print(f"[ql1.5] device={device} dtype={dtype} model={args.model}")

    # ---------------- model ----------------
    if args.smoke:
        class Tiny(torch.nn.Module):
            def __init__(self, V=2000, d=128, L=2):
                super().__init__()
                self.emb = torch.nn.Embedding(V, d)
                self.blocks = torch.nn.ModuleList([
                    torch.nn.ModuleDict({
                        'q': torch.nn.Linear(d, d), 'o': torch.nn.Linear(d, d),
                        'g': torch.nn.Linear(d, 4 * d), 'u': torch.nn.Linear(4 * d, d)})
                    for _ in range(L)])
                self.ln = torch.nn.LayerNorm(d)
                self.head = torch.nn.Linear(d, V, bias=False)
                self.config = type('C', (), {'pad_token_id': 0})

            def forward(self, ids, labels=None):
                h = self.emb(ids)
                for b in self.blocks:
                    h = h + torch.nn.functional.gelu(b['o'](
                        torch.nn.functional.gelu(b['q'](h))))
                    h = h + b['u'](torch.nn.functional.gelu(b['g'](h)))
                logits = self.head(self.ln(h))
                loss = None
                if labels is not None:
                    loss = F.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)),
                        labels[:, 1:].reshape(-1))
                return type('O', (), {'loss': loss, 'logits': logits})()

        model = Tiny().to(device)
        tok = lambda text, return_tensors="pt": type(
            'T', (), {'input_ids': torch.randint(0, 2000, (1, 512))})()
        global ALLOW_SYNTHETIC
        ALLOW_SYNTHETIC = True
        print("[ql1.5] smoke: tiny random GPT (no downloads)")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32,     # fp32 master weights,
            attn_implementation="sdpa").to(device)     # bf16 via autocast
    model.train()

    # ---------------- dense reference ----------------
    ppl_dense = ppl_eval(model, tok, device)
    print(f"[ql1.5] dense PPL = {ppl_dense:.3f}")
    teacher = None
    if args.distill:
        import copy
        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    # ---------------- compress ----------------
    candidates = []
    if args.smoke:
        for i, b in enumerate(model.blocks):
            for name in ('q', 'o', 'g', 'u'):
                candidates.append((f"block{i}.{name}", b[name]))
    else:
        for i, layer in enumerate(model.model.layers):
            for name in args.targets.split(','):
                mod = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name)
                candidates.append((f"L{i}.{name}", mod))
    print(f"[ql1.5] {len(candidates)} candidate Linear layers")
    dense_before = count_params(model)
    cand_params = sum(m.weight.numel() for _, m in candidates)

    plan = {}                                  # idx -> rank (tt) / True (mpo)
    alloc_report = None
    if args.format == 'tt' and args.target_saving <= 0:
        print("[ql1.5] target-saving 0 -> dense reference run (no compression)")
    elif args.format == 'tt':
        ranks = [int(r) for r in args.ranks.split(',')]
        L, K = len(candidates), len(ranks) + 1           # last option = dense
        err = torch.zeros(L, K); par = torch.zeros(L, K)
        t0 = time.time()
        for l, (_, m) in enumerate(candidates):
            W = m.weight.detach()
            for k, r in enumerate(ranks):
                e, n = tt_rel_error(W, args.d, r)
                err[l, k], par[l, k] = e ** 2, n
            err[l, -1], par[l, -1] = 0.0, W.numel()
        budget = cand_params * (1 - args.target_saving)
        alloc_report = allocate(err, par, budget, method=args.alloc,
                                sweeps=args.alloc_sweeps)
        choice = alloc_report[alloc_report['best']]['choice']
        print(f"[ql2] rank allocation ({time.time() - t0:.0f}s): "
              + ", ".join(f"{k}: err={v['err']:.3f} params={v['params']:.3e}"
                          for k, v in alloc_report.items() if k != 'best')
              + f" -> {alloc_report['best']}")
        for l, k in enumerate(choice):
            if k < len(ranks):
                plan[l] = ranks[k]
    else:
        scores = sensitivity_profile([m.weight.detach() for _, m in candidates],
                                     chi_probe=8)
        order = sorted(range(len(candidates)), key=lambda i: scores[i])
        for idx in order[:max(1, int(len(candidates) * args.compress_frac))]:
            plan[idx] = True

    def _replace(root, target, repl):
        for name, child in root.named_children():
            if child is target:
                setattr(root, name, repl)
                return True
            if _replace(child, target, repl):
                return True
        return False

    from phase4_llm.tn_layers import mpo_decompose
    for idx, r in plan.items():
        name, mod = candidates[idx]
        with torch.no_grad():
            if args.format == 'tt':
                new = TTLinear(mod.in_features, mod.out_features, d=args.d,
                               rank=r, eps=0.0, bias=mod.bias is not None,
                               weight=mod.weight.detach(), device=device,
                               dtype=torch.float32)
            else:
                new = MPOLinear(mod.in_features, mod.out_features,
                                chi_max=args.chi, eps=args.eps,
                                bias=mod.bias is not None, device=device)
                cs = mpo_decompose(mod.weight.detach().float(), args.chi, args.eps,
                                   new.row_bits, new.col_bits)
                for p, c in zip(new.cores, cs):
                    p.copy_(c)
            if mod.bias is not None:
                new.bias.copy_(mod.bias)
        assert _replace(model, mod, new), f"cannot replace {name}"
    print(f"[ql1.5] compressed {len(plan)}/{len(candidates)} layers")

    dense_after = count_params(model)
    saved = 1 - dense_after / dense_before
    print(f"[ql1.5] params: {dense_before:,} -> {dense_after:,} "
          f"(saved {100 * saved:.1f}% total, "
          f"{100 * (dense_before - dense_after) / cand_params:.1f}% of candidates)")

    # ---------------- eval + heal ----------------
    ppl0 = ppl_eval(model, tok, device)
    print(f"[ql1.5] compressed PPL (before heal) = {ppl0:.3f}")

    rows = heal_data(tok, n=args.steps * args.batch, seq=args.seq)
    if args.heal_scope == 'tn':      # only TN cores + biases (+ norms)
        for n_, p in model.named_parameters():
            p.requires_grad_('cores' in n_ or 'norm' in n_ or n_.endswith('bias'))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warm = max(1, args.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) *
        0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    if device == 'cuda' and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    model.train()
    t0, peak, step = time.time(), 0.0, 0
    for step in range(args.steps):
        batch = torch.stack([rows[(step * args.batch + j) % len(rows)]
                             for j in range(args.batch)]).to(device)
        with torch.autocast(device_type='cuda', dtype=dtype,
                            enabled=device == 'cuda'):
            out = model(batch, labels=batch)
            loss = out.loss
            if teacher is not None:
                with torch.no_grad():
                    tl = teacher(batch).logits.float()
                kd = F.kl_div(F.log_softmax(out.logits.float() / args.kd_t, -1),
                              F.log_softmax(tl / args.kd_t, -1), log_target=True,
                              reduction='batchmean') * args.kd_t ** 2 / batch.shape[1]
                loss = (1 - args.kd_alpha) * loss + args.kd_alpha * kd
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if device == 'cuda':
            peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
        if step % 50 == 0 or step == args.steps - 1:
            print(f"  step {step:4d} loss {loss.item():.4f} "
                  f"({(step + 1) / (time.time() - t0):.2f} it/s)", flush=True)
    train_time = time.time() - t0
    tok_s = args.steps * args.batch * rows[0].numel() / train_time
    del teacher
    ppl1 = ppl_eval(model, tok, device)

    report = {'model': args.model, 'format': args.format, 'd': args.d,
              'ranks': args.ranks, 'target_saving': args.target_saving,
              'chi': args.chi, 'eps': args.eps,
              'alloc': args.alloc, 'distill': args.distill,
              'steps': args.steps, 'batch': args.batch, 'seq': args.seq,
              'compressed_layers': len(plan), 'candidate_layers': len(candidates),
              'params_total_before': dense_before, 'params_total_after': dense_after,
              'param_saving_pct': round(100 * saved, 2),
              'candidate_saving_pct': round(100 * (dense_before - dense_after)
                                            / cand_params, 2),
              'ppl_dense': round(ppl_dense, 4),
              'ppl_compressed': round(ppl0, 4), 'ppl_healed': round(ppl1, 4),
              'ppl_relative_increase_pct': round(100 * (ppl1 - ppl_dense) / ppl_dense, 2),
              'heal_recovery_pct': round(100 * (ppl0 - ppl1) /
                                         max(ppl0 - ppl_dense, 1e-9), 1),
              'peak_vram_gb': round(peak, 2), 'train_tokens_per_s': round(tok_s, 1),
              'train_time_s': round(train_time, 1), 'device': device,
              'allocation': {k: ({kk: vv for kk, vv in v.items() if kk != 'choice'}
                                 if isinstance(v, dict) else v)
                             for k, v in (alloc_report or {}).items()}}
    import os
    os.makedirs(args.out, exist_ok=True)
    with open(f'{args.out}/ql15_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    if args.save_model and not args.smoke:
        torch.save(model, f'{args.out}/model.pt')
        tok.save_pretrained(f'{args.out}/tokenizer')
    print("\n[ql1.5] report:", json.dumps(report, indent=2))
    ok = report['ppl_relative_increase_pct'] <= args.gate
    print(f"QL1.5 ACCEPTANCE (PPL vs dense +<={args.gate}%):",
          "PASS" if ok else "FAIL")


if __name__ == '__main__':
    main()
