"""
QL1.5 — Heal fine-tune for TN-compressed LLM layers (CUDA)

Pipeline (CompactifAI-style):
  1. load a small open model (HF transformers), bf16 on CUDA
  2. sensitivity-profile candidate Linear layers, compress the most
     compressible ones into MPOLinear (adaptive chi)
  3. baseline: WikiText-2 test PPL
  4. heal: AdamW on MPO cores + biases (low LR, cosine, grad ckpt)
  5. report: PPL delta, params saved, peak VRAM, tokens/s

Smoke mode (--smoke) runs the same pipeline on a tiny randomly
initialized GPT without any downloads — use it to verify a fresh GPU
pod before the real run.

RunPod quick start (see docs/QL1.5_RUNPOD.md):
  python phase4_llm/heal_finetune.py --smoke
  python phase4_llm/heal_finetune.py --model Qwen/Qwen3-0.6B \
      --chi 32 --eps 1e-3 --steps 300 --out /workspace/ql15
"""

import argparse, json, math, time
import torch
import torch.nn.functional as F

from phase4_llm.tn_layers import MPOLinear, sensitivity_profile, mpo_num_params


# ----------------------------------------------------------------------
def ppl_eval(model, tokenizer, device, max_sents=64, stride=256, win=1024):
    """WikiText-2 test perplexity (sliding window)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"][:max_sents * 4])
    except Exception:
        print("  [warn] wikitext unavailable, using synthetic text")
        text = "the quick brown fox jumps over the lazy dog. " * 4000
    enc = tokenizer(text, return_tensors="pt").input_ids[0].to(device)
    nll_sum, n_tok = 0.0, 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
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
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(ds["text"][:n * 2])
    except Exception:
        text = "quantum tensor networks compress large language models. " * 4000
    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    rows = []
    for i in range(n):
        j = (i * seq) % max(enc.numel() - seq - 1, 1)
        rows.append(enc[j:j + seq + 1])
    return rows


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
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
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
        print("[ql1.5] smoke: tiny random GPT (no downloads)")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype,
            attn_implementation="flash_attention_2" if device == 'cuda'
            else "eager").to(device)
    model.train()

    # ---------------- compress ----------------
    candidates = []
    if args.smoke:
        for i, b in enumerate(model.blocks):
            for name in ('q', 'o', 'g', 'u'):
                candidates.append((f"block{i}.{name}", b[name]))
    else:
        for i, layer in enumerate(model.model.layers):
            for name, mod in (('q_proj', layer.self_attn.q_proj),
                              ('o_proj', layer.self_attn.o_proj),
                              ('gate_proj', layer.mlp.gate_proj),
                              ('up_proj', layer.mlp.up_proj)):
                candidates.append((f"L{i}.{name}", mod))
    print(f"[ql1.5] {len(candidates)} candidate Linear layers")

    scores = sensitivity_profile([m.weight.detach() for _, m in candidates],
                                 chi_probe=8)
    order = sorted(range(len(candidates)),
                   key=lambda i: scores[i])          # most compressible first
    n_comp = max(1, int(len(candidates) * args.compress_frac))
    picked = order[:n_comp]
    dense_before = count_params(model)

    for idx in picked:
        name, mod = candidates[idx]
        new = MPOLinear(mod.in_features, mod.out_features,
                        chi_max=args.chi, eps=args.eps, bias=mod.bias is not None,
                        device=device, dtype=dtype)
        from phase4_llm.tn_layers import mpo_decompose
        with torch.no_grad():
            cs = mpo_decompose(mod.weight.detach().to(dtype),
                               args.chi, args.eps,
                               new.row_bits, new.col_bits)
            for p, c in zip(new.cores, cs):
                p.copy_(c)
            if mod.bias is not None:
                new.bias.copy_(mod.bias)
        # generic in-place replacement via object identity
        def _replace(root, target, repl):
            for name, child in root.named_children():
                if child is target:
                    setattr(root, name, repl)
                    return True
                if _replace(child, target, repl):
                    return True
            return False
        assert _replace(model, mod, new), f"cannot replace {name}"
        print(f"  compressed {name:14s} score={scores[idx]:.4f}")

    dense_after = count_params(model)
    saved = 1 - dense_after / dense_before
    print(f"[ql1.5] params: {dense_before:,} -> {dense_after:,} "
          f"(saved {100 * saved:.1f}%)")

    # ---------------- eval + heal ----------------
    ppl0 = ppl_eval(model, tok, device)
    print(f"[ql1.5] baseline PPL = {ppl0:.3f}")

    rows = heal_data(tok, n=max(args.steps * args.batch // 8, 64))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    if device == 'cuda':
        model.gradient_checkpointing_enable()
    t0, peak, step = time.time(), 0.0, 0
    for step in range(args.steps):
        batch = torch.stack([rows[(step * args.batch + j) % len(rows)]
                             for j in range(args.batch)]).to(device)
        loss = model(batch, labels=batch).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if device == 'cuda':
            peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
        if step % 50 == 0:
            print(f"  step {step:4d} loss {loss.item():.4f} "
                  f"({(step + 1) / (time.time() - t0):.2f} it/s)")
    ppl1 = ppl_eval(model, tok, device)
    tok_s = args.steps * args.batch * rows[0].numel() / (time.time() - t0)

    report = {'model': args.model, 'chi': args.chi, 'eps': args.eps,
              'compressed_layers': n_comp, 'param_saving_pct': round(100 * saved, 2),
              'ppl_baseline': round(ppl0, 4), 'ppl_healed': round(ppl1, 4),
              'ppl_relative_increase_pct': round(100 * (ppl1 - ppl0) / ppl0, 2),
              'peak_vram_gb': round(peak, 2), 'train_tokens_per_s': round(tok_s, 1),
              'device': device}
    import os
    os.makedirs(args.out, exist_ok=True)
    with open(f'{args.out}/ql15_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("\n[ql1.5] report:", json.dumps(report, indent=2))
    ok = report['ppl_relative_increase_pct'] <= 3.0
    print("QL1.5 ACCEPTANCE (PPL +<=3%):", "PASS" if ok else "FAIL (heal longer)")


if __name__ == '__main__':
    main()
