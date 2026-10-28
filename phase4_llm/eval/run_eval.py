"""
QL4: open benchmark & reproducibility harness
=============================================
Head-to-head: dense base model vs TN-compressed (+healed) model, every
number reproducible from repo + this command line.

  1. lm-evaluation-harness tasks (multiple-choice / generation), via
     HFLM(pretrained=<in-memory model>) so custom TT layers just work.
  2. robustness suite (QRB-style, our own definition -- see docs):
       * noise:   WikiText-2 PPL with 0 / 5 / 10 % character corruption
       * drift:   NLL per position bucket over long windows (WikiText-103)
  3. JSON + markdown table per model, and a combined comparison.

  python phase4_llm/eval/run_eval.py --dense Qwen/Qwen3-0.6B \
      --compressed /workspace/ql15/model.pt --out /workspace/ql4
"""

import argparse, json, math, os, random, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import torch
import torch.nn.functional as F

DEFAULT_TASKS = "arc_easy,arc_challenge,hellaswag,piqa,winogrande,mmlu,gsm8k"


def load(spec, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if spec.endswith('.pt'):
        from phase4_llm.tn_layers import load_compressed
        model = load_compressed(spec, device)
        tok = AutoTokenizer.from_pretrained(os.path.join(os.path.dirname(spec),
                                                         'tokenizer'))
    else:
        tok = AutoTokenizer.from_pretrained(spec)
        model = AutoModelForCausalLM.from_pretrained(
            spec, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    model.eval()
    return model.to(torch.bfloat16), tok


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def corrupt(text, rate, rng):
    if rate <= 0:
        return text
    chars = list(text)
    alpha = "abcdefghijklmnopqrstuvwxyz"
    for i in range(len(chars)):
        if chars[i].isalpha() and rng.random() < rate:
            op = rng.random()
            if op < 0.5:
                chars[i] = rng.choice(alpha)
            elif op < 0.75 and i + 1 < len(chars):
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
            else:
                chars[i] = ''
    return ''.join(chars)


@torch.no_grad()
def window_nll(model, ids, win, device, buckets=None):
    """Mean NLL over non-overlapping windows; optional per-bucket NLL."""
    tot, n, k = 0.0, 0, 0
    bsum = None
    for i in range(0, ids.numel() - win, win):
        x = ids[i:i + win].unsqueeze(0).to(device)
        logits = model(x).logits.float()
        nll = F.cross_entropy(logits[0, :-1], x[0, 1:], reduction='none')
        tot += nll.sum().item(); n += nll.numel(); k += 1
        if buckets:
            b = torch.stack([nll[s:e].mean() for s, e in buckets]).cpu()
            bsum = b if bsum is None else bsum + b
    return tot / max(n, 1), (bsum / max(k, 1)).tolist() if bsum is not None else None


def robustness(model, tok, device, drift_len=8192, seed=0):
    from datasets import load_dataset
    out = {}
    test = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                                    split="test")["text"])[:300_000]
    clean = None
    for rate in (0.0, 0.05, 0.10):
        ids = tok(corrupt(test, rate, random.Random(seed)),
                  return_tensors="pt").input_ids[0]
        nll, _ = window_nll(model, ids, 1024, device)
        out[f'ppl_noise_{int(rate * 100)}'] = round(math.exp(nll), 3)
        if rate == 0:
            clean = nll
        else:
            out[f'nll_gap_noise_{int(rate * 100)}'] = round(nll - clean, 4)
    long = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                                    split="test")["text"])
    ids = tok(long, return_tensors="pt").input_ids[0][:drift_len * 6 + 1]
    edges = [0, 512, 2048, 4096, drift_len - 1]
    bk = list(zip(edges[:-1], edges[1:]))
    nll, b = window_nll(model, ids, drift_len, device, bk)
    out['ppl_long'] = round(math.exp(nll), 3)
    out['nll_by_position'] = {f'{s}-{e}': round(v, 4) for (s, e), v in zip(bk, b)}
    return out


def lm_eval_tasks(model, tok, tasks, limit, batch):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tok,
              batch_size=int(batch) if str(batch).isdigit() else batch)
    res = lm_eval.simple_evaluate(model=lm, tasks=tasks.split(','), limit=limit,
                                  log_samples=False, random_seed=0,
                                  numpy_random_seed=0, torch_random_seed=0)
    out = {}
    for t, m in res['results'].items():
        for key in ('acc_norm,none', 'acc,none', 'exact_match,strict-match',
                    'exact_match,flexible-extract'):
            if key in m:
                out[t] = round(100 * m[key], 2)
                out[t + '_metric'] = key.split(',')[0]
                se = m.get(key.replace(',', '_stderr,'))
                if isinstance(se, (int, float)):
                    out[t + '_stderr'] = round(100 * se, 2)
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dense', default='Qwen/Qwen3-0.6B')
    ap.add_argument('--compressed', default=None, help='model.pt from heal_finetune')
    ap.add_argument('--tasks', default=DEFAULT_TASKS)
    ap.add_argument('--limit', type=float, default=500)
    ap.add_argument('--batch', default='16')
    ap.add_argument('--skip-lm-eval', action='store_true')
    ap.add_argument('--out', default='/workspace/ql4')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)
    specs = [('dense', args.dense)] + ([('compressed', args.compressed)]
                                       if args.compressed else [])
    allres = {}
    for name, spec in specs:
        t0 = time.time()
        model, tok = load(spec, device)
        r = {'spec': spec, 'params': nparams(model)}
        print(f"[ql4] {name}: {spec} ({r['params']:,} params)", flush=True)
        r['robustness'] = robustness(model, tok, device)
        print(f"[ql4] {name} robustness: {r['robustness']}", flush=True)
        if not args.skip_lm_eval:
            lim = int(args.limit) if args.limit >= 1 else args.limit
            r['tasks'] = lm_eval_tasks(model, tok, args.tasks, lim, args.batch)
            print(f"[ql4] {name} tasks: {r['tasks']}", flush=True)
        r['wall_s'] = round(time.time() - t0, 1)
        allres[name] = r
        with open(f'{args.out}/ql4_{name}.json', 'w') as f:
            json.dump(r, f, indent=2)
        del model
        torch.cuda.empty_cache() if device == 'cuda' else None

    # combined markdown table
    lines = ["| metric | " + " | ".join(allres) + " |",
             "|---|" + "---|" * len(allres)]
    keys = ['params']
    rows = {k: [str(v[k]) for v in allres.values()] for k in keys}
    for k in allres['dense']['robustness']:
        if k != 'nll_by_position':
            rows[k] = [str(v['robustness'].get(k)) for v in allres.values()]
    for k in allres['dense'].get('tasks', {}):
        if not k.endswith(('_metric', '_stderr')):
            rows[k] = [str(v.get('tasks', {}).get(k)) for v in allres.values()]
    lines += [f"| {k} | " + " | ".join(v) + " |" for k, v in rows.items()]
    md = "\n".join(lines)
    with open(f'{args.out}/ql4_table.md', 'w') as f:
        f.write(md + "\n")
    with open(f'{args.out}/ql4_report.json', 'w') as f:
        json.dump(allres, f, indent=2)
    print(md)


if __name__ == '__main__':
    main()
