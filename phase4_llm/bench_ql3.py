"""
QL3 benchmarks -- entanglement diagnostics + QSA attention
==========================================================
  python phase4_llm/bench_ql3.py --model Qwen/Qwen3-0.6B --out /workspace/ql3

[A] entanglement spectrum of every attention head (eager attention on a
    2K-token WikiText window) -> per-layer, per-head effective rank
[B] kernel: SDPA (FlashAttention) causal vs QSA (flex_attention +
    block mask, selection cost included) at 8K / 16K / 32K
[C] quality: long-window PPL (16K) on WikiText-103 test -- dense SDPA vs
    QSA uniform top-k vs QSA entanglement-adaptive top-k (same mean k)

Acceptance (QL3): >= 1.2x attention throughput at 16K+ with <= 1 %
relative PPL increase vs dense.
"""

import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn.functional as F

from phase4_llm import qsa_attention as Q


def diag(model_id, dev, ntok=2048):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(model_id)
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32,
                                             attn_implementation="eager").to(dev).eval()
    text = "\n\n".join(load_dataset("wikitext", "wikitext-103-raw-v1",
                                    split="test")["text"][:3000])
    ids = tok(text, return_tensors="pt").input_ids[:, :ntok].to(dev)
    with torch.no_grad():
        att = m(ids, output_attentions=True).attentions      # L x (1,H,N,N)
    per_layer = []
    for A in att:
        st = Q.attention_entanglement(A[0])
        per_layer.append({k: v.tolist() for k, v in st.items()})
    del m, att
    torch.cuda.empty_cache() if dev == 'cuda' else None
    return per_layer


def kernel_bench(dev, lens, H=16, Hkv=8, D=128, topk=8, block=128):
    rows = []
    for N in lens:
        q = torch.randn(1, H, N, D, device=dev, dtype=torch.bfloat16)
        k = torch.randn(1, Hkv, N, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(1, Hkv, N, D, device=dev, dtype=torch.bfloat16)

        def dense():
            return F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                  enable_gqa=True)

        def qsa():
            return Q.qsa_attention(q, k, v, topk=topk, block=block)

        def run(fn, it=20):
            for _ in range(3):
                fn()
            torch.cuda.synchronize(); t = time.time()
            for _ in range(it):
                fn()
            torch.cuda.synchronize()
            return (time.time() - t) / it

        td, tq = run(dense), run(qsa)
        sel = Q.qsa_select(q, k, block, topk)
        rows.append({'N': N, 'dense_ms': round(td * 1e3, 3), 'qsa_ms': round(tq * 1e3, 3),
                     'speedup': round(td / tq, 3), 'density': round(Q.density(sel), 4),
                     'topk': topk, 'block': block})
        print(rows[-1], flush=True)
    return rows


def long_ppl(model_id, dev, win, nwin, mode, per_layer_topk=None, topk=8):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    Q.register_hf()
    tok = AutoTokenizer.from_pretrained(model_id)
    m = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa" if mode == 'dense' else "qsa").to(dev).eval()
    Q.QSA_CFG.update(topk=topk, per_layer_topk=per_layer_topk)
    text = "\n\n".join(load_dataset("wikitext", "wikitext-103-raw-v1",
                                    split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    tot, n = 0.0, 0
    t0 = time.time()
    with torch.no_grad():
        for w in range(nwin):
            x = ids[w * win:(w + 1) * win].unsqueeze(0).to(dev)
            if x.shape[1] < win:
                break
            lg = m(x).logits.float()
            tot += F.cross_entropy(lg[0, :-1], x[0, 1:], reduction='sum').item()
            n += win - 1
    del m
    torch.cuda.empty_cache()
    return {'mode': mode, 'win': win, 'windows': nwin, 'ppl': round(math.exp(tot / n), 4),
            'wall_s': round(time.time() - t0, 1)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-0.6B')
    ap.add_argument('--out', default='/workspace/ql3')
    ap.add_argument('--topk', type=int, default=8)
    ap.add_argument('--win', type=int, default=16384)
    ap.add_argument('--nwin', type=int, default=8)
    a = ap.parse_args()
    dev = 'cuda'
    os.makedirs(a.out, exist_ok=True)
    rep = {'gpu': torch.cuda.get_device_name(), 'model': a.model}

    print("[ql3] A: entanglement diagnostics", flush=True)
    d = diag(a.model, dev)
    rep['entanglement'] = [{'layer': i,
                            'eff_rank': [round(x, 2) for x in L['eff_rank']],
                            'row_entropy': [round(x, 3) for x in L['row_entropy']]}
                           for i, L in enumerate(d)]
    er = torch.tensor([L['eff_rank'] for L in d])                 # (L, H)
    rep['eff_rank_summary'] = {'min': er.min().item(), 'median': er.median().item(),
                               'max': er.max().item()}
    plt = [Q.head_budgets(er[l], a.topk, kmin=2, kmax=4 * a.topk)
           for l in range(er.shape[0])]
    # rescale so the mean budget matches the uniform top-k (same compute)
    mean_k = sum(map(sum, plt)) / (len(plt) * len(plt[0]))
    plt = [[max(2, round(k * a.topk / mean_k)) for k in row] for row in plt]
    rep['adaptive_topk'] = plt
    json.dump(rep, open(f'{a.out}/ql3_bench.json', 'w'), indent=2)

    print("[ql3] B: kernel throughput", flush=True)
    rep['kernel'] = kernel_bench(dev, [8192, 16384, 32768], topk=a.topk)
    rep['kernel_2x'] = kernel_bench(dev, [16384, 32768], topk=2 * a.topk)
    json.dump(rep, open(f'{a.out}/ql3_bench.json', 'w'), indent=2)

    print("[ql3] C: long-context PPL", flush=True)
    rep['quality'] = []
    for mode, kw in (('dense', {}), ('qsa_uniform', {'topk': a.topk}),
                     ('qsa_adaptive', {'topk': a.topk, 'per_layer_topk': plt}),
                     ('qsa_uniform_2x', {'topk': 2 * a.topk})):
        r = long_ppl(a.model, dev, a.win, a.nwin, mode, **kw)
        print(r, flush=True)
        rep['quality'].append(r)
        json.dump(rep, open(f'{a.out}/ql3_bench.json', 'w'), indent=2)
    base = rep['quality'][0]['ppl']
    for r in rep['quality'][1:]:
        r['ppl_rel_pct'] = round(100 * (r['ppl'] - base) / base, 3)
    json.dump(rep, open(f'{a.out}/ql3_bench.json', 'w'), indent=2)
    print(json.dumps(rep['quality'], indent=2))
