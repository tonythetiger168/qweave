"""
QL2 benchmarks -- TT layer speed + annealing allocator quality
==============================================================
  python phase4_llm/bench_ql2.py --out /workspace/ql2 [--quick]

[A] TTLinear vs nn.Linear (Qwen3-0.6B shapes), fwd+bwd tokens/s:
      dense | tt-eager (einsum chain) | tt-compiled (torch.compile ->
      Inductor/Triton) | tt-materialize (rebuild W, one cuBLAS GEMM)
[B] allocator: greedy vs SA vs SQA vs exact DP on random multiple-choice
    knapsack instances shaped like a 28-layer x 7-proj model.
"""

import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch

from phase4_llm.tn_layers import TTLinear, tt_matrix_reconstruct
from phase4_llm.anneal_alloc import greedy, anneal, sqa, exact_dp, energy


def timeit(fn, warm=3, iters=10, cuda=False):
    for _ in range(warm):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.time() - t) / iters


class TTMaterialize(torch.nn.Module):
    def __init__(self, tt):
        super().__init__()
        self.tt = tt

    def forward(self, x):
        W = tt_matrix_reconstruct(list(self.tt.cores), self.tt.out_modes,
                                  self.tt.in_modes)[:self.tt.out_features,
                                                    :self.tt.in_features]
        return torch.nn.functional.linear(x, W.to(x.dtype), self.tt.bias)


def bench_layers(dev, quick):
    shapes = [(1024, 2048, 'q_proj'), (1024, 3072, 'up_proj'),
              (3072, 1024, 'down_proj')]
    ranks = [32, 128] if quick else [32, 64, 128, 256]
    ntok = 1024 if quick else 8192
    cuda = dev == 'cuda'
    rows = []
    for fin, fout, name in shapes:
        x = torch.randn(ntok, fin, device=dev, requires_grad=True)
        dense = torch.nn.Linear(fin, fout, device=dev)
        variants = [('dense', dense, fin * fout)]
        for r in ranks:
            tt = TTLinear(fin, fout, d=4, rank=r, eps=0.0, device=dev)
            n = sum(c.numel() for c in tt.cores)
            variants.append((f'tt-eager r{r}', tt, n))
            variants.append((f'tt-mat r{r}', TTMaterialize(tt), n))
            if cuda and not quick:
                variants.append((f'tt-compiled r{r}', torch.compile(tt), n))
        for vname, mod, n in variants:
            def step():
                with torch.autocast(device_type=dev, dtype=torch.bfloat16,
                                    enabled=cuda):
                    y = mod(x)
                y.float().sum().backward()
            try:
                dt = timeit(step, cuda=cuda, iters=3 if quick else 20)
                rows.append({'layer': name, 'in': fin, 'out': fout,
                             'variant': vname, 'params': n,
                             'compression_x': round(fin * fout / n, 2),
                             'ms': round(dt * 1e3, 3),
                             'tokens_per_s': round(ntok / dt)})
            except Exception as e:                       # compile failures etc.
                rows.append({'layer': name, 'variant': vname, 'error': str(e)[:200]})
            print(rows[-1], flush=True)
    return rows


def bench_alloc(dev, quick):
    out = []
    seeds = [0, 1] if quick else [0, 1, 2, 3, 4]
    L = 28 if quick else 196
    for seed in seeds:
        g = torch.Generator().manual_seed(seed)
        ranks = torch.tensor([16, 32, 64, 128, 256], dtype=torch.float)
        base = torch.rand(L, 1, generator=g) * 0.8 + 0.2       # per-layer difficulty
        err = torch.cat([base * torch.exp(-ranks / (40 + 200 * torch.rand(L, 1, generator=g))),
                         torch.zeros(L, 1)], 1) ** 2
        size = torch.randint(1, 4, (L, 1), generator=g).float() * 1e6
        par = torch.cat([size * (ranks / 512) ** 1.3, size], 1)
        budget = float(par[:, -1].sum()) * 0.6
        err, par = err.to(dev), par.to(dev)
        res = {'seed': seed, 'layers': L}
        t = time.time(); c = greedy(err, par, budget)
        res['greedy'] = (float(energy(c, err, par, budget, 1.0)[1]), time.time() - t)
        sw = 100 if quick else 300
        t = time.time(); c, _ = anneal(err, par, budget, sweeps=sw, seed=seed)
        e, _, p = energy(c, err, par, budget, 1.0)
        res['sa'] = (float(e) if p <= budget else None, time.time() - t)
        t = time.time(); c, _ = sqa(err, par, budget, sweeps=sw, seed=seed)
        e, _, p = energy(c, err, par, budget, 1.0)
        res['sqa'] = (float(e) if p <= budget else None, time.time() - t)
        t = time.time(); c = exact_dp(err, par, budget)
        res['exact_dp'] = (float(energy(c.to(dev), err, par, budget, 1.0)[1]),
                           time.time() - t)
        c = exact_dp(err, par, budget, lower_bound=True)
        res['lower_bound'] = float(energy(c.to(dev), err, par, budget, 1.0)[1])
        best = res['exact_dp'][0]
        for k in ('greedy', 'sa', 'sqa'):
            v = res[k][0]
            res[k + '_gap_pct'] = None if v is None else round(100 * (v - best) / best, 3)
        print(res, flush=True)
        out.append(res)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/workspace/ql2')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(a.out, exist_ok=True)
    rep = {'device': dev,
           'gpu': torch.cuda.get_device_name() if dev == 'cuda' else None,
           'layers': bench_layers(dev, a.quick),
           'alloc': bench_alloc(dev, a.quick)}
    json.dump(rep, open(f'{a.out}/ql2_bench.json', 'w'), indent=2)
    print("saved", f'{a.out}/ql2_bench.json')
