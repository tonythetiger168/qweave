"""
QL2: annealing optimizer for per-layer TT-rank allocation
=========================================================
Which layer gets which TT rank under a global parameter budget is a
multiple-choice knapsack -- the combinatorial core of TN compression
(Minima's "sensitivity model" stage; FermiQLLM's "quantum simulated
annealing" claim). We solve it three ways and report the gap:

  * greedy   -- marginal error-reduction per parameter (baseline)
  * sa       -- GPU-parallel simulated annealing over R replicas
  * sqa      -- path-integral simulated QUANTUM annealing: P Trotter
                slices of a Potts model coupled by a ferromagnetic term
                J_perp(Gamma) = -(T/2) ln tanh(Gamma / (P T)) that
                tightens as the transverse field Gamma -> 0
                (Santoro et al., Science 2002; GTQA arXiv 2409.12240)
  * exact    -- DP over a discretized budget (small instances; for gap)

All of this is classical. "Quantum" = the SQA sampling schedule, not
hardware. Energy (minimized):
    E(x) = sum_l err[l, x_l] + lam * relu(sum_l par[l, x_l] - P) / P
"""

import math
from typing import Dict, List, Tuple
import torch


def energy(choice, err, par, budget, lam):
    # choice (..., L) long; err/par (L, K)
    L = err.shape[0]
    idx = torch.arange(L, device=err.device)
    e = err[idx, choice].sum(-1)
    p = par[idx, choice].sum(-1)
    return e + lam * torch.relu(p - budget) / budget, e, p


def greedy(err, par, budget) -> torch.Tensor:
    """Start every layer at its cheapest option, buy the best
    error-drop-per-parameter upgrade until the budget is spent."""
    L, K = err.shape
    ch = par.argmin(1).clone()
    used = par[torch.arange(L), ch].sum().item()
    while True:
        best, arg = 0.0, None
        for l in range(L):
            for k in range(K):
                dp = (par[l, k] - par[l, ch[l]]).item()
                de = (err[l, ch[l]] - err[l, k]).item()
                if de > 0 and dp > 0 and used + dp <= budget and de / dp > best:
                    best, arg = de / dp, (l, k, dp)
        if arg is None:
            return ch
        l, k, dp = arg
        ch[l] = k
        used += dp


def anneal(err, par, budget, lam=None, replicas=256, sweeps=400,
           t0=None, t1=None, seed=0, init=None) -> Tuple[torch.Tensor, float]:
    """Vectorized Metropolis SA over `replicas` independent chains."""
    g = torch.Generator(device=err.device).manual_seed(seed)
    L, K = err.shape
    lam = lam if lam is not None else 10.0 * float(err.max())
    t0 = t0 or float(err.std()) + 1e-6
    t1 = t1 or t0 * 1e-3
    ch = torch.randint(0, K, (replicas, L), generator=g, device=err.device)
    if init is not None:
        ch[0] = init
    E, _, _ = energy(ch, err, par, budget, lam)
    best_E, best = float('inf'), ch[0].clone()
    rr = torch.arange(replicas, device=err.device)
    for s in range(sweeps):
        T = t0 * (t1 / t0) ** (s / max(sweeps - 1, 1))
        for _ in range(L):
            l = torch.randint(0, L, (replicas,), generator=g, device=err.device)
            k = torch.randint(0, K, (replicas,), generator=g, device=err.device)
            new = ch.clone()
            new[rr, l] = k
            En, _, _ = energy(new, err, par, budget, lam)
            acc = torch.rand(replicas, generator=g, device=err.device) < \
                torch.exp(-(En - E).clamp(min=0) / T)
            ch = torch.where(acc[:, None], new, ch)
            E = torch.where(acc, En, E)
        _, e, p = energy(ch, err, par, budget, lam)
        Ef = torch.where(p <= budget, e, torch.full_like(e, float('inf')))
        m = Ef.min().item()                       # best FEASIBLE state only
        if m < best_E:
            best_E, best = m, ch[Ef.argmin()].clone()
    return best, best_E


def sqa(err, par, budget, lam=None, trotter=16, replicas=32, sweeps=400,
        T=None, gamma0=None, gamma1=1e-3, seed=0, init=None):
    """Path-integral simulated quantum annealing (Potts generalization).
    State: (replicas, trotter, L). Slice k sees E/P at temperature T;
    neighbouring slices are coupled with J_perp * [x_k != x_{k+1}].
    Each sweep proposes one move per (replica, slice) L times, updating
    even then odd slices (checkerboard) so neighbours are held fixed."""
    g = torch.Generator(device=err.device).manual_seed(seed)
    L, K = err.shape
    P = trotter
    dev = err.device
    lam = lam if lam is not None else 10.0 * float(err.max())
    T = T or 0.05 * (float(err.std()) + 1e-6) / P      # P*T = classical temp
    gamma0 = gamma0 or 2.0 * (float(err.std()) + 1e-6)
    ch = torch.randint(0, K, (replicas, P, L), generator=g, device=dev)
    if init is not None:
        ch[0, :] = init
    idx = torch.arange(L, device=dev)

    def slice_E(c):                                       # (..., L) -> (...)
        e = err[idx, c].sum(-1)
        p = par[idx, c].sum(-1)
        return e + lam * torch.relu(p - budget) / budget

    E = slice_E(ch)                                       # (R, P)
    best_E, best = float('inf'), None
    ar_r = torch.arange(replicas, device=dev)[:, None]
    ar_p = torch.arange(P, device=dev)[None, :]
    parity = (torch.arange(P, device=dev) % 2)[None, :]
    for s in range(sweeps):
        gam = gamma0 * (gamma1 / gamma0) ** (s / max(sweeps - 1, 1))
        Jp = -0.5 * T * math.log(math.tanh(gam / (P * T)) + 1e-300)
        for _ in range(L):
            for half in (0, 1):
                l = torch.randint(0, L, (replicas, P), generator=g, device=dev)
                v = torch.randint(0, K, (replicas, P), generator=g, device=dev)
                old = ch[ar_r, ar_p, l]
                new = ch.clone()
                new[ar_r, ar_p, l] = v
                En = slice_E(new)
                up = ch[ar_r, (ar_p + 1) % P, l]
                dn = ch[ar_r, (ar_p - 1) % P, l]
                dJ = Jp * ((v != up).float() + (v != dn).float()
                           - (old != up).float() - (old != dn).float())
                dE = (En - E) / P + dJ
                acc = (torch.rand(replicas, P, generator=g, device=dev) <
                       torch.exp(-dE.clamp(min=0) / T)) & (parity == half)
                ch = torch.where(acc[..., None], new, ch)
                E = torch.where(acc, En, E)
        e = err[idx, ch].sum(-1)
        Ef = torch.where(par[idx, ch].sum(-1) <= budget, e,
                         torch.full_like(e, float('inf')))
        m = Ef.min().item()                       # best FEASIBLE state only
        if m < best_E:
            flat = Ef.argmin().item()
            best_E, best = m, ch[flat // P, flat % P].clone()
    if best is None:                              # nothing feasible found
        best = init.clone() if init is not None else ch[0, 0].clone()
    return best, best_E


def exact_dp(err, par, budget, bins=100_000, lower_bound=False) -> torch.Tensor:
    """Multiple-choice knapsack DP on a discretized budget (numpy-vectorized).
    Costs are rounded UP (feasible, near-exact for fine bins); with
    lower_bound=True they are rounded DOWN, giving a relaxation whose
    error is a lower bound on the true optimum."""
    import numpy as np
    L, K = err.shape
    unit = budget / bins
    w = par.cpu().double().numpy() / unit
    w = (np.floor(w) if lower_bound else np.ceil(w)).astype(np.int64)
    e = err.cpu().double().numpy()
    INF = np.inf
    dp = np.full(bins + 1, INF); dp[0] = 0.0
    back = np.zeros((L, bins + 1), dtype=np.int16)
    for l in range(L):
        nd = np.full(bins + 1, INF)
        for k in range(K):
            wk = w[l, k]
            if wk > bins:
                continue
            cand = np.full(bins + 1, INF)
            cand[wk:] = dp[:bins + 1 - wk] + e[l, k]
            better = cand < nd
            nd[better] = cand[better]
            back[l, better] = k
        dp = nd
    b = int(np.argmin(dp))
    ch = [0] * L
    for l in range(L - 1, -1, -1):
        k = int(back[l, b])
        ch[l] = k
        b -= w[l, k]
    return torch.tensor(ch, device=err.device)


def allocate(err, par, budget, method='sqa', **kw) -> Dict:
    """Run greedy (+ chosen annealer seeded with greedy); return the better."""
    g = greedy(err, par, budget)
    Eg, eg, pg = energy(g, err, par, budget, kw.get('lam', 10.0 * float(err.max())))
    out = {'greedy': {'err': float(eg), 'params': float(pg), 'choice': g.tolist()}}
    if method in ('sa', 'sqa'):
        fn = anneal if method == 'sa' else sqa
        c, _ = fn(err, par, budget, init=g, **kw)
        _, e, p = energy(c, err, par, budget, 1.0)
        out[method] = {'err': float(e), 'params': float(p), 'choice': c.tolist()}
    feas = [(v['err'], k) for k, v in out.items() if v['params'] <= budget * 1.0001]
    out['best'] = min(feas)[1] if feas else 'greedy'
    return out
