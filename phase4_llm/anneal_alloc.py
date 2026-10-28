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
    best_E, best = E.min().item(), ch[E.argmin()].clone()
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
        m = E.min().item()
        if m < best_E:
            best_E, best = m, ch[E.argmin()].clone()
    return best, best_E


def sqa(err, par, budget, lam=None, trotter=16, replicas=32, sweeps=400,
        T=None, gamma0=None, gamma1=1e-3, seed=0, init=None):
    """Path-integral simulated quantum annealing (Potts generalization).
    State: (replicas, trotter, L). Each slice sees E/P; neighbouring
    slices are coupled with J_perp * [x_k != x_{k+1}]."""
    g = torch.Generator(device=err.device).manual_seed(seed)
    L, K = err.shape
    P = trotter
    lam = lam if lam is not None else 10.0 * float(err.max())
    T = T or 0.05 * (float(err.std()) + 1e-6)
    gamma0 = gamma0 or 3.0 * float(err.std() + 1e-6)
    ch = torch.randint(0, K, (replicas, P, L), generator=g, device=err.device)
    if init is not None:
        ch[0, :] = init
    idx = torch.arange(L, device=err.device)

    def slice_E(c):
        e = err[idx, c].sum(-1)
        p = par[idx, c].sum(-1)
        return e + lam * torch.relu(p - budget) / budget

    best_E, best = float('inf'), None
    R = torch.arange(replicas, device=err.device)
    for s in range(sweeps):
        gam = gamma0 * (gamma1 / gamma0) ** (s / max(sweeps - 1, 1))
        Jp = -0.5 * T * math.log(math.tanh(gam / (P * T)) + 1e-300)
        for _ in range(L):
            kslice = torch.randint(0, P, (replicas,), generator=g, device=err.device)
            l = torch.randint(0, L, (replicas,), generator=g, device=err.device)
            v = torch.randint(0, K, (replicas,), generator=g, device=err.device)
            cur = ch[R, kslice]                                  # (R, L)
            new = cur.clone(); new[R, l] = v
            dE = (slice_E(new) - slice_E(cur)) / P
            up, dn = ch[R, (kslice + 1) % P, l], ch[R, (kslice - 1) % P, l]
            old_v = cur[R, l]
            dJ = Jp * (((v != up).float() + (v != dn).float()) -
                       ((old_v != up).float() + (old_v != dn).float()))
            acc = torch.rand(replicas, generator=g, device=err.device) < \
                torch.exp(-(dE + dJ).clamp(min=0) / T)
            ch[R[acc], kslice[acc]] = new[acc]
        Es = slice_E(ch.reshape(-1, L))
        m = Es.min().item()
        if m < best_E:
            best_E, best = m, ch.reshape(-1, L)[Es.argmin()].clone()
    return best, best_E


def exact_dp(err, par, budget, bins=2000) -> torch.Tensor:
    """Multiple-choice knapsack DP on a discretized budget (gap reference)."""
    L, K = err.shape
    unit = budget / bins
    w = torch.ceil(par / unit).long().tolist()
    e = err.tolist()
    INF = float('inf')
    dp = [0.0] + [INF] * bins          # dp[b] = min err using exactly <= b
    back = []
    for l in range(L):
        nd = [INF] * (bins + 1)
        bk = [-1] * (bins + 1)
        for b in range(bins + 1):
            if dp[b] == INF:
                continue
            for k in range(K):
                nb = b + w[l][k]
                if nb <= bins and dp[b] + e[l][k] < nd[nb]:
                    nd[nb], bk[nb] = dp[b] + e[l][k], (b, k)
        dp = nd
        back.append(bk)
    b = min(range(bins + 1), key=lambda i: dp[i])
    ch = [0] * L
    for l in range(L - 1, -1, -1):
        pb, k = back[l][b]
        ch[l] = k
        b = pb
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
