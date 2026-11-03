"""
QL7: cardinality-constrained QUBO solvers ("pick k of n with interactions")
==========================================================================
QL2 (anneal_alloc.py) showed that annealing is the wrong tool for a
*separable* objective: greedy was within 1% of the optimum and SQA 19-32%
away (F5). QL7 therefore only uses annealing where the objective has
pairwise interaction terms, and always runs it head to head with strong
classical baselines.

Problem (minimised), x in {0,1}^n with sum(x) == k:

    E(x) = h^T x + x^T Q x          (Q symmetric; diag(Q) acts like h)

Solvers
  * exact        -- enumerate all C(n,k) subsets (n <= ~22), the reference
  * greedy       -- add the item with the best marginal energy, k times
  * greedy_swap  -- greedy, then best-improvement 1-swap local search
                    (the strong classical baseline; usually optimal on easy
                    instances)
  * sa           -- vectorised simulated annealing with swap moves
  * sqa          -- path-integral simulated QUANTUM annealing: P Trotter
                    slices with swap moves, neighbouring slices coupled by
                    J_perp(Gamma) = -(T/2) ln tanh(Gamma / (P T))
                    (same schedule as QL2's sqa, binary instead of Potts)
  * random       -- uniform random subset (the "no selection" control)

Swap moves keep sum(x) == k exactly, so there is no penalty term to tune.
A local field u = Q x is kept per state so a move costs O(n), not O(n^2).
All of this is classical; "quantum" names the SQA sampling schedule only.
"""

import itertools
import math
import time
from typing import Dict, Optional

import torch


def _sym(Q: torch.Tensor) -> torch.Tensor:
    return 0.5 * (Q + Q.T)


def energy(x: torch.Tensor, h: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """E for x of shape (..., n) (bool or float)."""
    xf = x.to(h.dtype)
    return xf @ h + ((xf @ Q) * xf).sum(-1)


def exact(h, Q, k, chunk=65536):
    """Brute force over all C(n, k) subsets. Returns (x bool (n,), E)."""
    n = h.numel()
    Q = _sym(Q)
    best_E, best = float('inf'), None
    it = itertools.combinations(range(n), k)
    while True:
        rows = list(itertools.islice(it, chunk))
        if not rows:
            break
        idx = torch.tensor(rows, dtype=torch.long, device=h.device)
        x = torch.zeros(len(rows), n, dtype=h.dtype, device=h.device)
        x.scatter_(1, idx, 1.0)
        E = energy(x, h, Q)
        m = int(E.argmin())
        if float(E[m]) < best_E:
            best_E, best = float(E[m]), x[m] > 0
    if best is None:                       # k == 0
        best = torch.zeros(n, dtype=torch.bool, device=h.device)
        best_E = 0.0
    return best, best_E


def greedy(h, Q, k, init=None):
    """Add k items one at a time by best marginal energy
    dE_j = h_j + Q_jj + 2 sum_{i in S} Q_ij."""
    Q = _sym(Q)
    n = h.numel()
    x = torch.zeros(n, dtype=torch.bool, device=h.device)
    u = torch.zeros(n, dtype=h.dtype, device=h.device)        # Q x
    diag = torch.diagonal(Q)
    for _ in range(k):
        d = h + diag + 2 * u
        d = torch.where(x, torch.full_like(d, float('inf')), d)
        j = int(d.argmin())
        x[j] = True
        u = u + Q[:, j]
    return x, float(energy(x, h, Q))


def local_swap(x, h, Q, max_iter=10_000):
    """Best-improvement 1-swap descent (out i in S, in j not in S)."""
    Q = _sym(Q)
    x = x.clone()
    diag = torch.diagonal(Q)
    for _ in range(max_iter):
        u = Q @ x.to(h.dtype)
        rem = -(h + 2 * u - diag)                 # dE of removing i (i in S)
        add = h + diag + 2 * u                    # dE of adding j (before removal)
        # swap dE = rem_i + add_j - 2 Q_ij
        D = rem[:, None] + add[None, :] - 2 * Q
        mask = x[:, None] & (~x)[None, :]
        D = torch.where(mask, D, torch.full_like(D, float('inf')))
        m = int(D.argmin())
        i, j = divmod(m, x.numel())
        if not float(D[i, j]) < -1e-12:
            break
        x[i], x[j] = False, True
    return x, float(energy(x, h, Q))


def greedy_swap(h, Q, k):
    x, _ = greedy(h, Q, k)
    return local_swap(x, h, Q)


def _random_states(R, n, k, g, device):
    keys = torch.rand(R, n, generator=g, device=device)
    idx = keys.argsort(-1)[:, :k]
    x = torch.zeros(R, n, dtype=torch.bool, device=device)
    x.scatter_(1, idx, True)
    return x


def random_subset(h, Q, k, seed=0):
    g = torch.Generator(device=h.device).manual_seed(seed)
    x = _random_states(1, h.numel(), k, g, h.device)[0]
    return x, float(energy(x, h, _sym(Q)))


def _pick(mask, g):
    """Uniformly pick one True index per row of a (R, n) bool mask."""
    keys = torch.rand(mask.shape, generator=g, device=mask.device)
    keys = torch.where(mask, keys, torch.full_like(keys, -1.0))
    return keys.argmax(-1)


def _scale(h, Q):
    s = float(torch.cat([h.abs().flatten(), Q.abs().flatten()]).max())
    return s if s > 0 else 1.0


def sa(h, Q, k, replicas=64, sweeps=200, t0=None, t1=None, seed=0, init=None):
    """Metropolis SA with swap moves over `replicas` chains.
    One sweep = n proposals per chain."""
    Q = _sym(Q)
    n, dev = h.numel(), h.device
    if k == 0 or k == n:
        x = torch.full((n,), k == n, dtype=torch.bool, device=dev)
        return x, float(energy(x, h, Q))
    g = torch.Generator(device=dev).manual_seed(seed)
    s = _scale(h, Q)
    t0 = t0 or s
    t1 = t1 or s * 1e-3
    x = _random_states(replicas, n, k, g, dev)
    if init is not None:
        x[0] = init
    xf = x.to(h.dtype)
    u = xf @ Q                                    # (R, n)
    E = xf @ h + (u * xf).sum(-1)
    diag = torch.diagonal(Q)
    rr = torch.arange(replicas, device=dev)
    best_E, best = float(E.min()), x[int(E.argmin())].clone()
    for t in range(sweeps):
        T = t0 * (t1 / t0) ** (t / max(sweeps - 1, 1))
        for _ in range(n):
            i = _pick(x, g)
            j = _pick(~x, g)
            dE = (-(h[i] + 2 * u[rr, i] - diag[i]) + (h[j] + diag[j] + 2 * u[rr, j])
                  - 2 * Q[i, j])
            acc = torch.rand(replicas, generator=g, device=dev) < torch.exp(
                -dE.clamp(min=0) / T)
            ai = rr[acc]
            if ai.numel():
                x[ai, i[acc]] = False
                x[ai, j[acc]] = True
                u[ai] += Q[j[acc]] - Q[i[acc]]
                E[ai] += dE[acc]
        m = int(E.argmin())
        if float(E[m]) < best_E:
            best_E, best = float(E[m]), x[m].clone()
    return best, float(energy(best, h, Q))


def sqa(h, Q, k, trotter=16, replicas=16, sweeps=200, T=None, gamma0=None,
        gamma1=1e-3, seed=0, init=None):
    """Path-integral SQA with swap moves. State (R, P, n); slice energy E/P at
    temperature T, ferromagnetic coupling J_perp between neighbouring slices
    on every bit. Even/odd slices are updated alternately (checkerboard)."""
    Q = _sym(Q)
    n, dev, P = h.numel(), h.device, trotter
    if k == 0 or k == n:
        x = torch.full((n,), k == n, dtype=torch.bool, device=dev)
        return x, float(energy(x, h, Q))
    g = torch.Generator(device=dev).manual_seed(seed)
    s = _scale(h, Q)
    T = T or 0.05 * s / P
    gamma0 = gamma0 or 2.0 * s
    x = _random_states(replicas * P, n, k, g, dev).view(replicas, P, n)
    if init is not None:
        x[0, :] = init
    xf = x.to(h.dtype)
    u = xf @ Q                                    # (R, P, n)
    E = xf @ h + (u * xf).sum(-1)                 # (R, P)
    diag = torch.diagonal(Q)
    ar = torch.arange(replicas, device=dev)[:, None].expand(replicas, P)
    ap = torch.arange(P, device=dev)[None, :].expand(replicas, P)
    parity = (torch.arange(P, device=dev) % 2)[None, :]
    up_p, dn_p = (ap + 1) % P, (ap - 1) % P
    best_E, best = float('inf'), None
    for t in range(sweeps):
        gam = gamma0 * (gamma1 / gamma0) ** (t / max(sweeps - 1, 1))
        Jp = -0.5 * T * math.log(math.tanh(gam / (P * T)) + 1e-300)
        for _ in range(n):
            for half in (0, 1):
                flat = x.view(replicas * P, n)
                i = _pick(flat, g).view(replicas, P)
                j = _pick(~flat, g).view(replicas, P)
                dE = (-(h[i] + 2 * u[ar, ap, i] - diag[i])
                      + (h[j] + diag[j] + 2 * u[ar, ap, j]) - 2 * Q[i, j])
                # coupling: count disagreements with neighbouring slices.
                # bit i goes 1 -> 0, bit j goes 0 -> 1
                xi_u, xi_d = x[ar, up_p, i], x[ar, dn_p, i]
                xj_u, xj_d = x[ar, up_p, j], x[ar, dn_p, j]
                # disagreement change at i: before x=1, after x=0
                d_i = (xi_u.float() + xi_d.float()) - ((~xi_u).float() + (~xi_d).float())
                d_j = ((~xj_u).float() + (~xj_d).float()) - (xj_u.float() + xj_d.float())
                dJ = Jp * (d_i + d_j)
                tot = dE / P + dJ
                acc = (torch.rand(replicas, P, generator=g, device=dev) <
                       torch.exp(-tot.clamp(min=0) / T)) & (parity == half)
                if acc.any():
                    r_, p_ = ar[acc], ap[acc]
                    ii, jj = i[acc], j[acc]
                    x[r_, p_, ii] = False
                    x[r_, p_, jj] = True
                    u[r_, p_] += Q[jj] - Q[ii]
                    E[r_, p_] += dE[acc]
        m = int(E.argmin())
        if float(E.flatten()[m]) < best_E:
            best_E = float(E.flatten()[m])
            best = x.view(replicas * P, n)[m].clone()
    return best, float(energy(best, h, Q))


SOLVERS = {'exact': exact, 'greedy': greedy, 'greedy_swap': greedy_swap,
           'sa': sa, 'sqa': sqa, 'random': random_subset}


def solve(h, Q, k, method='greedy_swap', **kw) -> Dict:
    """Run one solver; returns the subset, its energy and the wall time
    (the solve time is charged to the run's GPU-hours by the trainer)."""
    h = h.double()
    Q = _sym(Q.double())
    k = int(max(0, min(k, h.numel())))
    t = time.perf_counter()
    x, E = SOLVERS[method](h, Q, k, **kw)
    return {'x': x, 'idx': torch.nonzero(x).flatten().tolist(), 'energy': E,
            'seconds': time.perf_counter() - t, 'method': method}


def interaction_strength(h, Q) -> float:
    """||offdiag(Q)||_F / (||h + diag(Q)||_2 + eps): 0 means separable, where
    greedy (top-k by h + diag Q) is exactly optimal."""
    Q = _sym(Q.double())
    d = torch.diagonal(Q)
    off = Q - torch.diag(d)
    return float(off.norm() / ((h.double() + d).norm() + 1e-30))
