"""
QL7: GRPO building blocks + the two QUBO-driven savings
=======================================================
Target (docs/FERMIQLLM_SURPASS_PLAN.md): reach the same final score with
<= 0.75x the GPU-hours of a named GRPO baseline, otherwise publish a
negative result. FermiQLLM claims "-25% RL cost" with no baseline.

Where GRPO spends GPU time (per step): rollout generation (vLLM), the
policy forward/backward on the rollouts, and small overheads. A saving s_g
on generation and s_t on training gives (Amdahl, see `amdahl_ratio`)

    ratio = f_gen (1 - s_g) + f_train (1 - s_t) + f_other   (x steps ratio)

so the lever has to hit the big phase. QL7 tests two levers, each with an
interacting QUBO solved by SQA *and* by greedy / greedy+swap / random:

1. Prompt selection (hits generation AND training). In GRPO a group whose
   G completions all get the same reward has advantage 0 and contributes
   no gradient, yet it paid for G rollouts. Each prompt keeps a Beta
   posterior over its pass rate p; the probability its group carries
   signal is  E[1 - p^G - (1-p)^G]  (closed form under the Beta, see
   `expected_signal`). From a candidate pool we choose B prompts by

       min  -sum_i s_i x_i + lam * sum_{i<j} sim_ij x_i x_j,  sum x = B

   where sim is the cosine similarity of cheap hashed n-gram features of
   the prompts (redundant prompts give correlated gradients). Named
   baselines: uniform sampling (standard GRPO) and DAPO-style dynamic
   sampling (generate, then drop zero-signal groups).

2. Layer-block selection (hits training only). Each step only k of L
   transformer blocks get gradients. With per-block gradients g_i and
   Hessian cross terms M_ij = g_i^T H_ij g_j (measured on probe steps by
   finite differences, `layer_interactions`), the second-order loss change
   of an SGD step on the set S is

       dL(S) = -eta sum_{i in S} |g_i|^2 + eta^2/2 sum_{i,j in S} M_ij

   which is a QUBO with genuine interactions. If the measured interaction
   strength is small, greedy is optimal and SQA can only lose (QL2's F5).

`GpuHourMeter` charges every phase, *including the QUBO solve*, to the run.
"""

import math
import re
import time
import zlib
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence

import torch

from phase4_llm import ql7_qubo

# --------------------------------------------------------------- rewards

_NUM = re.compile(r'-?\d[\d,]*\.?\d*')


def _to_float(s: str) -> Optional[float]:
    s = s.replace(',', '').replace('$', '').strip().rstrip('.')
    try:
        return float(s)
    except ValueError:
        return None


def extract_answer(text: str) -> Optional[float]:
    """GSM8K-style final answer: '#### x', then '\\boxed{x}', then
    'answer is x', then the last number in the text."""
    for pat in (r'####\s*([^\n]+)', r'\\boxed\{([^}]*)\}', r'answer is[:\s]*\$?([^\s\n]+)'):
        m = re.findall(pat, text)
        if m:
            nums = _NUM.findall(m[-1])
            if nums:
                v = _to_float(nums[0])
                if v is not None:
                    return v
    nums = _NUM.findall(text)
    return _to_float(nums[-1]) if nums else None


def gsm8k_gold(answer_field: str) -> float:
    return _to_float(answer_field.split('####')[-1])


def math_reward(completion: str, gold: float, tol=1e-6) -> float:
    v = extract_answer(completion)
    return float(v is not None and gold is not None and abs(v - gold) <= tol * max(1, abs(gold)))


# ------------------------------------------------------------- GRPO math

def group_advantages(rewards: torch.Tensor, scale_std=True, eps=1e-4) -> torch.Tensor:
    """rewards (B, G) -> advantages (B, G): r - mean_group, optionally / std
    (std=False is Dr. GRPO). Zero-variance groups get exactly 0."""
    r = rewards.float()
    a = r - r.mean(1, keepdim=True)
    if scale_std:
        a = a / (r.std(1, keepdim=True, unbiased=False) + eps)
    return a


def zero_signal(rewards: torch.Tensor) -> torch.Tensor:
    """(B, G) -> bool (B,): all completions in the group got the same reward."""
    return (rewards.max(1).values - rewards.min(1).values) <= 1e-9


def grpo_loss(logp, old_logp, adv, mask, ref_logp=None, clip=0.2, clip_high=None,
              beta=0.0, loss_type='dapo'):
    """Clipped policy-gradient loss on completion tokens.
    logp/old_logp/ref_logp/mask: (N, T); adv: (N,).
    loss_type 'dapo' = mean over all tokens in the batch; 'grpo' = mean per
    sequence, then over sequences. Returns (loss, stats)."""
    mask = mask.float()
    ratio = torch.exp(logp - old_logp)
    hi = clip if clip_high is None else clip_high
    a = adv[:, None]
    pg = -torch.minimum(ratio * a, ratio.clamp(1 - clip, 1 + hi) * a)
    if beta and ref_logp is not None:                  # k3 estimator
        d = ref_logp - logp
        pg = pg + beta * (torch.exp(d) - d - 1)
    if loss_type == 'dapo':
        loss = (pg * mask).sum() / mask.sum().clamp(min=1)
    else:
        loss = ((pg * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()
    with torch.no_grad():
        clipped = ((ratio - 1).abs() > clip).float()
        frac = float((clipped * mask).sum() / mask.sum().clamp(min=1))
    return loss, {'clip_frac': frac}


# ------------------------------------------------- 1. prompt selection

def beta_moment(a, b, G: int):
    """E[p^G] for p ~ Beta(a, b) = prod_{t<G} (a+t)/(a+b+t)."""
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    out = torch.ones_like(a)
    for t in range(G):
        out = out * (a + t) / (a + b + t)
    return out


def expected_signal(a, b, G: int):
    """P(group of G Bernoulli(p) rewards is not all-equal), p ~ Beta(a, b)."""
    return 1 - beta_moment(a, b, G) - beta_moment(b, a, G)


def hashed_features(texts: Sequence[str], dim=2048, n=3) -> torch.Tensor:
    """Unit-norm hashed word n-gram counts (1..n), cheap and deterministic."""
    X = torch.zeros(len(texts), dim, dtype=torch.float64)
    for r, t in enumerate(texts):
        w = re.findall(r'[a-z]+|\d', t.lower())
        for m in range(1, n + 1):
            for i in range(len(w) - m + 1):
                X[r, zlib.crc32(' '.join(w[i:i + m]).encode()) % dim] += 1
    return X / X.norm(dim=1, keepdim=True).clamp(min=1e-12)


class PromptSelector:
    """Beta-posterior pass rates + QUBO choice of B prompts per step.

    method: 'uniform'  -- standard GRPO (random prompts, no posterior used)
            'greedy' | 'greedy_swap' | 'sa' | 'sqa' | 'random' -- QUBO solver
            ('random' = random subset of the candidate pool: pool control)
    Each epoch every prompt is offered as a candidate once; a candidate not
    chosen is skipped until the next epoch (return_unchosen=True puts it
    back instead, which only reorders the epoch and saves nothing). The
    posterior decays toward the prior (`forget`) each time the prompt is
    used so pass-rate estimates track the improving policy.
    """

    def __init__(self, texts: Sequence[str], G: int, method='uniform', pool_factor=4,
                 lam=0.5, prior=(1.0, 1.0), forget=0.7, explore=0.1, seed=0, decay=1.0,
                 solver_kw=None, feat_dim=2048, return_unchosen=False):
        self.n, self.G, self.method = len(texts), G, method
        self.return_unchosen = return_unchosen
        self.decay = decay
        self.pool_factor, self.lam = pool_factor, lam
        self.a0, self.b0 = prior
        self.a = torch.full((self.n,), float(prior[0]), dtype=torch.float64)
        self.b = torch.full((self.n,), float(prior[1]), dtype=torch.float64)
        self.seen = torch.zeros(self.n, dtype=torch.long)
        self.forget, self.explore = forget, explore
        self.g = torch.Generator().manual_seed(seed)
        self.solver_kw = solver_kw or {}
        self.feats = hashed_features(texts, feat_dim) if method not in ('uniform',) else None
        self._epoch: List[int] = []
        self.history: List[Dict] = []

    def _refill(self):
        self._epoch = torch.randperm(self.n, generator=self.g).tolist()

    def _take(self, m):
        """m distinct prompts from the current epoch (refilling across the
        epoch boundary without repeating a prompt inside one batch)."""
        out, seen = [], set()
        while len(out) < m:
            if not self._epoch:
                self._refill()
                self._epoch = [i for i in self._epoch if i not in seen]
            i = self._epoch.pop()
            out.append(i)
            seen.add(i)
        return out

    def next_batch(self, B: int) -> Dict:
        t = time.perf_counter()
        # time decay: every estimate drifts back to the prior, so a prompt
        # skipped as "no signal" is retried once its estimate goes stale
        self.a = self.a0 + (self.a - self.a0) * self.decay
        self.b = self.b0 + (self.b - self.b0) * self.decay
        if self.method == 'uniform':
            idx = self._take(B)
            info = {'idx': idx, 'solve_s': time.perf_counter() - t, 'energy': None}
            self.history.append(info)
            return info
        M = min(self.n, B * self.pool_factor)
        pool = self._take(M)
        pool_t = torch.tensor(pool)
        n_exp = int(round(self.explore * B))
        # exploration: a few uniformly random prompts from the pool
        perm = torch.randperm(M, generator=self.g)
        exp_local = perm[:n_exp]
        rest = perm[n_exp:]
        s = expected_signal(self.a[pool_t[rest]], self.b[pool_t[rest]], self.G)
        F = self.feats[pool_t[rest]]
        S = F @ F.T
        S.fill_diagonal_(0)
        h = -s
        Q = 0.5 * self.lam * S                     # x^T Q x counts each pair twice
        r = ql7_qubo.solve(h, Q, B - n_exp, self.method,
                           **({'seed': len(self.history)} | self.solver_kw
                              if self.method in ('sa', 'sqa', 'random') else {}))
        chosen_local = rest[torch.tensor(r['idx'], dtype=torch.long)].tolist() + exp_local.tolist()
        idx = [pool[i] for i in chosen_local]
        if self.return_unchosen:
            back = set(pool) - set(idx) - set(self._epoch)
            self._epoch.extend(sorted(back))
            p = torch.randperm(len(self._epoch), generator=self.g).tolist()
            self._epoch = [self._epoch[i] for i in p]
        # default: unchosen candidates wait for the next epoch -- that is what
        # saves rollouts (a prompt predicted to carry no signal is skipped)
        info = {'idx': idx, 'solve_s': time.perf_counter() - t, 'energy': r['energy'],
                'pred_signal': float(s[torch.tensor(r['idx'], dtype=torch.long)].mean())
                if r['idx'] else None,
                'interaction': ql7_qubo.interaction_strength(h, Q)}
        self.history.append(info)
        return info

    def update(self, idx: Sequence[int], rewards: torch.Tensor):
        """rewards (B, G) in [0, 1] for prompts idx (binary or fractional)."""
        r = rewards.double()
        for row, i in enumerate(idx):
            succ = float(r[row].sum())
            self.a[i] = self.forget * self.a[i] + (1 - self.forget) * self.a0 + succ
            self.b[i] = self.forget * self.b[i] + (1 - self.forget) * self.b0 + (r.shape[1] - succ)
            self.seen[i] += 1


# -------------------------------------------- 2. layer-block selection

def block_params(model, block_names: Optional[List[str]] = None):
    """Group trainable params into transformer blocks ('...layers.<i>.').
    Params outside the blocks (embeddings, final norm, head) form no block
    and are always trained (or always frozen, by the caller's choice)."""
    groups = defaultdict(list)
    rest = []
    for n, p in model.named_parameters():
        m = re.search(r'(?:^|\.)(?:layers|h|blocks)\.(\d+)\.', n)
        if m:
            groups[int(m.group(1))].append(p)
        else:
            rest.append(p)
    return [groups[i] for i in sorted(groups)], rest


def _gvec(ps):
    return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in ps]


def _dot(a, b):
    return sum(float((x.double() * y.double()).sum()) for x, y in zip(a, b))


def layer_interactions(model, loss_fn, blocks, eps=1e-3):
    """Per-block |g_i|^2 and the Hessian cross matrix M_ij = g_i^T H_ij g_j by
    finite differences of the gradient: one extra backward per block,
    H[:, j] g_j ~= (grad L(theta + e g_j) - grad L(theta)) / e.
    `loss_fn()` must compute the loss on a FIXED probe batch (deterministic).
    Returns (g2 (L,), M (L, L)) as float64 tensors."""
    L = len(blocks)
    model.zero_grad(set_to_none=True)
    loss_fn().backward()
    g = [_gvec(b) for b in blocks]
    g2 = torch.tensor([_dot(gi, gi) for gi in g], dtype=torch.float64)
    M = torch.zeros(L, L, dtype=torch.float64)
    for j in range(L):
        nj = math.sqrt(max(float(g2[j]), 1e-30))
        e = eps / nj                              # step of length eps along g_j
        with torch.no_grad():
            for p, d in zip(blocks[j], g[j]):
                p.add_(d, alpha=e)
        model.zero_grad(set_to_none=True)
        loss_fn().backward()
        gj = [_gvec(b) for b in blocks]
        with torch.no_grad():
            for p, d in zip(blocks[j], g[j]):
                p.sub_(d, alpha=e)
        for i in range(L):
            M[i, j] = _dot([x - y for x, y in zip(gj[i], g[i])], g[i]) / e
    model.zero_grad(set_to_none=True)
    return g2, 0.5 * (M + M.T)


class LayerSelector:
    """Choose k of L blocks per step.
    method: 'none' (all blocks), 'topk' (largest EMA |g|^2, separable),
            'random', or a QUBO solver ('greedy', 'greedy_swap', 'sa', 'sqa')
            on dL(S) with the measured cross terms.
    Gradient-norm EMAs of unselected blocks go stale, so a UCB-style
    staleness bonus keeps every block in rotation."""

    def __init__(self, n_blocks, k, method='none', eta=1.0, stale_bonus=0.05, ema=0.7,
                 seed=0, solver_kw=None):
        self.L, self.k, self.method = n_blocks, k, method
        self.eta, self.stale_bonus, self.ema = eta, stale_bonus, ema
        self.g2 = torch.ones(n_blocks, dtype=torch.float64)
        self.M = torch.zeros(n_blocks, n_blocks, dtype=torch.float64)
        self.age = torch.zeros(n_blocks, dtype=torch.float64)
        self.gen = torch.Generator().manual_seed(seed)
        self.solver_kw = solver_kw or {}
        self.step = 0
        self.last_interaction = None

    def set_probe(self, g2, M):
        self.g2 = g2.clone()
        self.M = M.clone()
        self.age.zero_()

    def observe(self, idx, g2_obs):
        for i, v in zip(idx, g2_obs):
            self.g2[i] = self.ema * self.g2[i] + (1 - self.ema) * float(v)

    def choose(self) -> Dict:
        t = time.perf_counter()
        self.step += 1
        if self.method == 'none' or self.k >= self.L:
            idx = list(range(self.L))
        elif self.method == 'random':
            idx = sorted(torch.randperm(self.L, generator=self.gen)[:self.k].tolist())
        else:
            scale = float(self.g2.max()) or 1.0
            bonus = self.stale_bonus * scale * torch.sqrt(self.age)
            h = -self.eta * self.g2 - bonus
            if self.method == 'topk':
                idx = sorted(torch.topk(-h, self.k).indices.tolist())
            else:
                Q = 0.5 * self.eta ** 2 * self.M
                r = ql7_qubo.solve(h, Q, self.k, self.method,
                                   **({'seed': self.step} | self.solver_kw
                                      if self.method in ('sa', 'sqa') else {}))
                idx = r['idx']
                self.last_interaction = ql7_qubo.interaction_strength(h, Q)
        self.age += 1
        self.age[idx] = 0
        return {'idx': idx, 'solve_s': time.perf_counter() - t}


def apply_layer_mask(blocks, idx):
    """requires_grad only on the chosen blocks. Frozen blocks skip their
    weight-gradient GEMMs, and autograd stops below the lowest chosen block.
    AdamW skips params whose .grad is None, so frozen blocks also skip the
    optimizer update (their moments simply pause)."""
    keep = set(idx)
    for i, b in enumerate(blocks):
        for p in b:
            p.requires_grad_(i in keep)
            if i not in keep:
                p.grad = None


# --------------------------------------------------------- accounting

class GpuHourMeter:
    """Wall-clock per phase x number of GPUs. CUDA is synchronised at phase
    boundaries so asynchronous kernels are charged to the right phase."""

    def __init__(self, n_gpus=1, sync=None):
        self.n_gpus = n_gpus
        self.sec = defaultdict(float)
        self.sync = sync if sync is not None else torch.cuda.is_available()

    @contextmanager
    def phase(self, name):
        if self.sync:
            torch.cuda.synchronize()
        t = time.perf_counter()
        try:
            yield
        finally:
            if self.sync:
                torch.cuda.synchronize()
            self.sec[name] += time.perf_counter() - t

    def gpu_hours(self, exclude=('eval',)):
        return self.n_gpus * sum(v for k, v in self.sec.items() if k not in exclude) / 3600

    def fractions(self, exclude=('eval',)):
        tot = sum(v for k, v in self.sec.items() if k not in exclude) or 1.0
        return {k: v / tot for k, v in self.sec.items() if k not in exclude}


def amdahl_ratio(f_gen, f_train, gen_saving=0.0, train_saving=0.0, steps_ratio=1.0,
                 overhead=0.0):
    """Predicted GPU-hour ratio vs baseline. f_* are the baseline's phase
    fractions (the rest, 1 - f_gen - f_train, is unchanged). `overhead` is
    added time as a fraction of the baseline step (e.g. the QUBO solve)."""
    f_other = 1.0 - f_gen - f_train
    return steps_ratio * (f_gen * (1 - gen_saving) + f_train * (1 - train_saving)
                          + f_other + overhead)


# --------------------------------------- toy environment (CPU validation)

class ToyPolicy(torch.nn.Module):
    """Residual-MLP policy for a one-token answer task: prompt id -> answer.
    Exposes `.layers.<i>.` blocks so block_params/LayerSelector apply."""

    def __init__(self, n_prompts, n_answers, d=32, n_layers=4, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.nn.Embedding(n_prompts, d)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d, d), torch.nn.GELU())
             for _ in range(n_layers)])
        self.head = torch.nn.Linear(d, n_answers)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * (0.3 if p.dim() > 1 else 0.0))
            self.head.weight.mul_(0.1)            # start near-uniform over answers

    def forward(self, ids):
        x = self.emb(ids)
        for blk in self.layers:
            x = x + blk(x)
        return torch.log_softmax(self.head(x), -1)


def toy_task(n_prompts=64, n_answers=6, seed=0):
    """Random targets; the random init makes some prompts easy (quickly
    all-correct, zero signal) and some hard (all-wrong, zero signal)."""
    g = torch.Generator().manual_seed(seed)
    target = torch.randint(0, n_answers, (n_prompts,), generator=g)
    texts = [f'item {i} group {i % 7} kind {i % 3}' for i in range(n_prompts)]
    return target, texts


def toy_train(method='uniform', layer_method='none', steps=60, B=8, G=8, k_layers=2,
              n_prompts=64, n_answers=6, lr=0.02, seed=0, probe_every=0):
    """GRPO on the toy task; returns dict with reward curve and counts of
    zero-signal groups. One policy-gradient step per batch (on-policy)."""
    torch.manual_seed(seed)
    target, texts = toy_task(n_prompts, n_answers, seed)
    pol = ToyPolicy(n_prompts, n_answers, n_layers=4, seed=seed)
    blocks, rest = block_params(pol)
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=0.0)
    ps = PromptSelector(texts, G, method=method, seed=seed, pool_factor=4,
                        solver_kw={'sweeps': 20, 'replicas': 4, 'trotter': 8}
                        if method == 'sqa' else {'sweeps': 40} if method == 'sa' else {})
    ls = LayerSelector(len(blocks), k_layers, method=layer_method, seed=seed)
    gen = torch.Generator().manual_seed(seed + 1)
    rewards_log, zero_log, frozen_ok = [], [], True
    all_ids = torch.arange(n_prompts)
    for step in range(steps):
        if probe_every and layer_method not in ('none',) and step % probe_every == 0:
            for b in blocks:
                for p in b:
                    p.requires_grad_(True)
            probe = all_ids[:16]

            def lf():
                lp = pol(probe)
                return -(lp[torch.arange(len(probe)), target[probe]]).mean()
            g2, M = layer_interactions(pol, lf, blocks)
            ls.set_probe(g2, M)
        sel = ps.next_batch(B)
        idx = torch.tensor(sel['idx'])
        with torch.no_grad():
            lp = pol(idx)
            acts = torch.multinomial(lp.exp(), G, replacement=True, generator=gen)  # (B, G)
        r = (acts == target[idx][:, None]).float()
        ps.update(sel['idx'], r)
        zero_log.append(int(zero_signal(r).sum()))
        rewards_log.append(float(r.mean()))
        adv = group_advantages(r)
        lc = ls.choose()
        apply_layer_mask(blocks, lc['idx'])
        lp = pol(idx)
        logp = lp.gather(1, acts)                                   # (B, G)
        loss, _ = grpo_loss(logp.reshape(-1, 1), logp.detach().reshape(-1, 1),
                            adv.reshape(-1), torch.ones(B * G, 1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if layer_method != 'none':
            frozen = set(range(len(blocks))) - set(lc['idx'])
            frozen_ok &= all(p.grad is None for i in frozen for p in blocks[i])
            ls.observe(lc['idx'], [sum(float((p.grad ** 2).sum()) for p in blocks[i]
                                       if p.grad is not None) for i in lc['idx']])
        opt.step()
    with torch.no_grad():
        acc = float((pol(all_ids).argmax(-1) == target).float().mean())
        exp_r = float(pol(all_ids).exp()[all_ids, target].mean())
    return {'reward': rewards_log, 'zero_groups': zero_log, 'final_greedy_acc': acc,
            'final_expected_reward': exp_r, 'frozen_ok': frozen_ok,
            'rollouts': steps * B * G}
