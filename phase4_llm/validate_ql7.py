"""
QL7 validation (CPU, no downloads, runs in CI)
==============================================
  [1] QUBO energy / swap moves keep |S| = k; exact solver vs brute force
  [2] separable QUBO: greedy is exactly optimal (F5 sanity)
  [3] interacting QUBO: SA / SQA / greedy+swap reach the exact optimum;
      plain greedy does not always (so the head-to-head is meaningful)
  [4] GRPO advantages: zero-variance groups -> 0, group mean 0
  [5] GRPO loss: on-policy gradient = -adv * grad logp / tokens; clipping
  [6] GSM8K answer extraction and grading
  [7] Beta expected-signal formula vs Monte Carlo
  [8] PromptSelector: B distinct prompts, epoch coverage, posterior update
  [9] layer_interactions: finite-difference cross terms vs exact Hessian
      on a quadratic; LayerSelector + apply_layer_mask freeze blocks
 [10] toy GRPO end to end (uniform / QUBO prompts / QUBO layers) learns
 [11] gate: hours-to-target interpolation, ratio, non-inferiority, persistence
 [12] Amdahl budget: which phase fractions can reach 0.75x
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase4_llm import ql7_gate as G  # noqa: E402
from phase4_llm import ql7_qubo as Q  # noqa: E402
from phase4_llm import ql7_rl as R  # noqa: E402

PASS = []


def check(name, cond, info=''):
    PASS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}", flush=True)


def rand_qubo(n, scale, seed):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(n, generator=g, dtype=torch.float64)
    A = torch.randn(n, n, generator=g, dtype=torch.float64)
    return h, 0.5 * (A + A.T) * scale


def t1():
    h, Qm = rand_qubo(10, 0.5, 0)
    x = torch.zeros(10, dtype=torch.bool)
    x[[1, 4, 7]] = True
    e = float(Q.energy(x, h, Qm))
    xf = x.double()
    ok = abs(e - float(h @ xf + xf @ Qm @ xf)) < 1e-9
    best, E = Q.exact(h, Qm, 4)
    # brute force by hand
    import itertools
    bf = min(float(Q.energy(torch.tensor([i in c for i in range(10)]), h, Qm))
             for c in itertools.combinations(range(10), 4))
    r = [Q.solve(h, Qm, 4, m, **({'seed': 1, 'sweeps': 20} if m in ('sa', 'sqa') else {}))
         for m in ('greedy', 'greedy_swap', 'sa', 'sqa', 'random')]
    card = all(len(v['idx']) == 4 for v in r) and int(best.sum()) == 4
    check('1 QUBO energy, exact solver, cardinality', ok and abs(E - bf) < 1e-9 and card,
          f'E*={E:.4f}')


def t2():
    ok = True
    for s in range(5):
        h, _ = rand_qubo(14, 0, s)
        Qm = torch.diag(torch.randn(14, dtype=torch.float64, generator=torch.Generator().manual_seed(s)))
        ok &= abs(Q.solve(h, Qm, 5, 'greedy')['energy'] - Q.solve(h, Qm, 5, 'exact')['energy']) < 1e-9
        ok &= Q.interaction_strength(h, Qm) < 1e-12
    check('2 separable QUBO: greedy == exact, interaction 0', ok)


def t3():
    hits = {m: 0 for m in ('greedy', 'greedy_swap', 'sa', 'sqa')}
    N = 8
    for s in range(N):
        h, Qm = rand_qubo(14, 0.7, 100 + s)
        E = Q.solve(h, Qm, 5, 'exact')['energy']
        for m in hits:
            kw = {'seed': s, 'sweeps': 60} if m in ('sa', 'sqa') else {}
            hits[m] += abs(Q.solve(h, Qm, 5, m, **kw)['energy'] - E) < 1e-9
    ok = hits['sa'] >= N - 1 and hits['sqa'] >= N - 1 and hits['greedy'] < N
    check('3 interacting QUBO: SA/SQA reach optimum, greedy not always', ok, str(hits))


def t4():
    r = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0], [1, 0, 1, 0]], dtype=torch.float)
    a = R.group_advantages(r)
    z = R.zero_signal(r)
    ok = torch.all(a[:2] == 0) and abs(float(a[2].mean())) < 1e-6 and z.tolist() == [True, True, False]
    ok &= abs(float(a[2, 0]) - 1.0) < 1e-3          # (1 - .5) / .5
    check('4 GRPO advantages (zero-variance groups -> 0)', ok)


def t5():
    torch.manual_seed(0)
    w = torch.randn(3, 5, requires_grad=True)
    logp = torch.log_softmax(w, -1)[:, :4]               # (N=3, T=4)
    adv = torch.tensor([1.0, -0.5, 0.0])
    mask = torch.ones(3, 4)
    loss, _ = R.grpo_loss(logp, logp.detach(), adv, mask)
    g1, = torch.autograd.grad(loss, w, retain_graph=True)
    ref = -(adv[:, None] * logp).sum() / mask.sum()
    g2, = torch.autograd.grad(ref, w)
    ok = torch.allclose(g1, g2, atol=1e-6)
    # clipping: ratio far above 1+eps with positive adv -> zero gradient
    lp = torch.zeros(1, 1, requires_grad=True)
    l2, st = R.grpo_loss(lp + 1.0, torch.zeros(1, 1), torch.tensor([1.0]), torch.ones(1, 1))
    g3, = torch.autograd.grad(l2, lp)
    ok &= abs(float(g3)) < 1e-9 and st['clip_frac'] == 1.0
    check('5 GRPO loss gradient + clipping', ok)


def t6():
    cases = [('so 3+4=7.\n#### 7', 7), ('The answer is $1,234.', 1234), ('\\boxed{-2.5}', -2.5),
             ('first 3 then 18', 18), ('no number', None)]
    ok = all((R.extract_answer(t) == v) for t, v in cases)
    ok &= R.math_reward('... #### 72', R.gsm8k_gold('blah\n#### 72')) == 1.0
    ok &= R.math_reward('... #### 71', 72.0) == 0.0
    check('6 GSM8K answer extraction / grading', ok)


def t7():
    g = torch.Generator().manual_seed(0)
    ok = True
    for a, b, Gs in ((1, 1, 8), (2, 5, 8), (7, 2, 4)):
        p = torch.distributions.Beta(float(a), float(b)).sample((200_000,))
        mc = float((1 - p ** Gs - (1 - p) ** Gs).mean())
        ok &= abs(mc - float(R.expected_signal(a, b, Gs))) < 5e-3
    check('7 Beta expected signal = Monte Carlo', ok)


def t8():
    texts = [f'question {i} about {i % 5} apples' for i in range(40)]
    ok = True
    for m in ('uniform', 'greedy_swap', 'sqa'):
        ps = R.PromptSelector(texts, 4, method=m, seed=0,
                              solver_kw={'sweeps': 5, 'replicas': 2, 'trotter': 4} if m == 'sqa' else {})
        seen = set()
        for _ in range(10):
            b = ps.next_batch(8)['idx']
            ok &= len(set(b)) == 8
            seen |= set(b)
            ps.update(b, torch.ones(8, 4))
        ok &= len(seen) >= (40 if m == 'uniform' else 30)
    ps = R.PromptSelector(texts, 4, method='greedy', seed=0)
    ps.update([0, 1], torch.tensor([[1., 1, 1, 1], [1., 0, 1, 0]]))
    s = R.expected_signal(ps.a[[0, 1]], ps.b[[0, 1]], 4)
    ok &= float(s[1]) > float(s[0])                # all-correct prompt has less signal
    check('8 PromptSelector: distinct prompts, coverage, posterior', ok)


def t9():
    # quadratic loss 0.5 th^T H th with 3 "blocks": cross terms must equal g_i^T H_ij g_j
    torch.manual_seed(0)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 1, bias=False) for _ in range(3)])

    m = M().double()
    A = torch.randn(6, 6, dtype=torch.float64)
    H = A @ A.T + torch.eye(6, dtype=torch.float64)
    blocks, _ = R.block_params(m)

    def lf():
        th = torch.cat([b[0].flatten() for b in blocks])
        return 0.5 * th @ H @ th
    g2, Mx = R.layer_interactions(m, lf, blocks, eps=1e-4)
    th = torch.cat([b[0].detach().flatten() for b in blocks])
    g = H @ th
    ref = torch.zeros(3, 3, dtype=torch.float64)
    for i in range(3):
        for j in range(3):
            ref[i, j] = g[2 * i:2 * i + 2] @ H[2 * i:2 * i + 2, 2 * j:2 * j + 2] @ g[2 * j:2 * j + 2]
    ok = torch.allclose(Mx, ref, rtol=1e-4, atol=1e-6)
    ok &= torch.allclose(g2, torch.stack([g[2 * i:2 * i + 2].pow(2).sum() for i in range(3)]))
    ls = R.LayerSelector(3, 1, method='greedy_swap')
    ls.set_probe(g2, Mx)
    c = ls.choose()['idx']
    R.apply_layer_mask(blocks, c)
    ok &= len(c) == 1 and all(p.requires_grad == (i in c) for i, b in enumerate(blocks) for p in b)
    check('9 Hessian cross terms (finite diff) + layer freezing', ok)


def t10():
    res = {}
    for m, l in (('uniform', 'none'), ('greedy_swap', 'none'), ('uniform', 'greedy_swap')):
        res[(m, l)] = R.toy_train(m, l, steps=40, seed=0, probe_every=20)
    ok = all(v['final_expected_reward'] > 1.0 / 6 + 0.1 for v in res.values())
    ok &= res[('uniform', 'greedy_swap')]['frozen_ok']
    info = {f'{m}/{l}': round(v['final_expected_reward'], 3) for (m, l), v in res.items()}
    check('10 toy GRPO learns (uniform, QUBO prompts, QUBO layers)', ok, str(info))


def t11():
    def run(curve, final, correct):
        return {'curve': [{'gpu_hours': h, 'score': s, 'tag': 'curve'} for h, s in curve],
                'final': {'score': final, 'correct': correct, 'gpu_hours': curve[-1][0]},
                'log': [{'zero_groups': 1, 'groups': 4, 'select_s': 0.0}]}
    corr = [1] * 50 + [0] * 50
    base = [run([(0, 0), (1, 20), (2, 40), (3, 50), (4, 50)], 50, corr) for _ in range(2)]
    fast = [run([(0, 0), (0.5, 20), (1, 40), (1.5, 50), (2, 50)], 50, corr) for _ in range(2)]
    slow = [run([(0, 0), (1, 10), (2, 20), (3, 30), (4, 35)], 35, [1] * 35 + [0] * 65)
            for _ in range(2)]
    g = G.gate({'base': base, 'fast': fast, 'slow': slow}, delta=0.5)
    ok = g['arms']['fast']['pass'] and not g['arms']['slow']['pass'] and not g['arms']['base']['pass']
    ok &= abs(g['arms']['base']['ratio'] - 1.0) < 1e-9 and g['arms']['fast']['ratio'] < 0.6
    ok &= math.isinf(g['arms']['slow']['ratio'])
    # persistence: prompt 0 zero-signal twice, prompt 1 zero then mixed, prompt 2 mixed then zero
    log = [{'prompt_idx': [0, 1, 2], 'prompt_succ': [8, 0, 3]},
           {'prompt_idx': [0, 1, 2], 'prompt_succ': [8, 4, 0]}]
    ps = G.persistence([{'log': log}], 8)
    ok &= ps['revisits'] == 3 and abs(ps['p_zero_given_zero'] - 0.5) < 1e-12 and \
        abs(ps['false_skip'] - 1 / 3) < 1e-12
    check('11 gate: hours-to-target, ratio, non-inferiority', ok,
          f"fast={g['arms']['fast']['ratio']:.3f}")


def t12():
    # skipping zero-signal groups saves generation+training in proportion
    r1 = R.amdahl_ratio(0.6, 0.35, gen_saving=0.3, train_saving=0.3)
    r2 = R.amdahl_ratio(0.6, 0.35, train_saving=0.4)          # layers only: train phase
    ok = abs(r1 - (0.6 * 0.7 + 0.35 * 0.7 + 0.05)) < 1e-12 and r1 < 0.75 and r2 > 0.75
    check('12 Amdahl: prompt lever can reach 0.75x, layer lever alone cannot', ok,
          f'prompts-30%={r1:.3f} layers-40%={r2:.3f}')


if __name__ == '__main__':
    for t in (t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12):
        t()
    print(f'QL7 validation: {sum(PASS)}/{len(PASS)} passed')
    sys.exit(0 if all(PASS) else 1)
