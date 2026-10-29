"""
QL5 validation suite (CPU, no downloads) -- run from repo root:
    python phase4_llm/validate_ql5.py

Checks
  1. G1 re-gauging keeps A B and balances the two sites
  2. D1 weighted embedding SVD: exact at full rank, error = weighted tail,
     beats plain SVD in the weighted norm
  3. LowRankEmbedding / tied LowRankHead match the dense layers
  4. chunked KD loss == full-logit KD loss (value and gradient)
  5. Fisher importance matches a direct (grad * W)^2 computation
  6. Fisher-weighted allocation gives important layers more rank
  7. INT8 / FP8 fake-quant error bounds
  8. save_compressed / load_compressed roundtrip with embedding + tied head
  9. 13-gram decontamination drops overlapping documents only
 10. mixture packer: window length and source proportions
 11. paired bootstrap + gate logic; lm-eval per-sample parsing
 12. end-to-end ql5_compress.py --smoke hits the 20 % saving target
"""

import json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import torch
import torch.nn.functional as F

from phase4_llm.tn_layers import (LowRankLinear, LowRankEmbedding, LowRankHead,
                                  regauge_, weighted_embedding_svd, fake_quant_,
                                  save_compressed, load_compressed)
from phase4_llm.anneal_alloc import allocate
from phase4_llm.ql5_data import Decontam, MixturePacker, parse_mix
from phase4_llm.ql5_gate import bootstrap_diff, gate
from phase4_llm.eval.run_eval import per_sample
from phase4_llm import ql5_compress as q5

PASS = True


def check(name, cond, detail=""):
    global PASS
    PASS &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def section(t):
    print("=" * 70); print(t); print("=" * 70)


torch.manual_seed(0)

section("[1] G1 gauge fixing")
A = torch.randn(48, 8) * torch.logspace(-2, 2, 8)          # badly scaled gauge
B = torch.randn(8, 32) / torch.logspace(-2, 2, 8)[:, None]
W0 = A @ B
s = regauge_(A, B)
check("A B unchanged", torch.allclose(A @ B, W0, rtol=1e-4, atol=1e-5),
      f"max err {(A @ B - W0).abs().max():.2e}")
check("balanced: A^T A == B B^T == diag(s)",
      torch.allclose(A.T @ A, torch.diag(s).float(), atol=1e-3 * s.max().item()) and
      torch.allclose(B @ B.T, torch.diag(s).float(), atol=1e-3 * s.max().item()))
lr = LowRankLinear(A.clone(), B.clone())
x = torch.randn(5, 32)
y0 = lr(x); lr.regauge()
check("LowRankLinear.regauge is idempotent on the output",
      torch.allclose(lr(x), y0, atol=1e-4))

section("[2] D1 weighted embedding SVD")
V, d = 300, 24
E = torch.randn(V, d) @ torch.randn(d, d) / 5
freq = torch.zeros(V); freq[:40] = torch.rand(40) * 1000 + 100; freq[40:] = torch.rand(V - 40)
X = torch.randn(500, d) @ torch.randn(d, d)
G = X.T @ X
Af, Sig, Bf = weighted_embedding_svd(E, freq, G)
check("full rank reconstructs E", torch.allclose(Af @ Bf, E.double(), atol=1e-4),
      f"max err {(Af @ Bf - E.double()).abs().max():.2e}")
Dh = ((freq / freq.sum()) + 1e-6).sqrt().double()
S = torch.linalg.cholesky(G.double() + 1e-4 * G.diagonal().mean() * torch.eye(d))
werr = lambda Er: torch.linalg.norm(Dh[:, None] * (E.double() - Er) @ S).item()
r = 6
e_w = werr(Af[:, :r] @ Bf[:r])
check("weighted error == singular-value tail",
      abs(e_w - Sig[r:].pow(2).sum().sqrt().item()) < 1e-6 * max(e_w, 1), f"{e_w:.4f}")
U, s0, Vh = torch.linalg.svd(E.double(), full_matrices=False)
e_plain = werr((U[:, :r] * s0[:r]) @ Vh[:r])
check("beats plain SVD in the weighted norm", e_w < e_plain,
      f"weighted {e_w:.4f} < plain {e_plain:.4f}")

section("[3] low-rank embedding + tied head")
emb = LowRankEmbedding(Af[:, :r].float(), Bf[:r].float())
head = LowRankHead(tied=emb)
ids = torch.randint(0, V, (3, 7))
Wr = (Af[:, :r] @ Bf[:r]).float()
check("embedding lookup == rows of A B", torch.allclose(emb(ids), Wr[ids], atol=1e-5))
h = torch.randn(3, 7, d)
check("head logits == h (A B)^T", torch.allclose(head(h), h @ Wr.T, atol=1e-4))
m = torch.nn.ModuleDict({'e': emb, 'h': head})
check("tied factors counted once", sum(p.numel() for p in m.parameters()) == r * (V + d))

section("[4] chunked KD loss")
torch.manual_seed(1)
Vh_, dh = 97, 16
hs_lin = torch.nn.Linear(dh, Vh_, bias=False)
ht_lin = torch.nn.Linear(dh, Vh_, bias=False).requires_grad_(False)
hs = torch.randn(2, 11, dh, requires_grad=True)
ht = torch.randn(2, 11, dh)
lab = torch.randint(0, Vh_, (2, 11))
w = torch.tensor([1.0, 1.5])
a, T = 0.7, 2.0
ls, lt = hs_lin(hs[:, :-1]), ht_lin(ht[:, :-1])
ce = F.cross_entropy(ls.reshape(-1, Vh_), lab[:, 1:].reshape(-1), reduction='none').view(2, 10)
kl = F.kl_div(F.log_softmax(ls / T, -1), F.log_softmax(lt / T, -1), log_target=True,
              reduction='none').sum(-1) * T * T
ref = (w[:, None] * ((1 - a) * ce + a * kl)).mean()
gref = torch.autograd.grad(ref, [hs, hs_lin.weight])
got = q5.chunked_kd_loss(hs, ht, lab, w, hs_lin, ht_lin, alpha=a, T=T, chunk=7)
g = torch.autograd.grad(got, [hs, hs_lin.weight])
check("value matches", torch.allclose(got, ref, atol=1e-5), f"{got.item():.6f} vs {ref.item():.6f}")
check("gradients match", all(torch.allclose(x, y, atol=1e-5) for x, y in zip(g, gref)))
ce_only = q5.chunked_kd_loss(hs, None, lab, torch.ones(2), hs_lin, None, chunk=5)
check("CE-only path == mean CE", torch.allclose(ce_only, ce.mean(), atol=1e-5))

section("[5] Fisher importance")
model = q5.tiny_qwen3(128)
cands = [(f"L{i}.{n}", getattr(l.self_attn, n, None) or getattr(l.mlp, n))
         for i, l in enumerate(model.model.layers) for n in ('q_proj', 'down_proj')]
rows = [torch.randint(0, 128, (33,)) for _ in range(4)]
grams, hgram, imp, freq_c = q5.calibrate(model, cands, model.lm_head, rows,
                                         torch.device('cpu'), batch=4)
x = torch.stack(rows)
model.eval()
loss = model(x, labels=x).loss
gr = torch.autograd.grad(loss, [m.weight for _, m in cands])
direct = torch.tensor([float((g_ * m.weight.detach()).double().pow(2).sum())
                       for g_, (_, m) in zip(gr, cands)], dtype=torch.float64)
check("matches (dL/dW * W)^2", torch.allclose(imp, direct, rtol=1e-4), str(imp.tolist()))
check("grams are X^T X of layer inputs", grams[0].shape == (64, 64) and
      hgram.shape == (64, 64) and torch.allclose(grams[0], grams[0].T))
check("token frequencies count every input token", int(freq_c.sum()) == x.numel())
check("requires_grad restored", all(p.requires_grad for p in model.parameters()))

section("[6] Fisher-weighted allocation")
L, K = 20, 4
err = torch.tensor([[0.3, 0.15, 0.05, 0.0]]).repeat(L, 1)
par = torch.tensor([[10.0, 20.0, 30.0, 40.0]]).repeat(L, 1)
wimp = torch.ones(L); wimp[:10] = 10.0
budget = 0.6 * 40 * L
ch = allocate(err * wimp[:, None], par, budget, method='greedy')
choice = ch[ch['best']]['choice']
spent = sum(par[l, k].item() for l, k in enumerate(choice))
check("budget respected", spent <= budget, f"{spent} <= {budget}")
hi = np.mean([par[l, choice[l]].item() for l in range(10)])
lo = np.mean([par[l, choice[l]].item() for l in range(10, L)])
check("important layers keep more params", hi > lo, f"{hi:.1f} vs {lo:.1f}")

section("[7] INT8 / FP8 fake quant")
W = torch.randn(64, 256)
for fmt, bound in (('int8', 0.5 / 127), ('fp8', 2 ** -4)):
    Wq = W.clone(); fake_quant_(Wq, fmt)
    rel = ((Wq - W).abs() / W.abs().amax(-1, keepdim=True)).max().item()
    check(f"{fmt} max error / row max <= {bound:.4f}", rel <= bound + 1e-6, f"{rel:.4f}")

section("[8] save / load roundtrip")
with tempfile.TemporaryDirectory() as tdir:
    torch.manual_seed(2)
    model = q5.tiny_qwen3(128)
    model.save_pretrained(tdir)
    E = model.model.embed_tokens.weight.detach()
    Ae, _, Be = weighted_embedding_svd(E, torch.rand(128) + 0.1, None)
    new_e = LowRankEmbedding(Ae[:, :16].float(), Be[:16].float())
    model.model.embed_tokens = new_e
    model.lm_head = LowRankHead(tied=new_e)
    model.config.tie_word_embeddings = False
    lin = model.model.layers[0].mlp.down_proj
    U_, s_, Vh_ = torch.linalg.svd(lin.weight.detach(), full_matrices=False)
    model.model.layers[0].mlp.down_proj = LowRankLinear(U_[:, :24] * s_[:24], Vh_[:24])
    for p_ in model.parameters():                       # saved weights are bf16
        p_.data = p_.data.to(torch.bfloat16).float()
    model = model.eval()
    ids = torch.randint(0, 128, (2, 12))
    with torch.no_grad():
        ref = model(ids).logits.float()
    save_compressed(model, tdir, os.path.join(tdir, 'model.pt'))
    m2 = load_compressed(os.path.join(tdir, 'model.pt'), device='cpu', dtype=torch.float32)
    with torch.no_grad():
        got = m2(ids).logits.float()
    check("logits identical after reload", torch.allclose(got, ref, atol=1e-4),
          f"max diff {(got - ref).abs().max():.2e}")
    check("head still tied to the embedding",
          m2.lm_head.tied is m2.model.embed_tokens)

section("[9] decontamination")
dc = Decontam(["Natalia sold clips to 48 of her friends in April, and then she sold "
               "half as many clips in May. How many clips did Natalia sell altogether"])
check("overlapping doc dropped", not dc.clean(
    "Q: natalia sold clips to 48 of her friends in april and then she sold half as "
    "many clips in may. Answer?"))
check("clean doc kept", dc.clean("Tensor networks factor a large matrix into small cores "
                                 "joined by bond indices of limited dimension here."))
check("counts", dc.checked == 2 and dc.dropped == 1)

section("[10] mixture packer")
mix = parse_mix("a:0.7,b:0.2,c:0.1")
pk = MixturePacker(None, mix, seq=32, seed=0, synthetic=True, vocab=100)
out = [next(pk) for _ in range(2000)]
check("windows are seq+1 tokens", all(len(w_) == 33 for w_, _ in out))
frac = {k: pk.counts[k] / 2000 for k in mix}
check("source proportions follow weights",
      all(abs(frac[k] - mix[k]) < 0.03 for k in mix), str(frac))
pk2 = MixturePacker(None, mix, seq=32, seed=0, rank=1, world=2, synthetic=True, vocab=100)
check("ranks draw different streams", next(pk2)[0] != out[0][0])

section("[11] paired bootstrap + gate")
rng = np.random.default_rng(0)
xa = (rng.random(4000) < 0.45).astype(float)
m, lo, hi = bootstrap_diff(xa, xa.copy())
check("identical models: delta 0, CI [0, 0]", m == 0 and lo == 0 and hi == 0)
xb = xa.copy(); flip = rng.choice(np.where(xa == 1)[0], 120, replace=False); xb[flip] = 0
m, lo, hi = bootstrap_diff(xa, xb)
check("known -3 pt shift recovered", abs(m + 3.0) < 1e-9 and lo < m < hi, f"{m:.2f} [{lo:.2f},{hi:.2f}]")
dense = {'mmlu_a': {str(i): float(v) for i, v in enumerate(xa[:2000])},
         'mmlu_b': {str(i): float(v) for i, v in enumerate(xa[2000:])},
         'gsm8k': {str(i): float(v) for i, v in enumerate(xa[:1000])}}
comp = {'mmlu_a': dict(dense['mmlu_a']), 'mmlu_b': dict(dense['mmlu_b']),
        'gsm8k': {k: 0.0 for k in dense['gsm8k']}}
res = gate(dense, comp, saving_pct=21.0)
check("MMLU pooled across subjects", res['mmlu']['n'] == 4000)
check("GSM8K collapse fails the gate", not res['PASS'] and res['mmlu']['pass_point']
      and not res['gsm8k']['pass_point'])
check("saving below 20 % fails", not gate(dense, dense, saving_pct=19.0)['PASS']
      and gate(dense, dense, saving_pct=20.0)['PASS'])
ps = per_sample({'gsm8k': [{'doc_id': 0, 'filter': 'strict-match', 'exact_match': 1.0},
                           {'doc_id': 0, 'filter': 'flexible-extract', 'exact_match': 0.0}],
                 'mmlu_x': [{'doc_id': 3, 'filter': 'none', 'acc': 1.0}]})
check("per-sample parser keeps strict-match", ps == {'gsm8k': {'0': 1.0}, 'mmlu_x': {'3': 1.0}})

section("[12] end-to-end smoke")
with tempfile.TemporaryDirectory() as tdir:
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'ql5_compress.py'),
                        '--smoke', '--out', tdir, '--regauge-every', '2',
                        '--quant', 'int8', '--save-model'],
                       capture_output=True, text=True, timeout=600)
    ok = r.returncode == 0 and os.path.exists(os.path.join(tdir, 'ql5_report.json'))
    if not ok:
        print(r.stdout[-2000:], r.stderr[-2000:])
    rep = json.load(open(os.path.join(tdir, 'ql5_report.json'))) if ok else {}
    check("runs and writes a report", ok)
    check("total saving within 1 pt of 20 %", ok and abs(rep['param_saving_pct'] - 20) < 1,
          str(rep.get('param_saving_pct')))
    check("embedding compressed and layers allocated", ok and rep.get('embed_rank', 0) > 0
          and len(rep.get('layer_ranks', {})) > 0)
    check("PPLs finite, quantised copy evaluated", ok and all(
        np.isfinite(rep[k]) for k in ('ppl_dense', 'ppl_compressed', 'ppl_healed'))
        and 'int8' in rep['quant'])
    if ok:
        m3 = load_compressed(os.path.join(tdir, 'model.pt'), device='cpu')
        check("saved smoke model reloads", m3.lm_head.tied is m3.model.embed_tokens)

print()
print("QL5 VALIDATION:", "ALL PASS" if PASS else "FAILURES")
sys.exit(0 if PASS else 1)
