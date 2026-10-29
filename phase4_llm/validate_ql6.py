"""
QL6 validation suite (CPU, no downloads) -- run from repo root:
    python phase4_llm/validate_ql6.py

Checks
  1. byte roofline: dense Qwen3-0.6B bytes/token match a hand count;
     tied embedding read as one row, LM head as the full matrix
  2. ceilings: QL5's -20 % on 0.6B stays below 1.3x (so the gate needs
     more than low rank alone), INT8 is judged against dense INT8
  3. rank alignment: padding is exact, truncation is the optimal rank-r
     approximation of A B (Eckart-Young), every bond ends aligned
  4. sibling fusion: gate/up fused (dense and low rank) == unfused model
  5. LowRankLinear -> Linear pairs keeps the model output
  6. speculative decoding reproduces greedy exactly (self, noisy and
     random drafts) on the dynamic-cache and the static-cache (CUDA-graph)
     paths, and the acceptance statistics make sense
  7. speculative-decoding speedup formula: limits and a known value
  8. paired bootstrap speedup CI and gate logic (lossy needs QL5 PASS,
     lossless needs agreement, INT8 judged vs matched dense)
  9. end-to-end ql6_decode.py --smoke writes a report with all sections
"""

import json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import torch

from phase4_llm.tn_layers import LowRankLinear, LowRankEmbedding
from phase4_llm import ql6_decode as q6
from phase4_llm.ql5_compress import tiny_qwen3

PASS = True


def check(name, cond, detail=""):
    global PASS
    PASS &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def section(t):
    print("=" * 70); print(t); print("=" * 70)


torch.manual_seed(0)

section("[1] byte roofline")
m = q6.meta_qwen3('Qwen3-0.6B')
b = q6.decode_bytes(m, ctx=0)
per_layer = 1024 * 2048 + 2 * 1024 * 1024 + 2048 * 1024 + 3 * 1024 * 3072   # q,k,v,o,gate,up,down
check("transformer Linears = 28 x 15.7 M x 2 B", b['linear'] == 2 * 28 * per_layer,
      f"{b['linear'] / 2e6:.1f} M params")
check("LM head read in full (151,936 x 1,024)", b['head'] == 2 * 151936 * 1024)
check("tied embedding read as one row", b['embed'] == 2 * 1024)
b8k = q6.decode_bytes(m, ctx=8192)
check("KV cache term = 2 * 28 layers * 8 heads * 128 * ctx * 2 B",
      b8k['kv_cache'] == 2 * 28 * 8 * 128 * 8192 * 2, f"{b8k['kv_cache'] / 1e6:.0f} MB")

section("[2] ceilings")
rows = q6.roofline_table(models=('Qwen3-0.6B',), keeps=(1.0, 0.9, 0.6), embed_keep=(1.0, 0.5),
                         ctx=(0,), int8=(False, True))
get = lambda **k: next(r for r in rows if all(r[a] == v for a, v in k.items()))
ql5 = get(layer_keep=0.9, embed_keep=0.5, int8=False)
check("QL5-like plan (layers 0.9, embedding 0.5) saves ~20 %",
      18 <= ql5['param_saving_pct'] <= 22, f"{ql5['param_saving_pct']} %")
check("... and its ceiling is < 1.3x: low rank alone cannot pass at -20 %",
      ql5['ceiling_vs_dense_bf16'] < 1.3, f"{ql5['ceiling_vs_dense_bf16']}x")
deep = get(layer_keep=0.6, embed_keep=0.5, int8=False)
check("layers 0.6 + embedding 0.5 clears 1.3x on bytes", deep['ceiling_vs_dense_bf16'] > 1.3,
      f"{deep['ceiling_vs_dense_bf16']}x at {deep['param_saving_pct']} % saving")
q8 = get(layer_keep=0.9, embed_keep=0.5, int8=True)
check("INT8 ceiling vs dense INT8 equals the bf16 ratio",
      abs(q8['ceiling_vs_dense_int8'] - ql5['ceiling_vs_dense_bf16']) < 0.01)
dense = get(layer_keep=1.0, embed_keep=1.0, int8=False)
check("dense vs dense = 1.0", dense['ceiling_vs_dense_bf16'] == 1.0)

section("[3] rank alignment")
torch.manual_seed(1)
W = torch.randn(96, 80, dtype=torch.float64)
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
r0 = 37
A0 = (U[:, :r0] * S[:r0]) @ torch.randn(r0, r0, dtype=torch.float64)    # random gauge
B0 = torch.linalg.solve(A0.T @ A0, A0.T) @ (U[:, :r0] * S[:r0]) @ Vh[:r0]
lr = LowRankLinear(A0.clone(), B0.clone())
ref = (lr.A @ lr.B).detach().clone()
up = LowRankLinear(A0.clone(), B0.clone())
ch = q6.align_ranks_(torch.nn.Sequential(up), 8, 'up')
check("pad 37 -> 40 is exact", up.rank == 40 and torch.allclose(up.A @ up.B, ref, atol=1e-9),
      str(ch))
dn = LowRankLinear(A0.clone(), B0.clone())
q6.align_ranks_(torch.nn.Sequential(dn), 8, 'down')
Ur, Sr, Vr = torch.linalg.svd(ref)
best32 = (Ur[:, :32] * Sr[:32]) @ Vr[:32]
check("truncate 37 -> 32 = optimal rank-32 approximation",
      dn.rank == 32 and torch.allclose(dn.A @ dn.B, best32, atol=1e-8))
model = tiny_qwen3(128)
q6.synthetic_lowrank_(model, 0.5, 0, align=1)
q6.align_ranks_(model, 8)
check("every bond aligned after align_ranks_", not q6.rank_audit(model, 8)['misaligned'],
      f"{q6.rank_audit(model, 8)['n_bonds']} bonds")

section("[4] sibling fusion")
ids = torch.randint(1, 128, (2, 20))
for kind in ('dense', 'lowrank'):
    torch.manual_seed(2)
    m1 = tiny_qwen3(128).eval()
    if kind == 'lowrank':
        q6.synthetic_lowrank_(m1, 0.5, 0, align=8)
    with torch.no_grad():
        y0 = m1(ids).logits
        n = q6.fuse_siblings_(m1)
        y1 = m1(ids).logits
    check(f"{kind}: {n} MLPs fused, logits unchanged", n == 2 and torch.allclose(y0, y1, atol=1e-5),
          f"max |d| = {(y0 - y1).abs().max():.1e}")
b_unf = q6.decode_bytes(q6.meta_qwen3('Qwen3-0.6B'))['total']
mf = q6.meta_qwen3('Qwen3-0.6B'); q6.fuse_siblings_(mf)
check("fusion does not change the byte count", q6.decode_bytes(mf)['total'] == b_unf)

section("[5] LowRankLinear -> Linear pairs")
torch.manual_seed(3)
m2 = tiny_qwen3(128).eval()
q6.synthetic_lowrank_(m2, 0.5, 16, align=8)
with torch.no_grad():
    y0 = m2(ids).logits
    n = q6.as_linear_pairs_(m2)
    y1 = m2(ids).logits
check(f"{n} layers converted, logits unchanged", n == 14 and torch.allclose(y0, y1, atol=1e-5))

section("[6] speculative decoding is lossless")
torch.manual_seed(4)
# untied: a tied random model just repeats its input token, so every draft
# would trivially agree
tgt = tiny_qwen3(256, tie=False, seed=4).eval()
noisy = tiny_qwen3(256, tie=False, seed=4).eval()
with torch.no_grad():
    for p in noisy.parameters():
        p.add_(0.05 * p.std() * torch.randn_like(p))
rand = tiny_qwen3(256, tie=False, seed=99).eval()
prompt = torch.randint(1, 256, (1, 9), generator=torch.Generator().manual_seed(5))
ref = q6.greedy_generate(tgt, prompt, 40)
hf_ref = tgt.generate(prompt, max_new_tokens=40, min_new_tokens=40, do_sample=False,
                      attention_mask=torch.ones_like(prompt), pad_token_id=0)
check("own greedy loop == HF generate (greedy)", torch.equal(ref, hf_ref))
for name, d in (('self', tgt), ('noisy', noisy), ('random', rand)):
    for g in (1, 4, 7):
        out, st = q6.speculative_generate(tgt, d, prompt, 40, gamma=g)
        if g == 4:
            check(f"draft={name}: output == greedy, alpha={st['alpha']:.2f}, "
                  f"{st['tokens_per_target_call']:.2f} tok/target call", torch.equal(out, ref))
        elif not torch.equal(out, ref):
            check(f"draft={name} gamma={g}: output == greedy", False)
_, st = q6.speculative_generate(tgt, tgt, prompt, 40, gamma=4)
check("self-draft accepts everything (alpha = 1, 5 tokens per call)",
      st['alpha'] == 1.0 and st['target_calls'] == 8)
_, st = q6.speculative_generate(tgt, noisy, prompt, 40, gamma=4)
check("noisy draft: 0 < alpha < 1", 0 < st['alpha'] < 1, f"{st['alpha']:.2f}")
_, st = q6.speculative_generate(tgt, rand, prompt, 40, gamma=4)
check("random draft still makes >= 1 token per target call", st['tokens_per_target_call'] >= 1.0)
out, _ = q6.speculative_generate(tgt, noisy, prompt[:, :1], 13, gamma=4)
check("1-token prompt and max_new not a multiple of gamma",
      torch.equal(out, q6.greedy_generate(tgt, prompt[:, :1], 13)))
eos = int(ref[0, prompt.shape[1] + 5])
out, _ = q6.speculative_generate(tgt, noisy, prompt, 40, gamma=4, eos_token_id=eos)
check("stops at EOS like greedy", torch.equal(out, q6.greedy_generate(tgt, prompt, 40, eos)))
g_fn, s_fn, _ = q6.spec_engine(tgt, noisy, 'eager', 64)
check("lossless_check = 1.0 in fp32", q6.lossless_check(g_fn, lambda i, n: s_fn(i, n, 4),
                                                       [prompt], 30) == 1.0)
# static-cache path (the one compiled to CUDA graphs on GPU), uncompiled on CPU
g_fn, s_fn, d_fn = q6.spec_engine(tgt, noisy, 'static', 64, gammas=(1, 4, 7))
check("static greedy == greedy", torch.equal(g_fn(prompt, 40), ref))
ok = True
for d, name in ((tgt, 'self'), (noisy, 'noisy'), (rand, 'random')):
    _, s2, _ = q6.spec_engine(tgt, d, 'static', 64, gammas=(1, 4, 7))
    for g in (1, 4, 7):
        out, st = s2(prompt, 40, g)
        ok &= torch.equal(out, ref)
    for p1, n in ((prompt[:, :1], 13), (prompt, 3)):
        ok &= torch.equal(s2(p1, n, 4)[0], q6.greedy_generate(tgt, p1, n))
check("static speculative == greedy (3 drafts x gamma 1/4/7, short prompt, overshoot)", ok)
out, st = s_fn(prompt, 40, 4)
check("static and dynamic paths accept the same drafts",
      st['alpha'] > 0 and abs(st['alpha'] - q6.speculative_generate(tgt, noisy, prompt, 40, 4)[1]['alpha']) < 0.15,
      f"alpha {st['alpha']:.2f}")

section("[7] speculative speedup formula")
check("alpha=0 -> 1/(gamma c + 1)", abs(q6.spec_speedup(0.0, 4, 0.25) - 0.5) < 1e-12)
check("alpha=1, c=0 -> gamma + 1", abs(q6.spec_speedup(1.0, 4, 0.0) - 5) < 1e-12)
check("alpha=0.8, gamma=4, c=0.1 -> 2.40x", abs(q6.spec_speedup(0.8, 4, 0.1) - 2.4010) < 1e-3,
      f"{q6.spec_speedup(0.8, 4, 0.1):.4f}")

section("[8] bootstrap CI + gate")
rng = np.random.default_rng(0)
tb = rng.uniform(1.0, 2.0, 24)
s, lo, hi = q6.ratio_ci(tb, tb / 1.5)
check("exact 1.5x ratio -> CI collapses on 1.5", abs(s - 1.5) < 1e-9 and abs(lo - 1.5) < 1e-9)
s, lo, hi = q6.ratio_ci(tb, tb / 1.5 * rng.uniform(0.8, 1.2, 24))
check("noisy 1.5x -> CI brackets the estimate", lo < s < hi and 1.3 < s < 1.7, f"{s:.3f} [{lo:.3f}, {hi:.3f}]")
rep = {'decode': [
    {'variant': 'dense', 'engine': 'compiled'},
    {'variant': 'syn', 'engine': 'compiled', 'route': 'L1', 'synthetic': True,
     'speedup': 2.0, 'ci95': [1.9, 2.1]},
    {'variant': 'ql5', 'engine': 'compiled', 'route': 'L1', 'quality_pass': False,
     'speedup': 1.5, 'ci95': [1.4, 1.6]}]}
g = q6.gate(rep)
check("synthetic (speed-only) and quality-failing lossy variants cannot pass", not g['decode_PASS'])
rep['decode'].append({'variant': 'ql5+int8', 'engine': 'compiled', 'route': 'L1',
                      'quality_pass': True, 'speedup': 2.2, 'ci95': [2.1, 2.3],
                      'speedup_vs_matched': 1.2, 'ci95_vs_matched': [1.1, 1.3]})
check("INT8 judged vs dense INT8 (1.2x) -> FAIL", not q6.gate(rep)['decode_PASS'])
rep['decode'].append({'variant': 'spec', 'engine': 'eager', 'route': 'L2', 'agreement': 0.995,
                      'speedup': 1.35, 'ci95': [1.25, 1.45]})
g = q6.gate(rep)
check("lossless 1.35x [1.25, 1.45] -> PASS point, FAIL at CI bound",
      g['decode_PASS'] and not g['decode_PASS_at_CI_bound'])
rep['prefill'] = [{'tokens': 32768, 'speedup': 2.3}]
check("prefill 32K 2.3x -> PASS", q6.gate(rep)['prefill_PASS'])

section("[9] end-to-end smoke")
with tempfile.TemporaryDirectory() as tdir:
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'ql6_decode.py'),
                        '--smoke', '--out', tdir], capture_output=True, text=True, timeout=900)
    ok = r.returncode == 0 and os.path.exists(os.path.join(tdir, 'ql6_report.json'))
    if not ok:
        print(r.stdout[-3000:], r.stderr[-3000:])
    rep = json.load(open(os.path.join(tdir, 'ql6_report.json'))) if ok else {}
    check("runs and writes a report", ok)
    dec = [d for d in rep.get('decode', []) if 'error' not in d]
    check("decode rows for dense, synthetic and fused variants, no errors",
          ok and {'dense', 'dense+fuse'} <= {d['variant'] for d in dec}
          and any(d['variant'].startswith('synthetic') for d in dec)
          and not any('error' in d for d in rep['decode']),
          str([d.get('error') for d in rep.get('decode', []) if 'error' in d])[:300])
    check("speculative rows lossless with alpha in [0, 1]", ok and rep['spec'] and all(
        s['agreement'] == 1.0 and 0 <= s['alpha'] <= 1 for s in rep['spec']))
    check("self-draft alpha = 1", ok and any(s['draft'] == 'self' and s['alpha'] == 1.0
                                             for s in rep['spec']))
    check("prefill dense vs QSA timed", ok and rep['prefill'] and 'speedup' in rep['prefill'][0],
          str(rep.get('prefill'))[:200])
    check("roofline per variant + gate present", ok and 'dense' in rep['roofline'] and 'gate' in rep)

print()
print("QL6 VALIDATION:", "ALL PASS" if PASS else "FAILURES")
sys.exit(0 if PASS else 1)
