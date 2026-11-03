"""
QL7 GRPO runner: Qwen3-Base on GSM8K, with QUBO prompt / layer selection
=======================================================================
One run = one arm x one seed. Every arm uses the SAME loop, engine and
hyper-parameters; only the selection flags differ, so GPU-hour ratios
measure the selection and nothing else. A TRL GRPOTrainer reference run
(ql7_trl_ref.py) checks that this baseline loop is not slower than the
standard tool (else a "saving" could just be a slow baseline).

  python phase4_llm/ql7_grpo_train.py --model Qwen/Qwen3-0.6B-Base \
      --engine vllm --steps 300 --select-prompts sqa --out runs/qp_sqa_s0

Arms (docs/QL7_RUN_PLAN.md):
  base         --select-prompts uniform --select-layers none   (named baseline)
  dyn          --dynamic-sampling (DAPO: oversample, drop zero-signal groups)
  qp_<solver>  --select-prompts {greedy,greedy_swap,sa,sqa,random}
  ql_<solver>  --select-layers {topk,greedy_swap,sqa,random} --layer-k K

Outputs <out>/run.json: per-step log (phase seconds, rollouts, zero-signal
groups, solve time), eval curve vs GPU-hours, final per-item correctness.
GPU-hours exclude eval (identical work for every arm) and include the QUBO
solve and probe steps.

--smoke: tiny random Qwen3 + character tokenizer + synthetic addition
problems on CPU (no downloads); checks the whole loop end to end.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase4_llm import ql7_rl as R  # noqa: E402

TEMPLATE = "Question: {q}\nAnswer: Let's think step by step."


# ------------------------------------------------------------------ data

def load_gsm8k(split, limit=0):
    from datasets import load_dataset
    ds = load_dataset('openai/gsm8k', 'main', split=split)
    rows = [{'q': r['question'], 'gold': R.gsm8k_gold(r['answer'])} for r in ds]
    return rows[:limit] if limit else rows


def synth_rows(n, seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        rows.append({'q': f'{a}+{b}', 'gold': float(a + b)})
    return rows


class CharTok:
    """Tiny tokenizer for --smoke (printable ASCII + pad/eos)."""
    def __init__(self):
        self.chars = [chr(i) for i in range(32, 127)] + ['\n']
        self.pad_token_id, self.eos_token_id = 0, 1
        self.vocab_size = len(self.chars) + 2

    def encode(self, s):
        return [self.chars.index(c) + 2 for c in s if c in self.chars]

    def decode(self, ids):
        return ''.join(self.chars[i - 2] for i in ids if i >= 2)


class HFTok:
    def __init__(self, name):
        from transformers import AutoTokenizer
        self.t = AutoTokenizer.from_pretrained(name)
        self.eos_token_id = self.t.eos_token_id
        self.pad_token_id = self.t.pad_token_id if self.t.pad_token_id is not None else self.eos_token_id

    def encode(self, s):
        return self.t(s, add_special_tokens=False)['input_ids']

    def decode(self, ids):
        return self.t.decode(ids, skip_special_tokens=True)


# --------------------------------------------------------------- engines

def _pad_left(seqs, pad):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    att = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s)
        att[i, L - len(s):] = 1
    return ids, att


class HFEngine:
    def __init__(self, model, tok, dev):
        self.m, self.tok, self.dev = model, tok, dev

    @torch.no_grad()
    def generate(self, prompts, n, max_new, temperature, seed):
        torch.manual_seed(seed)
        seqs = [p for p in prompts for _ in range(n)]
        ids, att = _pad_left(seqs, self.tok.pad_token_id)
        was = self.m.training
        self.m.eval()
        out = self.m.generate(input_ids=ids.to(self.dev), attention_mask=att.to(self.dev),
                              max_new_tokens=max_new, do_sample=temperature > 0,
                              temperature=temperature if temperature > 0 else None,
                              top_p=1.0, top_k=0, pad_token_id=self.tok.pad_token_id,
                              eos_token_id=self.tok.eos_token_id)
        self.m.train(was)
        comp = out[:, ids.shape[1]:].tolist()
        res = []
        for c in comp:
            if self.tok.eos_token_id in c:
                c = c[:c.index(self.tok.eos_token_id) + 1]
            while c and c[-1] == self.tok.pad_token_id and self.tok.pad_token_id != self.tok.eos_token_id:
                c = c[:-1]
            res.append(c)
        return res

    def sync(self, model):
        pass


class VLLMEngine:
    """vLLM colocated on the training GPU; weights pushed after every step."""
    def __init__(self, name, tok, gpu_mem=0.35, max_len=1024, seed=0):
        os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
        from vllm import LLM
        self.tok = tok
        self.llm = LLM(model=name, dtype='bfloat16', gpu_memory_utilization=gpu_mem,
                       max_model_len=max_len, enable_prefix_caching=True, seed=seed)

    def generate(self, prompts, n, max_new, temperature, seed):
        from vllm import SamplingParams, TokensPrompt
        sp = SamplingParams(n=n, temperature=temperature, top_p=1.0, max_tokens=max_new,
                            seed=seed)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp,
                                 use_tqdm=False)
        return [list(c.token_ids) for o in outs for c in o.outputs]

    def _model(self):
        eng = self.llm.llm_engine
        for path in ('model_executor.driver_worker.worker.model_runner.model',
                     'model_executor.driver_worker.model_runner.model'):
            obj = eng
            try:
                for a in path.split('.'):
                    obj = getattr(obj, a)
                return obj
            except AttributeError:
                continue
        return None

    def sync(self, model):
        named = [(n, p.detach()) for n, p in model.named_parameters()]
        m = self._model()
        if m is not None:
            m.load_weights(named)
        else:                                   # newer vLLM: same-process apply_model
            self.llm.apply_model(lambda mm: mm.load_weights(named))
        try:
            self.llm.reset_prefix_cache()
        except Exception:
            pass


# ------------------------------------------------------------- training

def seq_logp(model, prompt_ids, comp_ids, pad, dev):
    """Per-token log-probs of completions (N, T) and their mask."""
    seqs = [p + c for p, c in zip(prompt_ids, comp_ids)]
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    att = torch.zeros_like(ids)
    cm = torch.zeros_like(ids, dtype=torch.float)
    for i, (p, c) in enumerate(zip(prompt_ids, comp_ids)):
        s = p + c
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
        cm[i, len(p):len(s)] = 1
    ids, att, cm = ids.to(dev), att.to(dev), cm.to(dev)
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None])[..., 0]
    return lp, cm[:, 1:]


def smoke_reward(text, gold):
    """--smoke only: a random tiny model never solves a problem, so reward
    'mentions the answer's last digit' to get groups with signal."""
    return float(str(int(gold))[-1] in text)


def evaluate(engine, tok, rows, max_new, bs=256):
    ok = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        prompts = [tok.encode(TEMPLATE.format(q=r['q'])) for r in chunk]
        comps = engine.generate(prompts, 1, max_new, 0.0, 0)
        ok += [R.math_reward(tok.decode(c), r['gold']) for c, r in zip(comps, chunk)]
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
    ap.add_argument('--engine', choices=['hf', 'vllm'], default='vllm')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--prompts', type=int, default=64, help='prompts per step (B)')
    ap.add_argument('--group', type=int, default=8, help='completions per prompt (G)')
    ap.add_argument('--max-new', type=int, default=384)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--lr', type=float, default=1e-6)
    ap.add_argument('--beta', type=float, default=0.0)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--clip-high', type=float, default=0.28)
    ap.add_argument('--micro', type=int, default=8, help='sequences per micro-batch')
    ap.add_argument('--select-prompts', default='uniform',
                    choices=['uniform', 'greedy', 'greedy_swap', 'sa', 'sqa', 'random'])
    ap.add_argument('--pool-factor', type=int, default=4)
    ap.add_argument('--lam', type=float, default=0.5)
    ap.add_argument('--explore', type=float, default=0.1)
    ap.add_argument('--prior', type=float, default=0.5,
                    help='Beta(prior, prior) pass-rate prior (U-shaped: rates are bimodal)')
    ap.add_argument('--prompt-decay', type=float, default=0.995,
                    help='per-step decay of pass-rate estimates toward the prior')
    ap.add_argument('--solver-sweeps', type=int, default=20)
    ap.add_argument('--overlap-select', type=int, default=1,
                    help='solve the next prompt QUBO on CPU during training (1) or inline (0)')
    ap.add_argument('--dynamic-sampling', action='store_true',
                    help='DAPO baseline: oversample 2x, keep groups with signal')
    ap.add_argument('--select-layers', default='none',
                    choices=['none', 'topk', 'greedy', 'greedy_swap', 'sa', 'sqa', 'random'])
    ap.add_argument('--layer-k', type=int, default=0, help='blocks trained per step')
    ap.add_argument('--probe-every', type=int, default=25)
    ap.add_argument('--probe-prompts', type=int, default=4)
    ap.add_argument('--eval-every', type=int, default=25)
    ap.add_argument('--eval-n', type=int, default=0,
                    help='test items for the curve (0 = full test set, same as the gate)')
    ap.add_argument('--train-limit', type=int, default=0)
    ap.add_argument('--test-limit', type=int, default=0, help='debug only; gate runs use 0')
    ap.add_argument('--max-gpu-hours', type=float, default=0)
    ap.add_argument('--n-gpus', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--grad-ckpt', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() and not a.smoke else 'cpu'

    from transformers import AutoModelForCausalLM
    if a.smoke:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        tok = CharTok()
        cfg = Qwen3Config(vocab_size=tok.vocab_size, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=16, max_position_embeddings=256, tie_word_embeddings=True,
                          pad_token_id=0, eos_token_id=1, bos_token_id=1)
        model = Qwen3ForCausalLM(cfg)
        train_rows, test_rows = synth_rows(64, a.seed), synth_rows(16, 999)
        a.engine = 'hf'
    else:
        tok = HFTok(a.model)
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev)  # fp32 master, bf16 autocast
        train_rows = load_gsm8k('train', a.train_limit)
        test_rows = load_gsm8k('test', a.test_limit)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
    model.train()
    engine = (VLLMEngine(a.model, tok, seed=a.seed) if a.engine == 'vllm'
              else HFEngine(model, tok, dev))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    blocks, _rest = R.block_params(model)
    texts = [r['q'] for r in train_rows]
    ps = R.PromptSelector(texts, a.group, method=a.select_prompts, pool_factor=a.pool_factor,
                          lam=a.lam, explore=a.explore, seed=a.seed, decay=a.prompt_decay,
                          prior=(a.prior, a.prior),
                          solver_kw={'sweeps': a.solver_sweeps}
                          if a.select_prompts in ('sa', 'sqa') else {})
    k = a.layer_k or max(1, len(blocks) // 2)
    ls = R.LayerSelector(len(blocks), k, method=a.select_layers, seed=a.seed,
                         solver_kw={'sweeps': a.solver_sweeps}
                         if a.select_layers in ('sa', 'sqa') else {})
    meter = R.GpuHourMeter(a.n_gpus, sync=dev == 'cuda')
    prompt_ids = [tok.encode(TEMPLATE.format(q=t)) for t in texts]
    eval_rows = test_rows[:a.eval_n] if a.eval_n else test_rows
    log, curve = [], []
    rollouts = 0

    def do_eval(step, rows, tag):
        with meter.phase('eval'):
            ok = evaluate(engine, tok, rows, a.max_new)
        rec = {'step': step, 'gpu_hours': meter.gpu_hours(), 'rollouts': rollouts,
               'score': 100.0 * sum(ok) / max(1, len(ok)), 'n': len(ok), 'tag': tag}
        curve.append(rec)
        print(json.dumps(rec), flush=True)
        return ok

    def save(final=None):
        json.dump({'args': vars(a), 'arm': os.path.basename(a.out.rstrip('/')),
                   'log': log, 'curve': curve, 'final': final,
                   'phase_seconds': dict(meter.sec), 'fractions': meter.fractions(),
                   'gpu_hours': meter.gpu_hours()},
                  open(os.path.join(a.out, 'run.json'), 'w'), indent=1)

    from concurrent.futures import ThreadPoolExecutor
    pool, pending = ThreadPoolExecutor(1), None
    do_eval(0, eval_rows, 'curve')
    for step in range(1, a.steps + 1):
        rec = {'step': step}
        # probe for layer selection: per-block |g|^2 and Hessian cross terms
        if a.select_layers not in ('none', 'random', 'topk') or (
                a.select_layers == 'topk' and step == 1):
            if (step - 1) % a.probe_every == 0:
                with meter.phase('probe'):
                    for b in blocks:
                        for p in b:
                            p.requires_grad_(True)
                    pi = list(range(a.probe_prompts))
                    pc = engine.generate([prompt_ids[i] for i in pi], 2, min(a.max_new, 128),
                                         0.0, 0)
                    pp = [prompt_ids[i] for i in pi for _ in range(2)]

                    def lf():                     # SFT-style NLL on fixed probe text
                        lp, cm = seq_logp(model, pp, pc, tok.pad_token_id, dev)
                        return -(lp * cm).sum() / cm.sum().clamp(min=1)
                    g2, M = R.layer_interactions(model, lf, blocks)
                    ls.set_probe(g2, M)
                    rec['probe_interaction'] = float(
                        (M - torch.diag(torch.diagonal(M))).norm() / (g2.norm() + 1e-30))
        want = a.prompts * (2 if a.dynamic_sampling else 1)
        with meter.phase('select'):        # only the time NOT hidden behind training
            sel = pending.result() if pending is not None else ps.next_batch(want)
            pending = None
        idx = sel['idx']
        with meter.phase('generate'):
            comps = engine.generate([prompt_ids[i] for i in idx], a.group, a.max_new,
                                    a.temperature, a.seed * 100003 + step)
        rollouts += len(comps)
        with meter.phase('score'):
            rf = smoke_reward if a.smoke else R.math_reward
            r = torch.tensor([rf(tok.decode(c), train_rows[i]['gold'])
                              for j, i in enumerate(idx) for c in comps[j * a.group:(j + 1) * a.group]]
                             ).view(len(idx), a.group)
            ps.update(idx, r)
            # the next batch only needs the updated posterior: solve its QUBO on a
            # CPU thread while the GPU trains (torch CPU ops release the GIL)
            if a.overlap_select and step < a.steps:
                pending = pool.submit(ps.next_batch, want)
            z = R.zero_signal(r)
            keep = list(range(len(idx)))
            if a.dynamic_sampling:
                keep = [j for j in keep if not z[j]][:a.prompts]
            adv = R.group_advantages(r)
        rec['prompt_idx'] = list(idx)                      # for the offline persistence
        rec['prompt_succ'] = [int(x) for x in r.sum(1).tolist()]  # analysis in ql7_gate
        rec.update(rollouts=len(comps), reward=float(r.mean()), zero_groups=int(z.sum()),
                   groups=len(idx), select_s=sel['solve_s'], pred_signal=sel.get('pred_signal'),
                   prompt_interaction=sel.get('interaction'))
        with meter.phase('train'):
            lc = ls.choose()
            R.apply_layer_mask(blocks, lc['idx'])
            items = [(j, g) for j in keep for g in range(a.group)
                     if abs(float(adv[j, g])) > 0]          # zero-adv rows add no gradient
            ntok = sum(len(comps[j * a.group + g]) for j, g in items) or 1
            opt.zero_grad(set_to_none=True)
            for s in range(0, len(items), a.micro):
                mb = items[s:s + a.micro]
                pi_ = [prompt_ids[idx[j]] for j, _ in mb]
                ci_ = [comps[j * a.group + g] or [tok.eos_token_id] for j, g in mb]
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
                    lp, cm = seq_logp(model, pi_, ci_, tok.pad_token_id, dev)
                av = torch.tensor([float(adv[j, g]) for j, g in mb], device=dev)
                # DAPO token-level mean over the whole step: sum over micro-batches
                loss, _ = R.grpo_loss(lp, lp.detach(), av, cm, clip=a.clip,
                                      clip_high=a.clip_high, loss_type='dapo')
                (loss * cm.sum() / ntok).backward()
            g2_obs = [sum(float((p.grad.float() ** 2).sum()) for p in blocks[i] if p.grad is not None)
                      for i in lc['idx']]
            ls.observe(lc['idx'], g2_obs)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], 1.0)
            opt.step()
            rec['layers'] = lc['idx'] if a.select_layers != 'none' else None
            rec['train_seqs'] = len(items)
        with meter.phase('sync'):
            engine.sync(model)
        rec['gpu_hours'] = meter.gpu_hours()
        log.append(rec)
        if step % a.eval_every == 0 or step == a.steps:
            do_eval(step, eval_rows, 'curve')
            save()
        if a.max_gpu_hours and meter.gpu_hours() >= a.max_gpu_hours:
            print('gpu-hour cap reached', flush=True)
            break
    ok = do_eval(log[-1]['step'] if log else 0, test_rows, 'final')
    save({'score': 100.0 * sum(ok) / max(1, len(ok)), 'correct': ok,
          'gpu_hours': meter.gpu_hours()})
    print('done', json.dumps({'gpu_hours': meter.gpu_hours(), 'final': curve[-1]['score'],
                              'fractions': meter.fractions()}), flush=True)


if __name__ == '__main__':
    main()
    sys.stdout.flush()
    os._exit(0)          # vLLM / datasets threads can abort at interpreter shutdown
