"""
QL7 reference: the same GRPO recipe in TRL's GRPOTrainer (vLLM colocate)
=======================================================================
Not a gate arm. It answers one question: is our baseline loop
(ql7_grpo_train.py, arm 'base') as fast per step as the standard tool?
If TRL is much faster per step, a QL7 "saving" against our own loop would
just be a slow baseline, and the gate must be re-run against TRL instead.

Writes <out>/trl_ref.json: steps, wall seconds, GPU-hours per step, final
GSM8K test accuracy (greedy, same template and grader as the arms).

--smoke: random tiny Qwen3 with the real Qwen3 tokenizer on CPU, a few steps.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase4_llm import ql7_rl as R  # noqa: E402
from phase4_llm.ql7_grpo_train import STOP, TEMPLATE, answer_text, load_gsm8k, synth_rows  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--prompts', type=int, default=64)
    ap.add_argument('--group', type=int, default=8)
    ap.add_argument('--max-new', type=int, default=384)
    ap.add_argument('--lr', type=float, default=1e-6)
    ap.add_argument('--micro', type=int, default=8)
    ap.add_argument('--n-gpus', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tok = AutoTokenizer.from_pretrained(a.model)
    if a.smoke:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=16, tie_word_embeddings=True)
        model = Qwen3ForCausalLM(cfg)
        train, test = synth_rows(32, a.seed), synth_rows(8, 999)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)  # fp32 master weights; bf16 autocast
        train, test = load_gsm8k('train'), load_gsm8k('test')
    ds = Dataset.from_list([{'prompt': TEMPLATE.format(q=r['q']), 'gold': r['gold']}
                            for r in train])

    def reward(prompts, completions, gold, **kw):
        return [R.math_reward(answer_text(c), g) for c, g in zip(completions, gold)]

    cuda = torch.cuda.is_available() and not a.smoke
    per_dev = max(a.group, min(a.micro, a.prompts * a.group))
    cfg = GRPOConfig(
        output_dir=os.path.join(a.out, 'trl'), learning_rate=a.lr, max_steps=a.steps,
        per_device_train_batch_size=per_dev,
        gradient_accumulation_steps=max(1, a.prompts * a.group // per_dev),
        num_generations=a.group, max_completion_length=a.max_new, temperature=1.0,
        beta=0.0, loss_type='dapo', epsilon=0.2, epsilon_high=0.28, num_iterations=1,
        use_vllm=cuda, vllm_mode='colocate', vllm_gpu_memory_utilization=0.35,
        bf16=cuda, use_cpu=not cuda, logging_steps=10, save_strategy='no', report_to='none',
        seed=a.seed, max_grad_norm=1.0, weight_decay=0.0, adam_beta2=0.99,
        generation_kwargs={'stop': STOP} if cuda else None)
    tr = GRPOTrainer(model=model, reward_funcs=reward, args=cfg, train_dataset=ds,
                     processing_class=tok)
    if cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    tr.train()
    if cuda:
        torch.cuda.synchronize()
    wall = time.perf_counter() - t
    steps = tr.state.global_step
    # final eval: greedy, same template and grader as the gate arms (HF generate)
    from phase4_llm.ql7_grpo_train import HFEngine, HFTok, evaluate
    m = tr.model.eval()
    ok = evaluate(HFEngine(m, HFTok(a.model), m.device), HFTok(a.model),
                  test[:16] if a.smoke else test, 8 if a.smoke else a.max_new,
                  bs=8 if a.smoke else 128)
    res = {'steps': steps, 'wall_s': wall, 'gpu_hours': a.n_gpus * wall / 3600,
           'gpu_hours_per_step': a.n_gpus * wall / 3600 / max(1, steps),
           'final_score': 100.0 * sum(ok) / max(1, len(ok)), 'n': len(ok),
           'args': vars(a), 'log_history': tr.state.log_history}
    json.dump(res, open(os.path.join(a.out, 'trl_ref.json'), 'w'), indent=1)
    print('trl_ref', json.dumps({k: res[k] for k in ('steps', 'gpu_hours', 'gpu_hours_per_step',
                                                     'final_score')}), flush=True)


if __name__ == '__main__':
    main()
    sys.stdout.flush()
    os._exit(0)
