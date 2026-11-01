"""
QL7 QUICK check: do vLLM rollouts follow the trained HF weights?
================================================================
Perturbs the HF policy (scales one MLP down-projection), pushes the weights
with VLLMEngine.sync, then compares vLLM prompt log-probs of a fixed text to
the HF log-probs. Pass: mean |diff| < 0.05 nats after the sync, and the
perturbation is visible (mean |diff| before the sync is larger). GPU only.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase4_llm.ql7_grpo_train import HFTok, VLLMEngine  # noqa: E402

TEXT = ("Question: Natalia sold clips to 48 of her friends in April, and then she sold half "
        "as many clips in May. How many clips did Natalia sell altogether?\nAnswer: 72")


def vllm_logp(eng, ids):
    from vllm import SamplingParams, TokensPrompt
    o = eng.llm.generate([TokensPrompt(prompt_token_ids=ids)],
                         SamplingParams(max_tokens=1, prompt_logprobs=0), use_tqdm=False)[0]
    return torch.tensor([list(d.values())[0].logprob if d else 0.0
                         for d in o.prompt_logprobs[1:]])


def hf_logp(model, ids):
    x = torch.tensor([ids], device=model.device)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        lg = model(input_ids=x).logits[0, :-1].float()
    return torch.log_softmax(lg, -1).gather(-1, x[0, 1:, None])[:, 0].cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM
    tok = HFTok(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).cuda()
    eng = VLLMEngine(a.model, tok)
    ids = tok.encode(TEXT)
    d0 = float((vllm_logp(eng, ids) - hf_logp(model, ids)).abs().mean())
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith('layers.0.mlp.down_proj.weight') or n.endswith('layers.1.mlp.down_proj.weight'):
                p.mul_(0.0)
    d_before = float((vllm_logp(eng, ids) - hf_logp(model, ids)).abs().mean())
    eng.sync(model)
    d_after = float((vllm_logp(eng, ids) - hf_logp(model, ids)).abs().mean())
    res = {'diff_initial': d0, 'diff_perturbed_before_sync': d_before,
           'diff_after_sync': d_after, 'pass': d_after < 0.05 and d_before > 2 * d_after}
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps(res), flush=True)
    sys.stdout.flush()
    os._exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
