"""
Load a DiracLM model folder (or hub id)
=======================================
    from phase4_llm.release.loader import load_diraclm
    model, tok = load_diraclm("DiracLM/DiracLM-4B")

A DiracLM folder is a normal transformers checkpoint plus
`diraclm_config.json`. If that file enables QSA, the "qsa" attention
implementation is registered and used: prompts of at least `min_len`
tokens get block-sparse prefill, shorter prompts and every decode step run
dense SDPA. Without qweave installed the checkpoint still loads as the
plain base architecture (dense attention), so no trust_remote_code is
needed.
"""

import json
import os
from typing import Dict, Optional, Tuple

CONFIG_NAME = "diraclm_config.json"


def read_diraclm_config(path: str) -> Dict:
    local = os.path.join(path, CONFIG_NAME)
    if os.path.exists(local):
        with open(local) as f:
            return json.load(f)
    if os.path.isdir(path):
        return {}
    try:                                     # hub id
        from huggingface_hub import hf_hub_download
        with open(hf_hub_download(path, CONFIG_NAME)) as f:
            return json.load(f)
    except Exception:                        # noqa: BLE001 - plain checkpoint
        return {}


def load_diraclm(path: str, device: Optional[str] = None, dtype=None,
                 qsa: Optional[bool] = None) -> Tuple[object, object]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = read_diraclm_config(path)
    q = cfg.get("components", {}).get("qsa")
    use_qsa = bool(q) if qsa is None else qsa
    attn = "sdpa"
    if use_qsa and q:
        from phase4_llm import qsa_attention as Q
        Q.register_hf()
        Q.QSA_CFG.update({k: q[k] for k in ("topk", "block", "local", "sink", "min_len") if k in q})
        attn = "qsa"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype or torch.bfloat16, attn_implementation=attn).to(device).eval()
    return model, tok
