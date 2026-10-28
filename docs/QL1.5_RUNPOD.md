# QL1.5 on RunPod — Quick Start

Goal: run the TN-compression heal fine-tune on a real 0.6B model with a
rented GPU. Total cost estimate: **< US$1** (under 1 hour on an RTX 4090).

## 1. Start a pod

- RunPod console → **Deploy** → *GPU Instances* → *Community Cloud*
- GPU: **RTX 4090 24GB** (~$0.34/h) or **A100 40/80GB** for headroom
- Template: **PyTorch 2.x (Ubuntu 22.04, CUDA 12.x)** — any recent one
- Disk: 20 GB container + 10 GB network volume (optional, for checkpoints)

## 2. Bootstrap

Open the pod web terminal (or SSH) and run:

```bash
curl -sL https://raw.githubusercontent.com/tonythetiger168/qweave/main/scripts/runpod_setup.sh | bash
```

The script:
1. installs `transformers datasets accelerate` (+ optional flash-attn)
2. runs the QL1 validation suite (14 checks)
3. runs the QL1.5 smoke pipeline (tiny random GPT, no downloads)
4. runs the real heal: Qwen3-0.6B, χ≤32, ε=1e-3, 300 steps, bf16 + grad ckpt

## 3. Expected output

`/workspace/ql15/ql15_report.json`:

```json
{ "model": "Qwen/Qwen3-0.6B", "chi": 32, "eps": 0.001,
  "compressed_layers": ..., "param_saving_pct": ...,
  "ppl_baseline": ..., "ppl_healed": ...,
  "ppl_relative_increase_pct": ...,   // acceptance: <= 3.0
  "peak_vram_gb": ..., "train_tokens_per_s": ... }
```

**Acceptance gate (QL1.5)**: PPL relative increase ≤ 3% after heal.

## 4. Tuning knobs

| Flag | Default | Notes |
|---|---|---|
| `--chi` | 32 | bond cap; 16/24 for harder budgets |
| `--eps` | 1e-3 | adaptive truncation threshold |
| `--compress-frac` | 0.5 | fraction of candidate layers compressed |
| `--steps` / `--lr` | 300 / 1e-4 | raise steps to 1000+ if PPL gate fails |
| `--model` | Qwen/Qwen3-0.6B | any HF causal LM of similar size |

## 5. Toward QL2

- Multi-GPU: RunPod 4×A100 pods; QL2 adds FSDP wrapping + the Triton TN
  matmul kernel (`phase4_llm/tn_layers.py` is the reference einsum version).
- The annealing optimizer (QUBO relaxation + GPU-simulated annealing) is
  specified in docs/QUANTUM_LLM_ROADMAP.md §QL2.

## 6. Troubleshooting

- `flash-attn` build fails → script already falls back to eager attention
  (slower but fine at 0.6B).
- OOM on 24GB → `--batch 2`, keep `--chi 32`.
- WikiText-2 download blocked → script falls back to synthetic text
  (PPL numbers become meaningless; fix network or pre-upload data).
