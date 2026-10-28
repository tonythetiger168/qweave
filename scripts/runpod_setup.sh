#!/bin/bash
# qweave QL1.5 — RunPod bootstrap (run inside a PyTorch CUDA pod)
set -e
cd /workspace
if [ ! -d qweave ]; then
  git clone https://github.com/tonythetiger168/qweave.git
fi
cd qweave

echo "=== [1/4] install deps"
pip install -q transformers datasets accelerate
pip install -q flash-attn --no-build-isolation 2>/dev/null || \
  echo "(flash-attn optional, falling back to eager attention)"

echo "=== [2/4] QL1 validation (14 checks)"
python3 phase4_llm/validate_ql1.py

echo "=== [3/4] QL1.5 pipeline smoke (no downloads)"
python3 phase4_llm/heal_finetune.py --smoke --steps 30 --out /workspace/ql15-smoke

echo "=== [4/4] real heal run (Qwen3-0.6B, fits in 24GB)"
python3 phase4_llm/heal_finetune.py \
  --model Qwen/Qwen3-0.6B --chi 32 --eps 1e-3 \
  --steps 300 --batch 4 --lr 1e-4 --out /workspace/ql15

echo "=== done. report at /workspace/ql15/ql15_report.json"
