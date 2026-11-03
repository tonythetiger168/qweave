#!/bin/bash
# qweave H2H -- head-to-head with FermiQLLM 1.0's "+10-20 % on MATH-500 / GPQA-Diamond / BBH"
# on ONE H100: score same-size open baselines, train an open GRPO recipe, score it, report.
#
# Shutdown contract: when the run ends (success, failure or timeout) the
# reports are pushed to HF with retries, THEN the pod is stopped (retried);
# a container restarted after a finished run goes straight to push + stop.
#
# Env:  HF_TOKEN (RunPod secret) -> reports pushed to <hf-user>/qweave-runs/runs/<pod>_h2h
#       BRANCH (default claude/h2h-fermiqllm), QUICK=1 for a ~15-minute pod check
set -u
export PIP_ROOT_USER_ACTION=ignore HF_HOME=/workspace/hf TOKENIZERS_PARALLELISM=false
OUT=/workspace/h2h; mkdir -p $OUT
exec > >(tee -a $OUT/run.log) 2>&1
POD=${RUNPOD_POD_ID:-local}

stop_pod() {
  for i in $(seq 1 20); do    # retried: the pod dies as soon as one call lands
    echo "stop_pod attempt $i $(date -u +%H:%M:%S)"
    [ -n "${RUNPOD_API_KEY:-}" ] && curl -sS -X POST "https://rest.runpod.io/v1/pods/$POD/stop" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" >/dev/null 2>&1
    command -v runpodctl >/dev/null && runpodctl stop pod "$POD" >/dev/null 2>&1
    sleep 30
  done
  echo "stop_pod: no stop call succeeded (no RUNPOD_API_KEY / runpodctl?)"
}

cat > /workspace/push_h2h.py <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = api.whoami()["name"] + "/qweave-runs"
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(folder_path="/workspace/h2h", repo_id=repo, repo_type="dataset",
                  path_in_repo="runs/" + os.environ.get("RUNPOD_POD_ID", "local") + "_h2h",
                  ignore_patterns=["*.safetensors", "*.bin", "*.pt", "**/checkpoint-*/**",
                                   "**/tokenizer*", "**/vocab*", "**/merges.txt"])
print("pushed ->", repo, flush=True)
PY
push() { python /workspace/push_h2h.py >/dev/null 2>&1 || true; }
push_retry() { # final upload: retried, and always before the pod is stopped
  for i in 1 2 3 4 5; do
    python /workspace/push_h2h.py && return 0
    echo "push attempt $i failed"; sleep $(( i * 20 ))
  done
  echo "push failed 5x -- reports stay on the pod disk ($OUT)"; return 1
}

stage() {   # stage <name> <timeout> <cmd...>
  local name=$1 to=$2; shift 2
  echo "=== $(date -u +%H:%M:%S) [$name] $*"
  local t=$(date +%s)
  timeout "$to" "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  grep -E "^\[h2h\]|^\[grpo\]|Error|error:" "$OUT/$name.log" | tail -12
  echo "$name rc=$rc wall=$(( $(date +%s) - t ))s" | tee -a $OUT/STAGES
  push
}

# container restarted after a finished run: never rerun, just push + stop
if [ -f $OUT/DONE ]; then push_retry; stop_pod; exec sleep infinity; fi
trap 'date -u > $OUT/DONE; push_retry; stop_pod' EXIT

echo "=== $(date -u +%H:%M:%S) setup"
nvidia-smi --query-gpu=name,memory.total --format=csv | tee $OUT/gpu.txt
pip install -q vllm trl "math-verify[antlr4_13_2]" datasets huggingface_hub 2>&1 | tail -3
python -c "import torch, vllm, trl, transformers; print('torch', torch.__version__, 'vllm', vllm.__version__, \
'trl', trl.__version__, 'tf', transformers.__version__)" | tee $OUT/versions.txt
cd /workspace && rm -rf qweave && git clone -q --depth 1 -b "${BRANCH:-claude/h2h-fermiqllm}" \
    https://github.com/tonythetiger168/qweave && cd qweave
git log --oneline -1 | tee $OUT/commit.txt
push

if [ "${QUICK:-0}" = 1 ]; then
  L="--limit 20"; GR="--steps 4 --prompts-per-step 4 --num-generations 4 --max-completion 256 --n-gsm 64 --n-math 64"
else
  L=""; GR="--steps 400"
fi
E="python phase4_llm/h2h/eval_h2h.py --out $OUT"

stage E_base     60m  $E --model Qwen/Qwen3-0.6B-Base --mode plain   --name qwen3-0.6b-base $L
stage E_nothink  60m  $E --model Qwen/Qwen3-0.6B      --mode nothink --name qwen3-0.6b-nothink $L
stage E_think    90m  $E --model Qwen/Qwen3-0.6B      --mode think   --name qwen3-0.6b-think $L
stage E_1p7b     60m  $E --model Qwen/Qwen3-1.7B      --mode nothink --name qwen3-1.7b-nothink $L
stage G_grpo    180m  python phase4_llm/h2h/grpo_h2h.py --model Qwen/Qwen3-0.6B $GR --out $OUT/grpo-0.6b
stage E_grpo     60m  $E --model $OUT/grpo-0.6b       --mode nothink --name qweave-grpo-0.6b-nothink $L
stage R_report   10m  python phase4_llm/h2h/report_h2h.py --dir $OUT --baseline qwen3-0.6b-nothink
cat $OUT/h2h_table.md 2>/dev/null
cat $OUT/STAGES
