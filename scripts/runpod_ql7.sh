#!/bin/bash
# qweave QL7 -- GRPO GPU-hours with QUBO prompt / layer selection, on ONE RunPod pod.
#
#   QUICK=1  1x H100, ~20-30 min (~US$2): CPU validation, smoke, vLLM weight-sync
#            check, 10-step base / qp / ql runs and a 10-step TRL reference
#   STAGE=A1 1x H100, Qwen3-0.6B-Base on GSM8K: baseline seed 0 + TRL reference,
#            200 steps each                        (~2-2.5 h, ~US$8-10 Secure)
#   STAGE=A2 4x H100: 2 more baseline seeds + 13 selection-arm runs, 200 steps,
#            3-4 runs per GPU, then the gate       (~4-5 h, ~US$65-80 Secure)
#   STAGE=B  Qwen3-4B-Base, the winning arm vs base, 3 seeds each
#            -- only after stage A passes; not costed for approval yet
#
# Env:  HF_TOKEN (RunPod secret) -> reports pushed to <hf-user>/qweave-runs
#       BRANCH (default claude/ql7-rl-cost), RUNPOD_API_KEY (optional, stops pod)
#       NO_STOP=1 keeps the pod up at the end
#       STEPS overrides the step count (default 200; QUICK 10)
# Results: /workspace/ql7_runs/stage<X>/*  and  hf://datasets/<user>/qweave-runs/runs/<pod>_ql7/
# Shutdown contract (same as runpod_ql5.sh): reports pushed to HF with retries,
# THEN the pod is stopped; a restarted container after a finished run only pushes + stops.
# Use a timeout in MINUTES around this script (GNU timeout rejects "5h30m").
set -u
export PIP_ROOT_USER_ACTION=ignore HF_HOME=/workspace/hf TOKENIZERS_PARALLELISM=false
export VLLM_ENABLE_V1_MULTIPROCESSING=0
Q=${QUICK:-0}; STAGE=${STAGE:-A1}; [ "$Q" = 1 ] && STAGE=Q
OUT=/workspace/ql7_runs/stage$STAGE; mkdir -p $OUT
exec > >(tee -a $OUT/run.log) 2>&1
POD=${RUNPOD_POD_ID:-local}

stop_pod() {
  [ "${NO_STOP:-0}" = 1 ] && return
  for i in $(seq 1 20); do
    echo "stop_pod attempt $i $(date -u +%H:%M:%S)"
    [ -n "${RUNPOD_API_KEY:-}" ] && curl -sS -X POST "https://rest.runpod.io/v1/pods/$POD/stop" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" >/dev/null 2>&1
    command -v runpodctl >/dev/null && runpodctl stop pod "$POD" >/dev/null 2>&1
    sleep 30
  done
}

cat > /workspace/push_ql7.py <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = api.whoami()["name"] + "/qweave-runs"
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(folder_path="/workspace/ql7_runs", repo_id=repo, repo_type="dataset",
                  path_in_repo="runs/" + os.environ.get("RUNPOD_POD_ID", "local") + "_ql7",
                  ignore_patterns=["*.pt", "*.safetensors", "**/trl/**"])
print("pushed ->", repo, flush=True)
PY
push() { python /workspace/push_ql7.py >/dev/null 2>&1 || true; }
push_retry() {
  for i in 1 2 3 4 5; do
    python /workspace/push_ql7.py && return 0
    echo "push attempt $i failed"; sleep $(( i * 20 ))
  done
  echo "push failed 5x -- reports stay on the pod disk ($OUT)"; return 1
}
# chain <gpu> <name> <cmd...> [-- <name> <cmd...>]...: sequential jobs on one GPU
chain() {
  local gpus=$1; shift
  ( while [ $# -gt 0 ]; do
      local name=$1; shift; local cmd=()
      while [ $# -gt 0 ] && [ "$1" != "--" ]; do cmd+=("$1"); shift; done
      [ $# -gt 0 ] && shift
      local t=$(date +%s)
      echo "=== $(date -u +%H:%M:%S) [$name] gpu=$gpus ${cmd[*]}"
      CUDA_VISIBLE_DEVICES=$gpus "${cmd[@]}" > "$OUT/$name.log" 2>&1
      echo "$name rc=$? wall=$(( $(date +%s) - t ))s" | tee -a $OUT/STAGES; push
    done ) &
}

if [ -f $OUT/DONE ] && [ "${NO_STOP:-0}" != 1 ]; then push_retry; stop_pod; exec sleep infinity; fi
trap 'date -u > $OUT/DONE; push_retry; stop_pod' EXIT

echo "=== $(date -u +%H:%M:%S) setup (stage $STAGE)"
nvidia-smi --query-gpu=name,memory.total --format=csv | tee $OUT/gpu.txt
# vLLM pins its own torch; install it first so training and rollouts share one torch
pip install -q vllm 2>&1 | tail -2
pip install -q "transformers>=4.56" datasets accelerate trl huggingface_hub scipy 2>&1 | tail -2
python -c "import torch, transformers, vllm, trl; print('torch', torch.__version__, 'tf', \
transformers.__version__, 'vllm', vllm.__version__, 'trl', trl.__version__)" | tee $OUT/versions.txt
cd /workspace && rm -rf qweave && git clone -q --depth 1 -b "${BRANCH:-claude/ql7-rl-cost}" \
    https://github.com/tonythetiger168/qweave && cd qweave
git log --oneline -1 | tee $OUT/commit.txt
python phase4_llm/validate_ql7.py > $OUT/validate_ql7.log 2>&1; tail -1 $OUT/validate_ql7.log
push

T=phase4_llm/ql7_grpo_train.py
M=${MODEL:-Qwen/Qwen3-0.6B-Base}
if [ "$STAGE" = Q ]; then
  S=${STEPS:-10}
  C="--model $M --steps $S --eval-every 5 --test-limit 200"
  chain 0 \
    Q_smoke     python $T --smoke --steps 3 --prompts 4 --group 4 --max-new 8 \
                  --select-prompts sqa --select-layers greedy_swap --layer-k 2 \
                  --probe-every 2 --out $OUT/Q_smoke -- \
    Q_sync      python phase4_llm/ql7_sync_check.py --model $M --out $OUT/Q_sync.json -- \
    base_s0     python $T $C --out $OUT/base_s0 -- \
    qp_sqa_s0   python $T $C --select-prompts sqa --out $OUT/qp_sqa_s0 -- \
    ql_sqa_s0   python $T $C --select-layers sqa --layer-k 14 --out $OUT/ql_sqa_s0 -- \
    trl_ref_s0  python phase4_llm/ql7_trl_ref.py --model $M --steps $S --out $OUT/trl_ref_s0
  wait
elif [ "$STAGE" = A1 ]; then
  # 1x H100: the named baseline + the TRL speed reference. Its logs alone give
  # the phase fractions (Amdahl budget) and the zero-signal persistence bound
  # (ql7_gate.py). If that bound cannot reach 0.75x, stop here: negative result.
  S=${STEPS:-200}
  C="--model $M --steps $S --eval-every 20"
  chain 0 \
    base_s0 python $T $C --seed 0 --out $OUT/base_s0 -- \
    trl_ref_s0 python phase4_llm/ql7_trl_ref.py --model $M --steps $S --out $OUT/trl_ref_s0
  wait
elif [ "$STAGE" = A2 ]; then
  # 4x H100: two more baseline seeds + every selection arm (gate needs A1's base_s0 too)
  S=${STEPS:-200}
  C="--model $M --steps $S --eval-every 20"
  chain 0 \
    base_s1 python $T $C --seed 1 --out $OUT/base_s1 -- \
    base_s2 python $T $C --seed 2 --out $OUT/base_s2 -- \
    dyn_s0 python $T $C --seed 0 --dynamic-sampling --out $OUT/dyn_s0 -- \
    qp_random_s0 python $T $C --seed 0 --select-prompts random --out $OUT/qp_random_s0
  # QUBO prompt selection: strong classical solver vs SQA, 3 seeds each
  chain 1 \
    qp_greedy_swap_s0 python $T $C --seed 0 --select-prompts greedy_swap --out $OUT/qp_greedy_swap_s0 -- \
    qp_greedy_swap_s1 python $T $C --seed 1 --select-prompts greedy_swap --out $OUT/qp_greedy_swap_s1 -- \
    qp_greedy_swap_s2 python $T $C --seed 2 --select-prompts greedy_swap --out $OUT/qp_greedy_swap_s2
  chain 2 \
    qp_sqa_s0 python $T $C --seed 0 --select-prompts sqa --out $OUT/qp_sqa_s0 -- \
    qp_sqa_s1 python $T $C --seed 1 --select-prompts sqa --out $OUT/qp_sqa_s1 -- \
    qp_sqa_s2 python $T $C --seed 2 --select-prompts sqa --out $OUT/qp_sqa_s2
  # layer-block selection (k = 14 of 28 blocks): separable vs interacting vs random
  chain 3 \
    ql_topk_s0 python $T $C --seed 0 --select-layers topk --layer-k 14 --out $OUT/ql_topk_s0 -- \
    ql_greedy_swap_s0 python $T $C --seed 0 --select-layers greedy_swap --layer-k 14 \
        --out $OUT/ql_greedy_swap_s0 -- \
    ql_sqa_s0 python $T $C --seed 0 --select-layers sqa --layer-k 14 --out $OUT/ql_sqa_s0 -- \
    ql_random_s0 python $T $C --seed 0 --select-layers random --layer-k 14 --out $OUT/ql_random_s0
  wait
else
  echo "stage $STAGE is not costed yet: run stages A1/A2 first"; exit 2
fi
python phase4_llm/ql7_gate.py $OUT --out $OUT/gate.json 2>&1 | tee $OUT/gate.txt
echo "=== $(date -u +%H:%M:%S) stage $STAGE finished"
cat $OUT/STAGES
