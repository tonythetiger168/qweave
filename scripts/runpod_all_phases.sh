#!/bin/bash
# qweave -- run ALL phases on one RunPod GPU pod, upload reports, stop the pod.
#
# Phases I-III (simulators) validation, QL1 validation, QL2 benchmarks,
# QL1.5 heal runs (dense baseline + TT compression ablations), QL3 QSA
# benchmarks, QL4 open benchmark (dense vs compressed).
#
# Env:  HF_TOKEN  (RunPod secret)  -> reports pushed to <hf-user>/qweave-runs
#       BRANCH    (default main)
#       QUICK=1   short smoke version of every stage
# Results: /workspace/qweave_runs/*  and  hf://datasets/<user>/qweave-runs/runs/<pod>/
set -u
export PIP_ROOT_USER_ACTION=ignore HF_HOME=/workspace/hf TOKENIZERS_PARALLELISM=false
OUT=/workspace/qweave_runs; mkdir -p $OUT
exec > >(tee -a $OUT/run.log) 2>&1
POD=${RUNPOD_POD_ID:-local}

stop_pod() {
  [ -n "${RUNPOD_API_KEY:-}" ] && curl -sS -X POST "https://rest.runpod.io/v1/pods/$POD/stop" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" >/dev/null 2>&1
  command -v runpodctl >/dev/null && runpodctl stop pod "$POD"
}

cat > /workspace/push.py <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = api.whoami()["name"] + "/qweave-runs"
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(folder_path="/workspace/qweave_runs", repo_id=repo, repo_type="dataset",
                  path_in_repo="runs/" + os.environ.get("RUNPOD_POD_ID", "local"),
                  ignore_patterns=["*.pt", "**/tokenizer/**"])
print("pushed ->", repo, flush=True)
PY

stage() {   # stage <name> <timeout> <cmd...>   (ONLY="a b c" runs a subset)
  local name=$1 to=$2; shift 2
  if [ -n "${ONLY:-}" ] && ! [[ " $ONLY " == *" $name "* ]]; then return; fi
  echo "=== $(date -u +%H:%M:%S) [$name] $*"
  local t=$(date +%s)
  timeout "$to" "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  tail -25 "$OUT/$name.log"
  echo "$name rc=$rc wall=$(( $(date +%s) - t ))s" | tee -a $OUT/STAGES
  python /workspace/push.py >/dev/null 2>&1 || true
}

trap 'date -u > $OUT/DONE; python /workspace/push.py || true; sleep 30; stop_pod' EXIT

echo "=== $(date -u +%H:%M:%S) setup"
nvidia-smi --query-gpu=name,memory.total --format=csv | tee $OUT/gpu.txt
pip install -q "transformers>=4.56" datasets accelerate "lm_eval>=0.4.8" huggingface_hub \
    scipy stim pymatching qiskit qiskit-aer 2>&1 | tail -3
python -c "import torch, transformers; print('torch', torch.__version__, 'tf', transformers.__version__)" \
    | tee $OUT/versions.txt
cd /workspace && rm -rf qweave && git clone -q --depth 1 -b "${BRANCH:-main}" \
    https://github.com/tonythetiger168/qweave && cd qweave
git log --oneline -1 | tee $OUT/commit.txt
python /workspace/push.py || echo "first push failed"

Q=${QUICK:-0}
if [ "$Q" = 1 ]; then S=60; W=4096; NW=2; LIM=20; else S=1500; W=16384; NW=8; LIM=500; fi
M=Qwen/Qwen3-0.6B

# ---- Phases I-III + QL1 ----
stage p1_validate   20m python phase1_basic/validate_p0.py
stage p2_validate   20m python phase2_noise/validate_p0b.py
stage p3_validate   30m python phase3_scale/validate_p0c.py
stage ql1_validate  20m python phase4_llm/validate_ql1.py
stage ql15_smoke    20m python phase4_llm/heal_finetune.py --smoke --steps 30 --out $OUT/ql15_smoke

# ---- QL2: kernels + allocator ----
stage ql2_bench     40m python phase4_llm/bench_ql2.py --out $OUT/ql2 $( [ "$Q" = 1 ] && echo --quick )
stage ql2_layers    20m python phase4_llm/bench_ql2.py --layers-only --out $OUT/ql2_layers

# ---- QL1.5: heal runs on Qwen3-0.6B ----
stage ql15_dense    30m python phase4_llm/heal_finetune.py --model $M --target-saving 0 --heal-scope all \
    --steps $(( S / 10 )) --batch 8 --seq 512 --out $OUT/ql15_dense
stage ql15_s30_kd   90m python phase4_llm/heal_finetune.py --model $M --target-saving 0.3 \
    --steps $S --batch 8 --seq 512 --lr 2e-5 --distill --save-model --out $OUT/ql15_s30_kd
stage ql15_dense_ft 60m python phase4_llm/heal_finetune.py --model $M --target-saving 0 --heal-scope all \
    --steps $S --batch 8 --seq 512 --lr 2e-5 --out $OUT/ql15_dense_ft
stage ql15_s30_ce   90m python phase4_llm/heal_finetune.py --model $M --target-saving 0.3 \
    --steps $S --batch 8 --seq 512 --lr 2e-5 --out $OUT/ql15_s30_ce
stage ql15_s15_kd   90m python phase4_llm/heal_finetune.py --model $M --target-saving 0.15 \
    --steps $S --batch 8 --seq 512 --lr 2e-5 --distill --save-model --out $OUT/ql15_s15_kd
stage ql15_s20_kd   90m python phase4_llm/heal_finetune.py --model $M --target-saving 0.2 \
    --steps $S --batch 8 --seq 512 --lr 2e-5 --distill --out $OUT/ql15_s20_kd
stage ql15_tt_raw   60m python phase4_llm/heal_finetune.py --model $M --format tt --alloc greedy \
    --target-saving 0.3 --steps $(( S / 15 )) --batch 8 --seq 512 --out $OUT/ql15_tt_raw
stage ql15_mpo_old  45m python phase4_llm/heal_finetune.py --model $M --format mpo \
    --chi 32 --compress-frac 0.5 --steps $(( S / 5 )) --batch 8 --seq 512 \
    --targets q_proj,o_proj,gate_proj,up_proj --out $OUT/ql15_mpo_old

# ---- QL3: entanglement diagnostics + QSA ----
stage ql3_bench     90m python phase4_llm/bench_ql3.py --model $M --win $W --nwin $NW --out $OUT/ql3

# ---- QL4: open benchmark ----
stage ql4_eval     150m python phase4_llm/eval/run_eval.py --dense $M \
    --compressed $OUT/ql15_s30_kd/model.pt --limit $LIM --out $OUT/ql4
stage ql4_eval_s15 120m python phase4_llm/eval/run_eval.py --skip-dense \
    --compressed $OUT/ql15_s15_kd/model.pt --limit $LIM --out $OUT/ql4_s15

echo "=== $(date -u +%H:%M:%S) all stages finished"
cat $OUT/STAGES
