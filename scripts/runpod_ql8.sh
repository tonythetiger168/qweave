#!/bin/bash
# qweave QL8 -- DiracLM release measurements on ONE RunPod pod. Nothing here
# publishes a model: stage F only BUILDS the release folder; uploading is a
# separate, approved step (python -m phase4_llm.release.package push ...).
#
#   QUICK=1   1x H100, ~20 min (~US$2): CPU checks, then the eval harness on
#             Qwen3-0.6B with --limit 20 per task (vLLM + HF backends)
#   STAGE=S1  1x H100, ~1.5 h (~US$6): QSA gate re-measured on each release base
#             (bench_ql3.py: entanglement, kernel 16K/32K, 16K-window PPL)
#   STAGE=S2  1x H100, ~2-3 h (~US$8-12, cap 4 h): MATH-500 / GPQA-D / BBH for the controls:
#             Qwen3-4B-Base (no thinking), Qwen3-4B (thinking, reference only),
#             Phi-4-mini-instruct
#   (RL)      the DiracLM-4B weights come from QL7 stage B (scripts/runpod_ql7.sh,
#             4x H100), costed in docs/QL8_RELEASE_PLAN.md, run only after QL7 stage A
#   STAGE=F   1x H100, ~2 h (~US$8): eval the trained checkpoint (MODEL_DIR) with
#             the S2 protocol, compare against Qwen3-4B-Base, build the release folder
#
# Env: HF_TOKEN (RunPod secret; reports -> <hf-user>/qweave-runs, private)
#      BRANCH (default main), RUNPOD_API_KEY (optional, stops pod), NO_STOP=1
#      MODEL_DIR (stage F: trained checkpoint dir or hub id)
# Shutdown contract (same as runpod_ql7.sh): reports pushed with retries, THEN
# the pod is stopped. Wrap in `timeout <minutes>m`.
set -u
export PIP_ROOT_USER_ACTION=ignore HF_HOME=/workspace/hf TOKENIZERS_PARALLELISM=false
Q=${QUICK:-0}; STAGE=${STAGE:-S1}; [ "$Q" = 1 ] && STAGE=Q
ROOT=/workspace/ql8; OUT=$ROOT/stage$STAGE; mkdir -p $OUT
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
cat > /workspace/push_ql8.py <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = api.whoami()["name"] + "/qweave-runs"
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(folder_path="/workspace/ql8", repo_id=repo, repo_type="dataset",
                  path_in_repo="runs/" + os.environ.get("RUNPOD_POD_ID", "local") + "_ql8",
                  ignore_patterns=["*.pt", "*.safetensors", "**/release/**"])
print("pushed ->", repo, flush=True)
PY
push() { python /workspace/push_ql8.py >/dev/null 2>&1 || true; }
push_retry() {
  for i in 1 2 3 4 5; do python /workspace/push_ql8.py && return 0
    echo "push attempt $i failed"; sleep $(( i * 20 )); done
  echo "push failed 5x -- reports stay on the pod disk ($ROOT)"; return 1
}
step() {   # step <name> <cmd...>
  local name=$1; shift; local t=$(date +%s)
  echo "=== $(date -u +%H:%M:%S) [$name] $*"
  "$@" > "$OUT/$name.log" 2>&1
  echo "$name rc=$? wall=$(( $(date +%s) - t ))s" | tee -a $OUT/STAGES; push
}

if [ -f $OUT/DONE ] && [ "${NO_STOP:-0}" != 1 ]; then push_retry; stop_pod; exec sleep infinity; fi
trap 'date -u > $OUT/DONE; push_retry; stop_pod' EXIT

echo "=== $(date -u +%H:%M:%S) setup (stage $STAGE)"
nvidia-smi --query-gpu=name,memory.total --format=csv | tee $OUT/gpu.txt
pip install -q vllm 2>&1 | tail -2
pip install -q "transformers>=4.56" datasets accelerate huggingface_hub scipy math-verify 2>&1 | tail -2
cd /workspace && rm -rf qweave && git clone -q --depth 1 -b "${BRANCH:-main}" \
    https://github.com/tonythetiger168/qweave && cd qweave
git rev-parse HEAD > $OUT/commit.txt; pip list 2>/dev/null | grep -Ei "^(torch|vllm|transformers) " > $OUT/versions.txt
EV="python -m phase4_llm.release.reasoning_eval"

case $STAGE in
Q)
  step validate python phase4_llm/validate_ql8.py
  step eval_vllm $EV run --model Qwen/Qwen3-0.6B --limit 20 --k 2 --max-tokens 2048 \
      --max-model-len 4096 --math-verify --out $OUT/eval/qwen3-0.6b
  step eval_hf $EV run --model Qwen/Qwen3-0.6B --backend hf --tasks math500 --limit 4 --k 1 \
      --max-tokens 512 --out $OUT/eval/qwen3-0.6b-hf
  step compare $EV compare $OUT/eval/qwen3-0.6b $OUT/eval/qwen3-0.6b-hf --out $OUT/compare.json
  ;;
S1)
  step ql3_4b python phase4_llm/bench_ql3.py --model Qwen/Qwen3-4B-Base --topk 8 --out $ROOT/ql3_4b
  step ql3_phi4mini python phase4_llm/bench_ql3.py --model microsoft/Phi-4-mini-instruct \
      --topk 8 --out $ROOT/ql3_phi4mini
  ;;
S2)
  step eval_qwen3_4b_base $EV run --model Qwen/Qwen3-4B-Base --no-thinking --max-tokens 4096 \
      --max-model-len 8192 --math-verify --out $ROOT/eval/qwen3-4b-base
  step eval_phi4mini $EV run --model microsoft/Phi-4-mini-instruct --no-thinking --max-tokens 4096 \
      --max-model-len 8192 --math-verify --out $ROOT/eval/phi4-mini
  step eval_qwen3_4b $EV run --model Qwen/Qwen3-4B --thinking --max-tokens 32768 \
      --max-model-len 40960 --math-verify --out $ROOT/eval/qwen3-4b
  ;;
F)
  : "${MODEL_DIR:?set MODEL_DIR to the trained DiracLM checkpoint}"
  # PRIOR_RUNS="runs/<pod>_ql8 ..." pulls S1/S2 reports (gate files, control evals)
  # and QL7/QL5 gate files from <hf-user>/qweave-runs into $ROOT first
  for r in ${PRIOR_RUNS:-}; do python - "$r" <<'PY'
import os, shutil, sys
from huggingface_hub import HfApi, snapshot_download
api = HfApi(token=os.environ["HF_TOKEN"]); repo = api.whoami()["name"] + "/qweave-runs"
d = snapshot_download(repo, repo_type="dataset", allow_patterns=[sys.argv[1] + "/**"])
shutil.copytree(os.path.join(d, sys.argv[1]), "/workspace/ql8", dirs_exist_ok=True)
print("pulled", sys.argv[1])
PY
  done
  python - <<'PY'   # base model license + notice go into the release
import os, shutil
from huggingface_hub import hf_hub_download
for fn in ("LICENSE", "NOTICE"):
    try:
        shutil.copy(hf_hub_download("Qwen/Qwen3-4B-Base", fn), os.environ["MODEL_DIR"])
    except Exception as e:  # noqa: BLE001
        print("no", fn, e)
PY
  step eval_diraclm_4b $EV run --model "$MODEL_DIR" --no-thinking --max-tokens 4096 \
      --max-model-len 8192 --math-verify --out $ROOT/eval/diraclm-4b
  step compare $EV compare $ROOT/eval/qwen3-4b-base $ROOT/eval/diraclm-4b --out $ROOT/eval/compare_4b.json
  step build python -m phase4_llm.release.package build \
      --recipe phase4_llm/release/recipes/diraclm-4b.json --artifacts $ROOT \
      --weights "$MODEL_DIR" --base-eval $ROOT/eval/qwen3-4b-base \
      --model-eval $ROOT/eval/diraclm-4b --out $ROOT/release \
      --repro "STAGE=F MODEL_DIR=<ckpt> bash scripts/runpod_ql8.sh   # commit $(cat $OUT/commit.txt)"
  ;;
*) echo "unknown STAGE=$STAGE"; exit 2 ;;
esac
echo "=== $(date -u +%H:%M:%S) stage $STAGE finished"; cat $OUT/STAGES
