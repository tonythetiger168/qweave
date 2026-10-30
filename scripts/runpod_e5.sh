#!/bin/bash
# Phase E5 on a RunPod 1x H100 pod: GPU state-vector + decoder benchmark.
# Usage (pod start command):  curl -sL <raw url>/scripts/runpod_e5.sh | bash
# Env: HF_TOKEN (RunPod secret), optional NO_STOP=1, CAP (default 80m).
set -u
BRANCH=${BRANCH:-claude/e5-gpu-bench}
CAP=${CAP:-80m}
OUT=/workspace/e5_runs
mkdir -p $OUT
exec > >(tee -a $OUT/run.log) 2>&1
echo "=== start $(date -u) pod=${RUNPOD_POD_ID:-local}"
nvidia-smi --query-gpu=name,memory.total --format=csv

cd /workspace
[ -d qweave ] || git clone -q --depth 1 -b $BRANCH https://github.com/tonythetiger168/qweave.git
cd qweave

pip install -q numpy scipy stim pymatching ldpc huggingface_hub 2>&1 | tail -2
pip install -q qiskit qiskit-aer-gpu 2>&1 | tail -2 || echo "(aer-gpu install failed)"
pip install -q cudaq-qec 2>&1 | tail -2 || echo "(cudaq-qec install failed)"
python -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"

OUT=$OUT timeout $CAP python scripts/e5_gpu_bench.py
echo "bench rc=$? $(date -u)"

python - <<'PY'
import os
from huggingface_hub import HfApi
pod = os.environ.get("RUNPOD_POD_ID", "local")
HfApi(token=os.environ["HF_TOKEN"]).upload_folder(
    repo_id="chienhsinlin/qweave-runs", repo_type="dataset",
    folder_path="/workspace/e5_runs", path_in_repo=f"runs/{pod}_e5")
print("uploaded to runs/%s_e5" % pod)
PY

if [ "${NO_STOP:-0}" != "1" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  runpodctl stop pod $RUNPOD_POD_ID
fi
