#!/bin/bash
# qweave QL6 -- batch-1 decode speed + long-context prefill, on ONE H100.
#
#   STAGE=S  speed only, no QL5 result needed (~2.5-3.5 h, ~US$8-15):
#            0.6B + 4B decode: dense vs shape-matched low-rank copies (keep
#            0.9 = the QL5 plan, 0.8, 0.6), +/- gate/up fusion, +/- torchao
#            INT8, eager and CUDA-graph engines; lossless speculative decoding
#            for 4B with off-the-shelf drafts (Qwen3-0.6B-Base, 1.7B-Base) and a
#            quickly healed QL-compressed 0.6B draft; QSA vs SDPA prefill at
#            8K/16K/32K; vLLM dense reference in its own venv.
#   STAGE=Q  same measurements on QL5's real models (needs QL5_DIR, i.e. run on
#            the QL5 pod after its stage A, ~1 h extra)
#   STAGE=K  Triton low-rank decode kernels (ql6_kernels.py) after stage S
#            showed torch low rank slower than dense: per-layer latency
#            microbench, 0.6B + 4B decode with/without kernels (CUDA graphs
#            only), 4B speculative rerun with the bf16-aware lossless check,
#            vLLM references (~2-2.5 h, ~US$8-10)
#   QUICK=1  minutes-long version (checks the pod, ~US$1)
#
# Env:  HF_TOKEN (RunPod secret) -> reports pushed to <hf-user>/qweave-runs
#       BRANCH (default claude/ql6-decode), QL5_DIR (default /workspace/ql5_runs/stageA),
#       VLLM=0 to skip the vLLM reference, RUNPOD_API_KEY (optional, stops pod)
# Results: /workspace/ql6_runs/*  and  hf://datasets/<user>/qweave-runs/runs/<pod>_ql6/
# Shutdown contract: when the run ends (success, failure or timeout) the
# reports are pushed to HF with retries, THEN the pod is stopped (retried);
# a container restarted after a finished run goes straight to push + stop.
set -u
export PIP_ROOT_USER_ACTION=ignore HF_HOME=/workspace/hf TOKENIZERS_PARALLELISM=false
STAGE=${STAGE:-S}; Q=${QUICK:-0}
OUT=/workspace/ql6_runs/stage$STAGE; mkdir -p $OUT
exec > >(tee -a $OUT/run.log) 2>&1
POD=${RUNPOD_POD_ID:-local}
QL5_DIR=${QL5_DIR:-/workspace/ql5_runs/stageA}

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

cat > /workspace/push_ql6.py <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = api.whoami()["name"] + "/qweave-runs"
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(folder_path="/workspace/ql6_runs", repo_id=repo, repo_type="dataset",
                  path_in_repo="runs/" + os.environ.get("RUNPOD_POD_ID", "local") + "_ql6",
                  ignore_patterns=["*.pt", "**/tokenizer/**", "**/tiny_base/**"])
print("pushed ->", repo, flush=True)
PY
push() { python /workspace/push_ql6.py >/dev/null 2>&1 || true; }
push_retry() { # final upload: retried, and always before the pod is stopped
  for i in 1 2 3 4 5; do
    python /workspace/push_ql6.py && return 0
    echo "push attempt $i failed"; sleep $(( i * 20 ))
  done
  echo "push failed 5x -- reports stay on the pod disk ($OUT)"; return 1
}
step() {   # step <name> <cmd...>: sequential, logged, pushed
  local name=$1; shift; local t=$(date +%s)
  echo "=== $(date -u +%H:%M:%S) [$name] $*"
  "$@" > "$OUT/$name.log" 2>&1
  echo "$name rc=$? wall=$(( $(date +%s) - t ))s" | tee -a $OUT/STAGES
  grep -E "QL6 DECODE|tok_s|speedup" "$OUT/$name.log" | tail -4
  push
}
# container restarted after a finished run: never rerun, just push + stop
if [ -f $OUT/DONE ] && [ "${KEEP_POD:-0}" != 1 ]; then push_retry; stop_pod; exec sleep infinity; fi
[ "${KEEP_POD:-0}" = 1 ] && trap 'date -u > $OUT/DONE; push_retry' EXIT \
                         || trap 'date -u > $OUT/DONE; push_retry; stop_pod' EXIT

echo "=== $(date -u +%H:%M:%S) setup (stage $STAGE, quick=$Q)"
nvidia-smi --query-gpu=name,memory.total,clocks.max.memory --format=csv | tee $OUT/gpu.txt
pip install -q "transformers>=4.56" datasets accelerate huggingface_hub torchao 2>&1 | tail -3
python -c "import torch, transformers; print('torch', torch.__version__, 'tf', transformers.__version__)
try:
    import torchao; print('torchao', torchao.__version__)
except Exception as e: print('torchao unavailable', e)" | tee $OUT/versions.txt
cd /workspace && rm -rf qweave && git clone -q --depth 1 -b "${BRANCH:-claude/ql6-decode}" \
    https://github.com/tonythetiger168/qweave && cd qweave
git log --oneline -1 | tee $OUT/commit.txt
python phase4_llm/validate_ql6.py > $OUT/validate_ql6.log 2>&1; tail -1 $OUT/validate_ql6.log
python phase4_llm/ql6_decode.py --roofline --out $OUT > $OUT/roofline.log 2>&1
push

D=phase4_llm/ql6_decode.py
M06=Qwen/Qwen3-0.6B-Base; M17=Qwen/Qwen3-1.7B-Base; M4=Qwen/Qwen3-4B-Base
if [ "$Q" = 1 ]; then
  COMMON="--n-prompts 4 --new-tokens 32 --repeats 1"; PF=8192; HEAL=1e6
else
  COMMON="--n-prompts 24 --new-tokens 256 --repeats 2"; PF=8192,16384,32768; HEAL=20e6
fi

if [ "$STAGE" = K ]; then
  KC=$([ "$Q" = 1 ] && echo "$COMMON" || echo "--n-prompts 12 --new-tokens 256 --repeats 1")
  step K0_draft06 python phase4_llm/ql5_compress.py --model $M06 --total-saving 0.2 \
      --embed-keep 0.5 --heal-tokens $HEAL --batch 16 --save-model --out $OUT/draft06
  step K1_06 python $D --model $M06 $KC --synthetic-keep 0.9,0.6 --embed-keep 0.5 \
      --engines compiled --triton --layer-bench --prefill "" --out $OUT/k06
  step K2_4b python $D --model $M4 $KC --synthetic-keep 0.9,0.6 --embed-keep 0.5 \
      --engines compiled --triton --layer-bench --prefill "" \
      --drafts $M06,$OUT/draft06/model.pt --gamma 3,5 --out $OUT/k4b
elif [ "$STAGE" = S ]; then
  # A QL-compressed 0.6B to use as a REAL draft for 4B (quality matters for
  # acceptance only, so a short 20M-token heal is enough; ~20 min).
  step S0_draft06 python phase4_llm/ql5_compress.py --model $M06 --total-saving 0.2 \
      --embed-keep 0.5 --heal-tokens $HEAL --batch 16 --save-model --out $OUT/draft06
  # 0.6B: route L1 speed (shape-matched copies), INT8, prefill
  step S1_06 python $D --model $M06 $COMMON --synthetic-keep 0.9,0.8,0.6 --embed-keep 0.5 \
      --engines eager,compiled --int8 --prefill $PF --out $OUT/s06
  # 4B: same + lossless speculative decoding with three drafts
  step S2_4b python $D --model $M4 $COMMON --synthetic-keep 0.9,0.8,0.6 --embed-keep 0.5 \
      --engines eager,compiled --int8 --prefill $PF \
      --drafts $M06,$M17,$OUT/draft06/model.pt --gamma 3,5 --out $OUT/s4b
else
  [ -f $QL5_DIR/A1_main/model.pt ] || { echo "no QL5 model at $QL5_DIR/A1_main"; exit 1; }
  pass() { python -c "import json,sys; print(int(json.load(open(sys.argv[1]))['PASS']))" \
             $QL5_DIR/$1/gate.json 2>/dev/null || echo 0; }
  step Q1_06 python $D --model $M06 $COMMON --synthetic-keep "" \
      --compressed $QL5_DIR/A1_main/model.pt --quality-pass $(pass A1_main) --int8 \
      --engines eager,compiled --prefill "" \
      --out $OUT/q06
  step Q2_4b python $D --model $M4 $COMMON --synthetic-keep "" --engines eager \
      --drafts $M06,$QL5_DIR/A1_main/model.pt --gamma 3,5 --prefill "" --out $OUT/q4b
fi

# dense vLLM reference (different engine, so not a gate baseline; it says how
# far HF + CUDA graphs is from a production engine). Own venv: vLLM pins torch.
if [ "${VLLM:-1}" = 1 ]; then
  cat > /workspace/vllm_ref.py <<'PY'
import json, sys, time
from vllm import LLM, SamplingParams
sys.path.insert(0, "/workspace/qweave")
from phase4_llm.ql6_decode import PROMPTS
model, n_new, n_p, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.5, max_model_len=4096)
def run(n):
    sp = SamplingParams(temperature=0, max_tokens=n, min_tokens=n, ignore_eos=True)
    ts = []
    for p in PROMPTS[:n_p]:
        t = time.perf_counter(); llm.generate([p], sp, use_tqdm=False); ts.append(time.perf_counter() - t)
    return ts
run(8); t1 = run(1); tn = run(n_new)
per = [(b - a) / (n_new - 1) for a, b in zip(t1, tn)]
r = {"model": model, "engine": "vllm", "tok_s": round(len(per) / sum(per), 1)}
print(r); json.dump(r, open(out, "w"))
PY
  python -m venv /workspace/vllm-env && /workspace/vllm-env/bin/pip install -q vllm ninja \
      2>&1 | tail -2
  NP=$([ "$Q" = 1 ] && echo 4 || echo 24); NT=$([ "$Q" = 1 ] && echo 32 || echo 256)
  for m in $M06 $M4; do
    # vLLM JIT-builds kernels with ninja, which must be on PATH
    PATH=/workspace/vllm-env/bin:$PATH step V_$(basename $m) \
        /workspace/vllm-env/bin/python /workspace/vllm_ref.py $m $NT $NP \
        $OUT/vllm_$(basename $m).json
  done
fi
echo "=== $(date -u +%H:%M:%S) stage $STAGE finished"
cat $OUT/STAGES
