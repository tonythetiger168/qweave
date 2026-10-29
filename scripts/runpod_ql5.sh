#!/bin/bash
# qweave QL5 -- knowledge-preserving compression, on ONE 4x H100 RunPod pod.
#
#   STAGE=A  Qwen3-0.6B-Base: main recipe + dense same-recipe control +
#            ablations, then full-test-set MMLU/GSM8K gate   (~3.5 h, ~US$45-70)
#   STAGE=B  Qwen3-4B-Base: main recipe + dense control, 4-GPU data parallel,
#            then the gate                                    (~7 h,  ~US$85-130)
#   STAGE=C  Qwen3-0.6B-Base follow-up to A (stage A models were deleted with its
#            pod): retrain the 6 ablations + a -5/-10/-15 % Pareto sweep, all at
#            25M heal tokens, each followed by its MMLU/GSM8K eval on the same GPU
#            (~3.5-4 h, ~US$55-65 on Secure 4x H100; use a 270m cap)
#   QUICK=1  minutes-long version of the chosen stage (checks the pod, ~US$3)
#
# Env:  HF_TOKEN (RunPod secret) -> reports pushed to <hf-user>/qweave-runs
#       BRANCH (default claude/ql5-quality), RUNPOD_API_KEY (optional, stops pod)
#       NO_STOP=1 keeps the pod up at the end (use it for QUICK=1)
# Results: /workspace/ql5_runs/*  and  hf://datasets/<user>/qweave-runs/runs/<pod>/
# Shutdown contract: when the run ends (success, failure or timeout) the
# reports are pushed to HF with retries, THEN the pod is stopped (retried);
# a container restarted after a finished run goes straight to push + stop.
set -u
export PIP_ROOT_USER_ACTION=ignore HF_HOME=/workspace/hf TOKENIZERS_PARALLELISM=false
STAGE=${STAGE:-A}; Q=${QUICK:-0}
OUT=/workspace/ql5_runs/stage$STAGE; mkdir -p $OUT
exec > >(tee -a $OUT/run.log) 2>&1
POD=${RUNPOD_POD_ID:-local}

stop_pod() {   # NO_STOP=1 keeps the pod running (e.g. after the QUICK check)
  [ "${NO_STOP:-0}" = 1 ] && return
  for i in $(seq 1 20); do    # retried: the pod dies as soon as one call lands
    echo "stop_pod attempt $i $(date -u +%H:%M:%S)"
    [ -n "${RUNPOD_API_KEY:-}" ] && curl -sS -X POST "https://rest.runpod.io/v1/pods/$POD/stop" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" >/dev/null 2>&1
    command -v runpodctl >/dev/null && runpodctl stop pod "$POD" >/dev/null 2>&1
    sleep 30
  done
  echo "stop_pod: no stop call succeeded (no RUNPOD_API_KEY / runpodctl?)"
}

cat > /workspace/push_ql5.py <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = api.whoami()["name"] + "/qweave-runs"
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(folder_path="/workspace/ql5_runs", repo_id=repo, repo_type="dataset",
                  path_in_repo="runs/" + os.environ.get("RUNPOD_POD_ID", "local") + "_ql5",
                  ignore_patterns=["*.pt", "**/tokenizer/**", "**/tiny_base/**"])
print("pushed ->", repo, flush=True)
PY
push() { python /workspace/push_ql5.py >/dev/null 2>&1 || true; }
push_retry() { # final upload: retried, and always before the pod is stopped
  for i in 1 2 3 4 5; do
    python /workspace/push_ql5.py && return 0
    echo "push attempt $i failed"; sleep $(( i * 20 ))
  done
  echo "push failed 5x -- reports stay on the pod disk ($OUT)"; return 1
}

# job <gpus> <name> <cmd...>: run in the background on the given GPUs
job() {
  local gpus=$1 name=$2; shift 2
  echo "=== $(date -u +%H:%M:%S) [$name] gpu=$gpus $*"
  ( local t=$(date +%s)
    CUDA_VISIBLE_DEVICES=$gpus "$@" > "$OUT/$name.log" 2>&1
    echo "$name rc=$? wall=$(( $(date +%s) - t ))s" | tee -a $OUT/STAGES; push ) &
}
# chain <gpus> <name> <cmd...> [-- <name> <cmd...>]...: sequential jobs on one GPU set
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

# container restarted after a finished run: never rerun, just push + stop
if [ -f $OUT/DONE ] && [ "${NO_STOP:-0}" != 1 ]; then push_retry; stop_pod; exec sleep infinity; fi
trap 'date -u > $OUT/DONE; push_retry; stop_pod' EXIT    # stop_pod honours NO_STOP

echo "=== $(date -u +%H:%M:%S) setup (stage $STAGE, quick=$Q)"
nvidia-smi --query-gpu=name,memory.total --format=csv | tee $OUT/gpu.txt
pip install -q "transformers>=4.56" datasets accelerate "lm_eval>=0.4.8" huggingface_hub \
    2>&1 | tail -3
python -c "import torch, transformers, lm_eval; print('torch', torch.__version__, \
'tf', transformers.__version__, 'lm_eval', lm_eval.__version__)" | tee $OUT/versions.txt
cd /workspace && rm -rf qweave && git clone -q --depth 1 -b "${BRANCH:-claude/ql5-quality}" \
    https://github.com/tonythetiger168/qweave && cd qweave
git log --oneline -1 | tee $OUT/commit.txt
python phase4_llm/validate_ql5.py > $OUT/validate_ql5.log 2>&1; tail -1 $OUT/validate_ql5.log
push

C=phase4_llm/ql5_compress.py
E=phase4_llm/eval/run_eval.py
G=phase4_llm/ql5_gate.py
if [ "$Q" = 1 ]; then TOK=2e6; ABL=1e6; LIM=40; else TOK=100e6; ABL=25e6; LIM=0; fi
GATE_TASKS=mmlu,gsm8k
ALL_TASKS=arc_easy,arc_challenge,hellaswag,piqa,winogrande,mmlu,gsm8k

ev() {   # ev <dir> <tasks>: evaluate <dir>/model.pt (full test sets) with per-sample logs
  python $E --skip-dense --compressed $1/model.pt --tasks $2 --limit $LIM --log-samples \
      --out $1/eval
}
gate() { # gate <dir>: dense-base eval vs <dir>
  python $G --dense $OUT/dense_base/eval/ql4_dense.json \
      --compressed $1/eval/ql4_compressed.json --ql5-report $1/ql5_report.json \
      --out $1/gate.json
}

if [ "$STAGE" = A ]; then
  M=Qwen/Qwen3-0.6B-Base
  # GPU0: main recipe (-20 % total: embedding keep 0.5 + Fisher-weighted layers), 100M tokens
  chain 0 \
    A1_main      python $C --model $M --total-saving 0.2 --embed-keep 0.5 --heal-tokens $TOK \
                   --eval-every 500 --quant int8,fp8 --save-model --out $OUT/A1_main -- \
    A1_eval      ev $OUT/A1_main $ALL_TASKS
  # GPU1: dense control, same data / tokens / KD, every param trainable (F3)
  chain 1 \
    A2_dense_ctl python $C --model $M --total-saving 0 --heal-scope all --heal-tokens $TOK \
                   --lr 2e-5 --save-model --out $OUT/A2_dense_ctl -- \
    A2_eval      ev $OUT/A2_dense_ctl $ALL_TASKS
  # GPU2: uncompressed reference eval, then 25M-token ablations (each vs A3_ref)
  chain 2 \
    A0_dense_eval python $E --dense $M --tasks $ALL_TASKS --limit $LIM --log-samples \
                   --out $OUT/dense_base/eval -- \
    A3_ref       python $C --model $M --heal-tokens $ABL --save-model --out $OUT/A3_ref -- \
    A3_nofisher  python $C --model $M --heal-tokens $ABL --alloc-weight none --save-model \
                   --out $OUT/A3_nofisher -- \
    A3_noembed   python $C --model $M --heal-tokens $ABL --embed-keep 1 --save-model \
                   --out $OUT/A3_noembed
  # GPU3: more 25M-token ablations
  chain 3 \
    A3_webonly   python $C --model $M --heal-tokens $ABL --calib-mix fineweb:1 \
                   --heal-mix fineweb:1 --source-weight "" --save-model --out $OUT/A3_webonly -- \
    A3_regauge   python $C --model $M --heal-tokens $ABL --regauge-every 200 --save-model \
                   --out $OUT/A3_regauge -- \
    A3_embed035  python $C --model $M --heal-tokens $ABL --embed-keep 0.35 --save-model \
                   --out $OUT/A3_embed035
  wait
  # ablation + quantised-copy evals (gate tasks only), 4 GPUs in parallel
  mkdir -p $OUT/A1_int8 && ln -sf $OUT/A1_main/model_int8.pt $OUT/A1_int8/model.pt && \
      ln -sfn $OUT/A1_main/tokenizer $OUT/A1_int8/tokenizer && \
      cp $OUT/A1_main/ql5_report.json $OUT/A1_int8/ql5_report.json
  chain 0 A1_int8_eval ev $OUT/A1_int8 $GATE_TASKS -- A3_ref_eval ev $OUT/A3_ref $GATE_TASKS
  chain 1 A3_nofisher_eval ev $OUT/A3_nofisher $GATE_TASKS -- \
          A3_noembed_eval ev $OUT/A3_noembed $GATE_TASKS
  chain 2 A3_webonly_eval ev $OUT/A3_webonly $GATE_TASKS
  chain 3 A3_regauge_eval ev $OUT/A3_regauge $GATE_TASKS -- \
          A3_embed035_eval ev $OUT/A3_embed035 $GATE_TASKS
  wait
  for d in A1_main A1_int8 A2_dense_ctl A3_ref A3_nofisher A3_noembed A3_webonly \
           A3_regauge A3_embed035; do
    echo "--- gate $d"; gate $OUT/$d 2>&1 | tail -3 | tee $OUT/$d.gate.txt
  done
elif [ "$STAGE" = C ]; then
  M=Qwen/Qwen3-0.6B-Base
  # tr <name> <flags...>: 25M-token run then its gate eval, as one chain segment
  # Sweep (S*) uses the no-Fisher allocation (best healed PPL in stage A) and keeps the
  # embedding's share of the saving at stage A's ~65 % (keep 0.881/0.755/0.629 ->
  # rank 896/768/640). Order = priority, in case the time cap hits.
  R="--model $M --heal-tokens $ABL --save-model"
  chain 0 \
    C_ref         python $C $R --out $OUT/C_ref -- \
    C_ref_eval    ev $OUT/C_ref $GATE_TASKS -- \
    S15           python $C $R --alloc-weight none --total-saving 0.15 --embed-keep 0.629 \
                    --out $OUT/S15 -- \
    S15_eval      ev $OUT/S15 $GATE_TASKS
  chain 1 \
    C_nofisher    python $C $R --alloc-weight none --quant int8 --out $OUT/C_nofisher -- \
    C_nofisher_eval ev $OUT/C_nofisher $GATE_TASKS -- \
    C_int8_link   bash -c "mkdir -p $OUT/C_nofisher_int8 && \
                    ln -sf $OUT/C_nofisher/model_int8.pt $OUT/C_nofisher_int8/model.pt && \
                    ln -sfn $OUT/C_nofisher/tokenizer $OUT/C_nofisher_int8/tokenizer && \
                    cp $OUT/C_nofisher/ql5_report.json $OUT/C_nofisher_int8/" -- \
    C_nofisher_int8_eval ev $OUT/C_nofisher_int8 $GATE_TASKS -- \
    S10           python $C $R --alloc-weight none --total-saving 0.10 --embed-keep 0.755 \
                    --out $OUT/S10 -- \
    S10_eval      ev $OUT/S10 $GATE_TASKS
  chain 2 \
    C_dense_eval  python $E --dense $M --tasks $GATE_TASKS --limit $LIM --log-samples \
                    --out $OUT/dense_base/eval -- \
    S05           python $C $R --alloc-weight none --total-saving 0.05 --embed-keep 0.881 \
                    --out $OUT/S05 -- \
    S05_eval      ev $OUT/S05 $GATE_TASKS -- \
    C_webonly     python $C $R --calib-mix fineweb:1 --heal-mix fineweb:1 --source-weight "" \
                    --out $OUT/C_webonly -- \
    C_webonly_eval ev $OUT/C_webonly $GATE_TASKS
  chain 3 \
    C_noembed     python $C $R --embed-keep 1 --out $OUT/C_noembed -- \
    C_noembed_eval ev $OUT/C_noembed $GATE_TASKS -- \
    C_embed035    python $C $R --embed-keep 0.35 --out $OUT/C_embed035 -- \
    C_embed035_eval ev $OUT/C_embed035 $GATE_TASKS -- \
    C_regauge     python $C $R --regauge-every 200 --out $OUT/C_regauge -- \
    C_regauge_eval ev $OUT/C_regauge $GATE_TASKS
  wait
  for d in C_ref C_nofisher C_nofisher_int8 C_noembed C_embed035 C_webonly C_regauge \
           S05 S10 S15; do
    echo "--- gate $d"
    python $G --dense $OUT/dense_base/eval/ql4_dense.json \
        --compressed $OUT/$d/eval/ql4_compressed.json --ql5-report $OUT/$d/ql5_report.json \
        --min-saving 0 --out $OUT/$d/gate.json 2>&1 | tail -3 | tee $OUT/$d.gate.txt
  done
else
  M=Qwen/Qwen3-4B-Base
  run4() { torchrun --standalone --nproc_per_node 4 "$@"; }
  job 0,1,2,3 B1_main run4 $C --model $M --total-saving 0.2 --embed-keep 0.5 \
      --heal-tokens $TOK --batch 8 --eval-every 500 --quant int8 --save-model --out $OUT/B1_main
  wait
  job 0,1,2,3 B2_dense_ctl run4 $C --model $M --total-saving 0 --heal-scope all \
      --heal-tokens $TOK --batch 4 --lr 1e-5 --save-model --out $OUT/B2_dense_ctl
  wait
  chain 0 B0_dense_eval python $E --dense $M --tasks $ALL_TASKS --limit $LIM --log-samples \
      --batch 8 --out $OUT/dense_base/eval
  chain 1 B1_eval ev $OUT/B1_main $ALL_TASKS
  chain 2 B2_eval ev $OUT/B2_dense_ctl $ALL_TASKS
  mkdir -p $OUT/B1_int8 && ln -sf $OUT/B1_main/model_int8.pt $OUT/B1_int8/model.pt && \
      ln -sfn $OUT/B1_main/tokenizer $OUT/B1_int8/tokenizer && \
      cp $OUT/B1_main/ql5_report.json $OUT/B1_int8/ql5_report.json
  chain 3 B1_int8_eval ev $OUT/B1_int8 $GATE_TASKS
  wait
  for d in B1_main B1_int8 B2_dense_ctl; do
    echo "--- gate $d"; gate $OUT/$d 2>&1 | tail -3 | tee $OUT/$d.gate.txt
  done
fi
echo "=== $(date -u +%H:%M:%S) stage $STAGE finished"
cat $OUT/STAGES
