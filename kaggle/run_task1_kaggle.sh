#!/usr/bin/env bash
# Task 1 on Kaggle (2 x T4, 16 GB each). Run from the repository root inside the notebook:
#
#   PA3_DEADLINE_EPOCH=<unix time> bash kaggle/run_task1_kaggle.sh
#
# Two chains run in parallel, one per GPU:
#   GPU 0: Full FT (+ midpoint/final eval and qualitative), LoRA-4, LoRA-16 (each: train, merge, eval, qualitative)
#   GPU 1: baseline eval + qualitative, Full FT + L2-SP, OPLoRA-16
# then diagnostics, the qualitative annotation export and the summary. With a single GPU the chains run in sequence.
#
# Every step is resumable and skips finished work. Training stops cleanly at PA3_DEADLINE_EPOCH with a recovery
# checkpoint (exit 3 = paused); evaluations only start if enough session time remains (EVAL_MIN). Rerun the
# notebook with the previous version's output attached until kaggle/carry.py reports TASK 1 ALL DONE.
#
# Platform settings (recorded in the run summaries as config overrides, no hyperparameter changes):
#   offload_optimizer_states=false  16 GB fits FP32 weights + gradients + AdamW states on the GPU
#   train.prune_recovery=true       superseded optimizer states / non-evaluated weights deleted (disk limit)
#   checkpoints in scratch space    carried between sessions by kaggle/carry.py
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
CKPT="${PA3_CKPT_ROOT:-/kaggle/tmp/checkpoints/task1}"
export PA3_CKPT_ROOT="$CKPT"
export PA3_OVERRIDES_TASK1="{\"offload_optimizer_states\": false, \"train.prune_recovery\": true, \"paths.checkpoints\": \"$CKPT\", \"paths.merged\": \"$CKPT/merged\"}"
EVAL_MIN="${EVAL_MIN:-150}" # minutes one evaluate.py pass needs: measured 117 min GSM8K generation (batch 8) +
                            # 15 min likelihood tasks (batch 4) = 132 min on an 8 GB laptop GPU; 150 min adds
                            # margin for a slower T4. These batch sizes are fixed, not "auto" -- see configs/task1.yaml.
OUT="results/_overnight"; LOGS="$OUT/logs"; DONE="$OUT/done"; STATUS="$OUT/STATUS.txt"
mkdir -p "$LOGS" "$DONE" "$CKPT"
source scripts/run_lib.sh
C="--config configs/task1.yaml"

note "===== Kaggle Task 1 session: $(minutes_left) min until deadline; checkpoints in $CKPT ====="
run_step t1_data  "" -m task1_finetune.data $C
run_step t1_probe "" -m task1_finetune.probe_set $C

method_chain() {   # METHOD -- train, then that method's evaluation conditions
  local m=$1
  run_timed "t1_train_$m" 20 "t1_data" -m task1_finetune.train $C --method "$m" || return 0
  case "$m" in
    full)
      run_timed t1_eval_full_mid "$EVAL_MIN" "t1_train_full" -m task1_finetune.evaluate $C --condition full_mid
      run_timed t1_eval_full     "$EVAL_MIN" "t1_train_full" -m task1_finetune.evaluate $C --condition full
      run_timed t1_qual_full_mid 15 "t1_train_full" -m task1_finetune.qualitative $C --condition full_mid
      run_timed t1_qual_full     15 "t1_train_full" -m task1_finetune.qualitative $C --condition full ;;
    l2sp)
      run_timed t1_eval_l2sp "$EVAL_MIN" "t1_train_l2sp" -m task1_finetune.evaluate $C --condition l2sp
      run_timed t1_qual_l2sp 15 "t1_train_l2sp" -m task1_finetune.qualitative $C --condition l2sp ;;
    *)
      run_timed "t1_merge_$m" 15 "t1_train_$m" -m task1_finetune.merge $C --method "$m" || return 0
      run_timed "t1_eval_$m" "$EVAL_MIN" "t1_merge_$m" -m task1_finetune.evaluate $C --condition "$m"
      run_timed "t1_qual_$m" 15 "t1_merge_$m" -m task1_finetune.qualitative $C --condition "$m" ;;
  esac
}
chain_a() { method_chain full; method_chain lora4; method_chain lora16; }
chain_b() {
  run_timed t1_eval_baseline "$EVAL_MIN" "t1_data" -m task1_finetune.evaluate $C --condition baseline
  run_timed t1_qual_baseline 15 "" -m task1_finetune.qualitative $C --condition baseline
  method_chain l2sp
  method_chain oplora16
}

N_GPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$N_GPU" -ge 2 ]; then
  note "running two chains in parallel on $N_GPU GPUs"
  ( export CUDA_VISIBLE_DEVICES=0 TAG=gpu0; chain_a ) &
  PID_A=$!
  ( export CUDA_VISIBLE_DEVICES=1 TAG=gpu1; chain_b ) &
  PID_B=$!
  wait $PID_A $PID_B
else
  note "single GPU: running chains in sequence"
  chain_a
  chain_b
fi

# Cross-run analyses: recomputed every session from whatever is finished (idempotent).
export CUDA_VISIBLE_DEVICES=0
rm -f "$DONE/t1_diagnostics" "$DONE/t1_qual_export" "$DONE/t1_summarize"
run_timed t1_diagnostics 15 "" -m task1_finetune.diagnostics $C
run_step  t1_qual_export "" -m task1_finetune.qualitative $C --export
run_step  t1_summarize   "" -m task1_finetune.summarize $C
note "===== Kaggle Task 1 session finished ($(minutes_left) min left) ====="
"$PY" kaggle/carry.py status | tee -a "$STATUS"
