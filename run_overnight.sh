#!/usr/bin/env bash
# Overnight runner for the whole PA3 pipeline (run from Git Bash in the repo root):
#
#   ./run_overnight.sh            # everything: setup -> Task 3 -> Task 2 -> Task 1 -> leakage check
#   ./run_overnight.sh task1      # one phase only (setup | task3 | task2 | task1 | final)
#   OPTIONAL_POOL49=1 ./run_overnight.sh task2   # also run the optional 49-visual-token Task 2 variant
#
# Resumable: a step whose done-marker exists is skipped, and the Python entry points themselves skip finished work
# and resume interrupted training from the latest recovery checkpoint. So simply rerun this script each night until
# STATUS says ALL DONE. A failed step is retried once; steps depending on a failed step are skipped, independent
# steps still run. Logs: results/_overnight/logs/<step>.log, summary: results/_overnight/STATUS.txt.
#
# Expected wall clock on an RTX 4060 Laptop (8 GB): Task 3 ~2-3 h (live generation), Task 2 ~3 h,
# Task 1 ~26-30 h (five 1,250-update runs at ~11-13 s/update + seven lm-eval passes). Plan for ~3 nights.
set -u
cd "$(dirname "$0")"
PY=".venv/Scripts/python.exe"
[ -x "$PY" ] || PY=".venv/bin/python"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
OUT="results/_overnight"; LOGS="$OUT/logs"; DONE="$OUT/done"; STATUS="$OUT/STATUS.txt"
mkdir -p "$LOGS" "$DONE"
PHASE="${1:-all}"
source scripts/run_lib.sh

note "===== run_overnight.sh phase=$PHASE pid=$$ ====="

phase_setup() {
  run_step env_check "" -m scripts.check_environment
  run_step prefetch  "" -m scripts.prefetch && export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
}

phase_task3() {
  local C="--config configs/task3.yaml"
  run_step t3_data      ""          -m task3_ttc.data $C
  run_step t3_offline   "t3_data"   -m task3_ttc.offline $C
  run_step t3_adaptive  "t3_offline" -m task3_ttc.adaptive $C
  run_step t3_live      "t3_data"   -m task3_ttc.live $C
  run_step t3_summarize "t3_offline" -m task3_ttc.summarize $C
}

phase_task2() {
  local C="--config configs/task2.yaml"
  run_step t2_data      ""                 -m task2_vlm.data_prep $C
  run_step t2_inspect   "t2_data"          -m task2_vlm.data_prep $C --inspect
  run_step t2_features  "t2_data"          -m task2_vlm.features $C
  run_step t2_checks    "t2_features"      -m task2_vlm.checks $C
  run_step t2_train     "t2_checks"        -m task2_vlm.train $C
  run_step t2_eval      "t2_train"         -m task2_vlm.evaluate $C
  run_step t2_summarize "t2_train"         -m task2_vlm.summarize $C
  if [ "${OPTIONAL_POOL49:-0}" = "1" ]; then
    run_step t2p_train     "t2_checks"     -m task2_vlm.train $C --variant pool49
    run_step t2p_eval      "t2p_train"     -m task2_vlm.evaluate $C --variant pool49
    run_step t2p_summarize "t2p_train"     -m task2_vlm.summarize $C --variant pool49
  fi
}

phase_task1() {
  local C="--config configs/task1.yaml"
  run_step t1_data          ""            -m task1_finetune.data $C
  run_step t1_probe         ""            -m task1_finetune.probe_set $C
  run_step t1_eval_baseline "t1_data"     -m task1_finetune.evaluate $C --condition baseline
  run_step t1_qual_baseline ""            -m task1_finetune.qualitative $C --condition baseline
  # Full FT (midpoint + final are evaluated), then L2-SP
  run_step t1_train_full    "t1_data"     -m task1_finetune.train $C --method full
  run_step t1_eval_full_mid "t1_train_full" -m task1_finetune.evaluate $C --condition full_mid
  run_step t1_eval_full     "t1_train_full" -m task1_finetune.evaluate $C --condition full
  run_step t1_qual_full_mid "t1_train_full" -m task1_finetune.qualitative $C --condition full_mid
  run_step t1_qual_full     "t1_train_full" -m task1_finetune.qualitative $C --condition full
  run_step t1_train_l2sp    "t1_data"     -m task1_finetune.train $C --method l2sp
  run_step t1_eval_l2sp     "t1_train_l2sp" -m task1_finetune.evaluate $C --condition l2sp
  run_step t1_qual_l2sp     "t1_train_l2sp" -m task1_finetune.qualitative $C --condition l2sp
  for m in lora4 lora16 oplora16; do
    run_step "t1_train_$m" "t1_data"        -m task1_finetune.train $C --method $m
    run_step "t1_merge_$m" "t1_train_$m"    -m task1_finetune.merge $C --method $m
    run_step "t1_eval_$m"  "t1_merge_$m"    -m task1_finetune.evaluate $C --condition $m
    run_step "t1_qual_$m"  "t1_merge_$m"    -m task1_finetune.qualitative $C --condition $m
  done
  run_step t1_diagnostics   "t1_probe"    -m task1_finetune.diagnostics $C
  run_step t1_qual_export   ""            -m task1_finetune.qualitative $C --export
  run_step t1_summarize     ""            -m task1_finetune.summarize $C
}

phase_final() {
  run_step leakage_check "" -m scripts.leakage_check
}

case "$PHASE" in
  setup) phase_setup ;;
  task3) phase_setup; phase_task3 ;;
  task2) phase_setup; phase_task2 ;;
  task1) phase_setup; phase_task1 ;;
  final) phase_final ;;
  all)   phase_setup; phase_task3; phase_task2; phase_task1; phase_final ;;
  *) echo "unknown phase $PHASE"; exit 2 ;;
esac

# Steps whose outputs change as more runs finish are re-run on every invocation (cheap, idempotent).
for s in t1_diagnostics t1_qual_export t1_summarize t2_summarize t3_summarize leakage_check; do rm -f "$DONE/$s"; done
if [ ${#FAILED[@]} -eq 0 ] && [ ${#PAUSED[@]} -eq 0 ]; then note "===== phase $PHASE: ALL DONE ====="
else note "===== phase $PHASE unfinished -- $(unfinished) ====="; fi
