# Shared step runner for the pipeline orchestrators (sourced by run_overnight.sh and kaggle/run_task1_kaggle.sh).
#
# The caller sets PY (python executable), LOGS, DONE and STATUS, and may set PA3_DEADLINE_EPOCH (unix seconds).
#   run_step NAME "DEPS" python-args...        run a step (skip if done; retry once on failure)
#   run_timed NAME MINUTES "DEPS" python-args  same, but only start if at least MINUTES remain before the deadline
# Exit code 3 from a step means "paused at the deadline with a recovery checkpoint": it is not retried, not marked
# done, and steps depending on it are skipped until a later invocation finishes it.

declare -A FAILED=()
declare -A PAUSED=()

note() { echo "[$(date '+%F %T')]${TAG:+ [$TAG]} $*" | tee -a "$STATUS"; }

# summary of unfinished steps for the final status line
unfinished() {
  local f="${!FAILED[*]}" p="${!PAUSED[*]}"
  echo "failed: ${f:-none}; paused/skipped: ${p:-none}"
}

minutes_left() {
  if [ -z "${PA3_DEADLINE_EPOCH:-}" ]; then echo 999999; return; fi
  echo $(( (${PA3_DEADLINE_EPOCH%.*} - $(date +%s)) / 60 ))
}

run_step() {
  local name="$1" deps="$2"; shift 2
  if [ -f "$DONE/$name" ]; then note "SKIP  $name (done)"; return 0; fi
  for d in $deps; do
    if [ -n "${FAILED[$d]:-}" ] || [ -n "${PAUSED[$d]:-}" ]; then
      note "SKIP  $name (dependency $d not finished)"; PAUSED[$name]=1; return 1
    fi
  done
  local attempt rc t0
  for attempt in 1 2; do
    note "START $name (attempt $attempt)"
    t0=$(date +%s)
    "$PY" "$@" >> "$LOGS/$name.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
      note "OK    $name ($(( ($(date +%s) - t0) / 60 )) min)"
      date '+%F %T' > "$DONE/$name"
      return 0
    fi
    if [ $rc -eq 3 ]; then
      note "PAUSE $name (session deadline; recovery checkpoint saved, resumes on the next run)"
      PAUSED[$name]=1
      return 1
    fi
    note "FAIL  $name attempt $attempt (exit $rc) -- last lines of $LOGS/$name.log:"
    grep -v -iE "warn|Timeout is disabled|you must provide|logic for timeout" "$LOGS/$name.log" | tail -n 15 | tee -a "$STATUS"
  done
  FAILED[$name]=1
  return 1
}

run_timed() {
  local name="$1" need="$2"; shift 2
  if [ -f "$DONE/$name" ]; then note "SKIP  $name (done)"; return 0; fi
  local left; left=$(minutes_left)
  if [ "$left" -lt "$need" ]; then
    note "DEFER $name (needs ~${need} min, ${left} min left before the session deadline)"
    PAUSED[$name]=1
    return 1
  fi
  run_step "$name" "$@"
}
