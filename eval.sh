#!/usr/bin/env bash
set -euo pipefail

#######################################
# CONFIG (edit these)
#######################################
WORKDIR="/workspace/dave/PaperDataExtraction"

# Path to your venv activate script (you said you'll set it manually)
VENV_ACTIVATE="/workspace/dave/PaperDataExtraction/.venv/bin/activate"

# Full command you want to run (hardcoded)
TRAIN_CMD='python eval.py \
  --base_model BioMistral/BioMistral-7B \
  --adapter_dir outputs/biomistral_mie_ke_ao_qlora \
  --eval_file train/test.jsonl \
  --load_in_4bit --bf16 \
  --max_new_tokens 256 \
  --save_preds eval_preds.jsonl'
#######################################

mkdir -p "${WORKDIR}/logs"

TS="$(date +%F_%H-%M-%S)"
LOG_FILE="${WORKDIR}/logs/train_${TS}.log"
PID_FILE="${WORKDIR}/logs/train_${TS}.pid"

# One-liner runner executed by a detached shell.
# Important bits:
# - cd WORKDIR
# - source venv activate
# - PYTHONUNBUFFERED=1 for real-time logs
# - exec ensures PID is the python process, not just a wrapper shell
RUNNER=$(cat <<EOF
cd $(printf '%q' "$WORKDIR") &&
source $(printf '%q' "$VENV_ACTIVATE") &&
export PYTHONUNBUFFERED=1 &&
exec ${TRAIN_CMD}
EOF
)

start_detached() {
  if command -v nohup >/dev/null 2>&1; then
    nohup bash -lc "$RUNNER" >"$LOG_FILE" 2>&1 &
  elif command -v setsid >/dev/null 2>&1; then
    setsid bash -lc "$RUNNER" >"$LOG_FILE" 2>&1 < /dev/null &
  else
    bash -lc "$RUNNER" >"$LOG_FILE" 2>&1 &
  fi

  PID=$!
  # detach from job control so SSH death doesn't kill it
  disown "$PID" >/dev/null 2>&1 || disown -h "$PID" >/dev/null 2>&1 || true

  echo "$PID" >"$PID_FILE"

  # convenience pointers
  echo "$LOG_FILE" > "${WORKDIR}/logs/latest.log.path"
  echo "$PID" > "${WORKDIR}/logs/latest.pid"
}

# sanity checks (fail fast, like reality should)
[[ -d "$WORKDIR" ]] || { echo "WORKDIR does not exist: $WORKDIR" >&2; exit 1; }
[[ -f "$VENV_ACTIVATE" ]] || { echo "VENV activate not found: $VENV_ACTIVATE" >&2; exit 1; }

start_detached

echo "Started training detached from SSH."
echo "PID:      $PID"
echo "LOG FILE: $LOG_FILE"
echo
echo "Monitor:"
echo "  tail -f $LOG_FILE"
echo "  ps -fp $PID"
echo
echo "Stop:"
echo "  kill $PID"
