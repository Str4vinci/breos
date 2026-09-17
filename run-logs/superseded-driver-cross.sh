#!/usr/bin/env bash
# Fill the missing battery-cost x degradation-model cells: 350 and 711 under
# field v2 and laboratory. The default model already has all three costs.
set -uo pipefail
ROOT=/tmp/breos-062-release.Zb9Wcm
OUT="$ROOT/results/cost-degradation-cross"
PY="$ROOT/.venv/bin/python"
STATUS="$ROOT/run-logs/status-cross.json"
cd "$ROOT"

write_status() {
  printf '{"arm": "%s", "state": "%s", "exit_code": %s, "at": "%s"}\n' \
    "$1" "$2" "$3" "$(date -Is)" > "$STATUS"
}

run_arm() {
  local name=$1 model=$2
  echo "=== $name ($model) starting $(date -Is) ==="
  write_status "$name" running null
  "$PY" tools/reproduce_article1.py \
    --rlp-directory "$ROOT/dev/article1-inputs/rlp" \
    --execution-backend numba \
    --calendar-model "$model" \
    --battery-cost 350 --battery-cost 711 \
    --skip-fixed --full-optimization --n-procs 10 \
    --output "$OUT/$name"
  local code=$?
  echo "=== $name exited $code $(date -Is) ==="
  [ "$code" -ne 0 ] && { write_status "$name" failed "$code"; exit "$code"; }
}

run_arm field-v2 naumann_lam_field_calibrated_v2
run_arm laboratory naumann_lam
write_status all complete 0
echo "=== cross complete $(date -Is) ==="
