#!/usr/bin/env bash
# Rerun the Monte Carlo study on the revised six-configuration set (C1-C6).
# Writes to a staging directory; the caller swaps it into monte-carlo-v1 after
# the reproducibility check on the carried-over cases passes.
set -uo pipefail
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
STAGE="$ROOT/results/publication-1c/monte-carlo-v1-staging"
PY="$ROOT/.venv/bin/python"
STATUS="$ROOT/run-logs/status-mc-c1c6.json"
cd "$ROOT"

write_status() {
  printf '{"stage": "%s", "state": "%s", "exit_code": %s, "at": "%s", "set": "C1-C6"}\n' \
    "$1" "$2" "$3" "$(date -Is)" > "$STATUS"
}

common=(
  --execution-backend numba
  --rlp-directory "$ROOT/dev/article1-inputs/rlp"
  --weather-file "$ROOT/dev/article1-inputs/weather/porto_historical_2005_2024_openmeteo.csv"
  --n-procs 10
  --output "$STAGE"
)

echo "=== monte-carlo C1-C6 starting $(date -Is) ==="
write_status monte-carlo running null
"$PY" tools/reproduce_article1_montecarlo.py \
  --case C1 --case C2 --case C3 --case C4 --case C5 --case C6 "${common[@]}"
code=$?
echo "=== monte-carlo exited $code $(date -Is) ==="
[ "$code" -ne 0 ] && { write_status monte-carlo failed "$code"; exit "$code"; }
write_status monte-carlo complete 0
