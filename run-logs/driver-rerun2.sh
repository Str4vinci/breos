#!/usr/bin/env bash
# Full rerun on the accepted representative set and the 1 C battery limit,
# then the battery-cost x degradation cross that the analysis stage does not
# cover. Waits for the Task 3 sweep to free the cores first.
set -uo pipefail
ROOT=/tmp/breos-062-release.Zb9Wcm
OUT="$ROOT/results/publication-1c"
CROSS="$ROOT/results/cost-degradation-cross"
PY="$ROOT/.venv/bin/python"
STATUS="$ROOT/run-logs/status-rerun2.json"
EXPECTED_HEAD=b3d0034
cd "$ROOT"

write_status() {
  printf '{"stage": "%s", "state": "%s", "exit_code": %s, "at": "%s", "config": "accepted set + 1 C"}\n' \
    "$1" "$2" "$3" "$(date -Is)" > "$STATUS"
}

write_status "waiting for breos-062-task3" waiting null
while [ "$(systemctl --user is-active breos-062-task3)" = active ]; do sleep 20; done

head_now=$(git rev-parse --short HEAD)
if [ "$head_now" != "$EXPECTED_HEAD" ]; then
  echo "refusing to run: HEAD is $head_now, expected $EXPECTED_HEAD" >&2
  write_status guard failed 2; exit 2
fi
if [ -n "$(git status --short --untracked-files=no)" ]; then
  echo "refusing to run: tracked worktree is dirty" >&2
  write_status guard failed 2; exit 2
fi

for stage in fixed analysis monte-carlo verify; do
  echo "=== $stage starting $(date -Is) ==="
  write_status "$stage" running null
  if [ "$stage" = verify ]; then
    "$PY" tools/run_article1.py verify --output "$OUT"
  else
    "$PY" tools/run_article1.py "$stage" \
      --execution-backend numba --n-procs 10 --output "$OUT"
  fi
  code=$?
  echo "=== $stage exited $code $(date -Is) ==="
  [ "$code" -ne 0 ] && { write_status "$stage" failed "$code"; exit "$code"; }
done

for arm in "field-v2 naumann_lam_field_calibrated_v2" "laboratory naumann_lam"; do
  set -- $arm
  echo "=== cross $1 starting $(date -Is) ==="
  write_status "cross-$1" running null
  "$PY" tools/reproduce_article1.py \
    --rlp-directory "$ROOT/dev/article1-inputs/rlp" \
    --execution-backend numba --calendar-model "$2" \
    --battery-cost 350 --battery-cost 711 \
    --skip-fixed --full-optimization --n-procs 10 \
    --output "$CROSS/$1"
  code=$?
  echo "=== cross $1 exited $code $(date -Is) ==="
  [ "$code" -ne 0 ] && { write_status "cross-$1" failed "$code"; exit "$code"; }
done

write_status all complete 0
echo "=== rerun complete $(date -Is) ==="
