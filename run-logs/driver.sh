#!/usr/bin/env bash
# Full 0.6.2 simulation candidate rerun: numba backend, 10 workers.
set -uo pipefail

ROOT=/tmp/breos-062-release.Zb9Wcm
OUT="$ROOT/results/upcoming-publication-trial"
STATUS="$ROOT/run-logs/status.json"
PY="$ROOT/.venv/bin/python"
EXPECTED_HEAD=fa5779c

cd "$ROOT"

head_now=$(git rev-parse --short HEAD)
if [ "$head_now" != "$EXPECTED_HEAD" ]; then
  echo "refusing to run: HEAD is $head_now, expected $EXPECTED_HEAD" >&2
  exit 2
fi
if [ -n "$(git status --short --untracked-files=no)" ]; then
  echo "refusing to run: tracked worktree is dirty" >&2
  exit 2
fi

write_status() {
  printf '{"stage": "%s", "state": "%s", "exit_code": %s, "at": "%s", "backend": "numba", "n_procs": 10, "output": "%s"}\n' \
    "$1" "$2" "$3" "$(date -Is)" "$OUT" > "$STATUS"
}

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
  if [ "$code" -ne 0 ]; then
    write_status "$stage" failed "$code"
    exit "$code"
  fi
done

write_status all complete 0
echo "=== all stages complete $(date -Is) ==="
