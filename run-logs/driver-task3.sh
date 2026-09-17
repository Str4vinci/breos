#!/usr/bin/env bash
# Re-run the accepted Task 3 end-of-life sweep on 0.6.2, after the
# cost x degradation cross frees the cores.
set -uo pipefail
ROOT=/tmp/breos-062-release.Zb9Wcm
PY="$ROOT/.venv/bin/python"
STATUS="$ROOT/run-logs/status-task3.json"
cd "$ROOT"

printf '{"stage": "waiting for breos-062-cross", "at": "%s"}\n' "$(date -Is)" > "$STATUS"
while [ "$(systemctl --user is-active breos-062-cross)" = active ]; do sleep 20; done

echo "=== task3 starting $(date -Is) ==="
printf '{"stage": "running", "at": "%s"}\n' "$(date -Is)" > "$STATUS"
"$PY" tools/revision/task3_eol_sweep.py \
  --rlp-directory "$ROOT/dev/article1-inputs/rlp" \
  --output "$ROOT/results/task3-eol-0.6.2" \
  --execution-backend numba --n-procs 16
code=$?
echo "=== task3 exited $code $(date -Is) ==="
printf '{"stage": "finished", "exit_code": %s, "at": "%s"}\n' "$code" "$(date -Is)" > "$STATUS"
exit "$code"
