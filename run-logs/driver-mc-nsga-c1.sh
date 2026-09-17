#!/usr/bin/env bash
# Monte Carlo replay of the NSGA-II max-NPV answer (6 PV / 0 kWh at 35/195),
# kept alongside the exhaustive optimum used by publication case C1. Written
# outside results/publication-1c so the publication bundle keeps only its own
# six configurations.
set -uo pipefail
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
PY="$ROOT/.venv/bin/python"
STATUS="$ROOT/run-logs/status-mc-nsga-c1.json"
cd "$ROOT"

printf '{"stage": "monte-carlo", "state": "running", "exit_code": null, "at": "%s", "set": "C1_NSGA"}\n' "$(date -Is)" > "$STATUS"
echo "=== nsga-c1 variant starting $(date -Is) ==="
"$PY" tools/reproduce_article1_montecarlo.py \
  --config validation/article1/article1-montecarlo-nsga-c1-variant.toml \
  --case C1_NSGA \
  --execution-backend numba \
  --rlp-directory "$ROOT/dev/article1-inputs/rlp" \
  --weather-file "$ROOT/dev/article1-inputs/weather/porto_historical_2005_2024_openmeteo.csv" \
  --n-procs 10 \
  --output "$ROOT/results/nsga-c1-variant"
code=$?
echo "=== nsga-c1 variant exited $code $(date -Is) ==="
printf '{"stage": "monte-carlo", "state": "%s", "exit_code": %s, "at": "%s", "set": "C1_NSGA"}\n' \
  "$([ $code -eq 0 ] && echo complete || echo failed)" "$code" "$(date -Is)" > "$STATUS"
exit $code
