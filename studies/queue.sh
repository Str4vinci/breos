#!/usr/bin/env bash
# Studies 12-16, one after another, after study 11's grid service finishes.
# Every step is resumable: rerunning this script skips finished outputs.
set -u
cd "$(dirname "$0")"
log() { echo "$(date '+%F %T') $*"; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
while systemctl --user is-active -q breos-s11-grid; do sleep 60; done
log "study 11 grid finished"

log "study 12 start"; (cd 12_year_order && uv run python run_order.py --workers 10); log "study 12 end $?"
log "study 13 start"; (cd 13_replacement_policy && uv run python run_policy.py --workers 10); log "study 13 end $?"

log "study 14 enumerate start"
for space in az180 az180-berlin az180-colorado az180-high_load az180-cheap_battery; do
  for sc in flat bi_greedy; do
    (cd 14_three_objectives && uv run python enumerate_space.py "$sc" "$space" --workers 10) || log "enumerate $sc $space failed"
  done
done
log "study 14 replay start"
for space in az180 az180-berlin az180-colorado az180-high_load az180-cheap_battery; do
  (cd 14_three_objectives && uv run python run_3obj.py "$space" flat,bi_greedy --seeds 1-50 --workers 10) || log "replay $space failed"
done
log "study 14 end"

log "study 15 start"; (cd 15_resources && uv run python run_resources.py --workers 3); log "study 15 end $?"

log "study 16 start"
for env in 16_version_drift/data/envs/*/; do
  name=$(basename "$env")
  (cd 16_version_drift && "data/envs/$name/bin/python" run_gallery.py "data/out/$name") || log "gallery $name failed"
done
log "study 16 end"
log "queue done"
