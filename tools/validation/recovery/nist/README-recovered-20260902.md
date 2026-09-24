# NIST replay package

This package records the NIST Gaithersburg Ground-array replay completed on
2026-09-02 with BREOS revision `f62f4f5bf3c14140ab189d35ea2885e6fcc60c6b`.

## Contents

- `data/nist_ground_2016.csv.gz` is the 366-day OEDI PVDAQ consolidation.
- `results/validate.csv` is the stage and loss-case validation table.
- `results/analysis/` contains the outage, loss-ladder, regime, and monthly
  tables.
- `drivers/` contains the recovered build, validation, and analysis scripts.
- `input_manifest.sha256` hashes the OEDI PVDAQ daily files used by the build.
- `logs/input_preflight.log` records the 366 successful input-hash checks run
  before the final replay.
- `nist_bulk_archives.sha256` hashes the calibrated NIST bulk archives kept in
  `$NIST_RAW_ROOT/`.

The PVDAQ source contains 525,847 rows from
`system_id=4902/year=2016`. The build found all 366 daily files and retained
the fixed EST offset used by the NIST logger.

## Reproduce

Run these commands from the BREOS repository root. Set `BREOS_ROOT` to the
clean checkout at commit `f62f4f5b` when you use another checkout.

```text
.venv/bin/python drivers/nist_build.py \
  --raw-dir $NIST_RAW_ROOT/pvdaq_system4902_2016 \
  --out data/nist_ground_2016.csv.gz
PYTHONPATH=$BREOS_ROOT \
  .venv/bin/python drivers/nist_validate.py \
  --data data/nist_ground_2016.csv.gz \
  --out results/validate.csv
PYTHONPATH=$BREOS_ROOT:$PWD/drivers \
  .venv/bin/python drivers/nist_analysis.py \
  --data data/nist_ground_2016.csv.gz \
  --outdir results/analysis
```

The replay produced 28 excluded outage or snow days. The loss-free Faiman
case had a normal-day AC bias of `-3.514%`. The checkpoint values in
`FINDINGS.md` and the parent recovery verification log match the replay.

The canonical bulk archives came from the [NIST PV Data portal](https://pvdata.nist.gov/)
and dataset DOI `10.18434/M3S67G`. Esposende is not part of this package.
