# Validation recovery provenance

This index points to the recovered upcoming-publication validation outputs and records the
checks run on 2026-09-02. The BREOS source revision used for every replay is
`f62f4f5bf3c14140ab189d35ea2885e6fcc60c6b`.

## Find the outputs

The six local-data packages are in
`/home/leo/Documents/BREOS_validation_recovery_20260902/`:

- `validation_sandia_task13_recovered_20260902`
- `validation_pcoe_recovered_20260902`
- `validation_reunion_microgrid_recovered_20260902`
- `validation_orientation_diversity_recovered_20260902`
- `validation_hkust_timing-corrected-exploratory-v4_recovered_20260902`
- `validation_dkasc_recovered_20260902`

The package-level `artifact_manifest.sha256` verifies every recovered output.
The package `input_manifest.sha256` files also match their source directories.
Run the checkpoint verifier from the repository root:

```text
.venv/bin/python tools/validation/recovery/verify_recovery.py \
  /home/leo/Documents/BREOS_validation_recovery_20260902
```

That verifier passed all 17 recorded checkpoints. The complete log is
`/home/leo/Documents/BREOS_validation_recovery_20260902/verification.log`.

## NIST replay

The NIST raw data is in `/home/leo/Documents/NIST_Gaithersburg_PV/`.
The three canonical 2016 bulk archives are present and match the recorded
hashes in `SHA256SUMS.txt`:

```text
f12498294ad3bec41150a9263ac28344563bc24896d907a8b05576d18ae1abf8  onemin-Ground-2016.zip
fc63b434da6f86c0b8dfe96195f99ea15b4f7c0b34076d7407836652901efa1b  onemin-WS_1-2016.zip
49de8481f17af242be0c79f6b5513686fbb5d7be4baf975a00347444586c8375  onemin-WS_2-2016.zip
```

The PVDAQ daily files used by the recovered NIST PV-chain tools are in
`/home/leo/Documents/NIST_Gaithersburg_PV/pvdaq_system4902_2016/`. The build
found 366 files, 525,847 rows, and the span `2016-01-01 00:00:00-05:00` to
`2016-12-31 23:59:00-05:00`.

The replay used the clean `f62f4f5b` checkout at
`/tmp/breos-validation-f62f4f5b`:

```text
.venv/bin/python tools/validation/recovery/nist/nist_build.py \
  --raw-dir /home/leo/Documents/NIST_Gaithersburg_PV/pvdaq_system4902_2016 \
  --out <nist-run>/nist_ground_2016.csv.gz
PYTHONPATH=/tmp/breos-validation-f62f4f5b \
  .venv/bin/python tools/validation/recovery/nist/nist_validate.py \
  --data <nist-run>/nist_ground_2016.csv.gz \
  --out <nist-run>/results/validate.csv
PYTHONPATH=/tmp/breos-validation-f62f4f5b:tools/validation/recovery/nist \
  .venv/bin/python tools/validation/recovery/nist/nist_analysis.py \
  --data <nist-run>/nist_ground_2016.csv.gz \
  --outdir <nist-run>/results/analysis
```

The replay passed. It reproduced the recorded NIST values, including 28
excluded outage or snow days, a 4.23% modelled-energy loss from those days,
and a normal-day Faiman loss-free bias of -3.514%.
The NIST package's `logs/input_preflight.log` records all 366 input-hash checks
before the final replay.

The NIST replay package is
`/home/leo/Documents/BREOS_validation_recovery_20260902/validation_nist_gaithersburg_recovered_20260902/`.
Its built dataset SHA-256 is
`9f9e4e8a4cc18bd9fee0326ffacb3aaebb1f82d22d9d975bdeb8561ccf2cda9d`.

The canonical NIST archive source is [the NIST PV Data portal](https://pvdata.nist.gov/),
dataset DOI `10.18434/M3S67G`. The OEDI PVDAQ source pattern and the full
method record are in `validation_nist_gaithersburg_recovered_20260902/README.md`
and `FINDINGS.md`.

## Scope exclusions

Esposende remains excluded at the user's request. No Esposende input or result
is part of the recovery archive.

The recovery archive and its SHA-256 sidecar are:

```text
/home/leo/Documents/BREOS_validation_recovery_20260902.tar.gz
/home/leo/Documents/BREOS_validation_recovery_20260902.tar.gz.sha256
```

The archive SHA-256 is
`88d8090cb39fdb68d377879cc210a6e0fcc54a7e3567691eb9df61bb3673ff22`.
