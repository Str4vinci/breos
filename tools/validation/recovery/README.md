# Upcoming-publication validation recovery

This directory preserves the drivers that generated the deleted
external-validation packages. Sandia Task 13, PCoE, Reunion, and orientation
diversity were reconstructed by replaying the recorded `apply_patch` calls
from the 2026-08-29 Codex transcript. HKUST v4 was reconstructed from the
surviving v3 driver plus the two recorded v4 patches. The DKASC and NIST
drivers were recovered from the local `validate/external-pv-datasets` branch.

The drivers here are formatted for the current tree. They differ from the
historical reconstructions only in layout and in the environment-controlled
input and output paths, so the recovery writes to a new immutable package
instead of the deleted location. The scientific calculations are unchanged,
and `verify_recovery.py` checks their outputs against the recorded historical
checkpoints. The unformatted historical text is not kept here: the archived
package recorded in `PROVENANCE-20260902.md` is the authoritative copy, and it
is verified by hash rather than by a second checkout of the same program.

## Paths

Raw inputs stay outside the repository because of size and licensing. Nothing
here hard-codes a machine-local location:

- `BREOS_VALIDATION_DATA` is the directory holding the downloaded datasets.
  Each package README names the subdirectory it expects.
- `BREOS_VALIDATION_ROOT` is the pinned article worktree the drivers import
  BREOS from.
- `BREOS_VALIDATION_OUTPUT` is the package directory to write, defaulting to a
  path under `results/` relative to the working directory.
- `BREOS_VALIDATION_DKASC_RAW`, or `--dkasc-raw-dir`, is the unpacked DKASC
  archive.

`$RECOVERY_ROOT` and `$NIST_RAW_ROOT` in these documents stand for the output
root passed to the runner and the directory holding the NIST bulk archives.

The unattended runner waits for an optional process ID, then rebuilds the packages that still have local raw data. It verifies selected outputs against the numbers recorded before deletion. It never deletes or overwrites an existing output package.

NIST is not part of the unattended run because its portal requires a separate dynamic archive download. The 2016 replay is complete at `$RECOVERY_ROOT/validation_nist_gaithersburg_recovered_20260902`. Esposende is excluded by decision.

Example:

```bash
.venv/bin/python tools/validation/recovery/run_recovery.py \
  --wait-pid 642394 \
  --breos-root /tmp/breos-article1-0.6.0 \
  --output-root /path/to/new/validation-recovery-20260902
```

The output root contains `status.json`, one directory per validation package, `verification.log`, and `artifact_manifest.sha256`.
