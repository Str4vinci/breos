# Upcoming-publication validation recovery

This directory preserves the drivers that generated the deleted
external-validation packages. Sandia Task 13, PCoE, Reunion, and orientation
diversity were reconstructed by replaying the recorded `apply_patch` calls
from the 2026-08-29 Codex transcript. HKUST v4 was reconstructed from the
surviving v3 driver plus the two recorded v4 patches. The DKASC and NIST
drivers were recovered from the local `validate/external-pv-datasets` branch.

`originals/` preserves the historical reconstructions. For DKASC and NIST,
these files are byte-for-byte copies from the branch. The executable copies
outside `originals/` are formatted for the current tree. The four
package-local runners also add one environment-controlled output path so the
recovery can write to a new immutable package instead of the deleted location.
The scientific calculations are unchanged, and `verify_recovery.py` checks
their outputs against the recorded historical checkpoints.

The unattended runner waits for an optional process ID, then rebuilds the packages that still have local raw data. It verifies selected outputs against the numbers recorded before deletion. It never deletes or overwrites an existing output package.

NIST is not part of the unattended run because its portal requires a separate dynamic archive download. The 2016 replay is complete at `/home/leo/Documents/BREOS_validation_recovery_20260902/validation_nist_gaithersburg_recovered_20260902`. Esposende is excluded by decision.

Example:

```bash
.venv/bin/python tools/validation/recovery/run_recovery.py \
  --wait-pid 642394 \
  --breos-root /tmp/breos-article1-0.6.0 \
  --output-root /path/to/new/validation-recovery-20260902
```

The output root contains `status.json`, one directory per validation package, `verification.log`, and `artifact_manifest.sha256`.
