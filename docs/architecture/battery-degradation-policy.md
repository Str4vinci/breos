# Battery degradation policy

This policy defines the scientific and public-API gates for battery-degradation
changes. Implementation details and the current model catalog live in
[BLAST Degradation Engine](blast-degradation-engine.md) and
[Battery degradation models](../api/degradation-models.md).

## Stable public behavior

`breos.App` is the primary stable entry point. Native Naumann/Lam degradation
remains its default; BLAST is an explicit opt-in through
`degradation_engine="blast"` and a named `blast_model`. Unsupported
combinations raise rather than silently falling back to native behavior.

The App-level `battery_type` key was already rejected by strict validation in
0.3.4. Its targeted 0.4.0 error is migration guidance, not a newly removed App
feature. The lower-level `BatteryConfig(battery_type="LFP")` API remains
supported for native degradation and is not a BLAST model selector.

Changes to these selectors, documented result/provenance fields, or default
behavior require focused `App` tests and an explicit changelog migration note.

## Default-change evidence

Keep `naumann_lam_field_calibrated` as the residential-LFP default. It maps to
native v1; native v2 remains an explicit sensitivity model. The recorded
calibration evidence is:

| Native calibration | Full-fit RMSE | Leave-one-system-out result | Policy |
| --- | ---: | ---: | --- |
| v1 | 4.40 percentage points | mean RMSE 6.00 pp | Current default |
| v2 | 4.65 percentage points | mean RMSE 5.49 pp; pooled RMSE 5.61 pp | Sensitivity model |

V2's cross-validation result is encouraging, but the systems used in model
selection are not a genuinely held-out validation set. Do not change the
generic default based only on synthetic trajectories, in-sample fit, or
upstream BLAST parity. A default change requires frozen model choices,
pack-level field data excluded from fitting and model selection, per-system as
well as pooled error reporting, and a documented impact on residential studies.

The most interpretable native-versus-BLAST comparison uses BLAST's
`lfp_gr_sonymurata_3ah` model because both paths ultimately relate to the
Naumann Sony/Murata 3 Ah LFP data. The 250 Ah prismatic model is a useful
stationary-cell sensitivity, not equivalent default-selection evidence.
Upstream parity proves implementation fidelity; it does not prove real-world
predictive validity.

## Scientific interpretation

A `blast_model` key identifies a particular empirical cell model, not a generic
chemistry curve. Model identity, citations, experimental ranges, and output
capabilities come from the single public registry. Experimental ranges drive
warnings; they are not recommended pack settings. Operating defaults are added
only when a source supports them, with precedence remaining explicit user
configuration over sourced profile defaults over global defaults.

BLAST outputs are cell-model projections, not pack-calibrated predictions.
Thermal gradients, cell variation, imbalance, interconnects, and BMS behavior
can all change pack-level capacity and power fade. Any future cell-to-pack layer
must use an explicit documented mapping and validation on packs outside its fit
set; an unexplained degradation multiplier is insufficient.

Multi-decade SOH remains model-dependent. Preserve the selected model key and
range/horizon warnings with the result. If a future API reports results across
several plausible models, call that a **model sensitivity range**, not a
statistical confidence interval unless a probabilistic model has been
calibrated.

## Resistance and screening paths

BLAST resistance outputs are diagnostic only. They must not alter dispatch,
efficiency, or charge/discharge power limits until each model's resistance
metric and test protocol have a defensible, validated mapping to pack behavior.
The executable isolation guard is
`tests/test_battery.py::TestSimulateEnergyBalance::test_blast_resistance_output_is_diagnostic_only`.

The reference Python path remains the accuracy and calibration basis. Optional
Numba kernels are explicitly approximate screening tools and must not provide
calibration evidence or replace the reference path in published accuracy
comparisons merely because they are faster.

## Provenance, legal records, and fixtures

Keep these records separate:

- `breos/degradation/blast/VENDORED.md` is the source/provenance authority for
  the pinned upstream commit, transformations, and file hashes.
- `ATTRIBUTIONS.md` plus the bundled `LICENSE` and `NOTICE` preserve legal and
  scientific credits.
- The BREOS adapter, dispatch integration, registry, and runners are original
  BREOS code outside the vendored-source table.
- Parity fixtures are derived scientific test data, not upstream source or
  license artifacts. They stay fixed in CI. Regeneration must use the pinned,
  unmodified BLAST-Lite source and be reviewed as a scientific data change.

The executable artifact and fixture gates are maintained in
[Public Release Checklist](../release.md); this policy does not duplicate that
release procedure.
