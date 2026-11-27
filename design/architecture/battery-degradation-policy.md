# Battery degradation policy

This policy defines the scientific and public-API gates for battery-degradation
changes. Implementation details and the current model catalog live in
[BLAST Degradation Engine](blast-degradation-engine.md) and
[Battery degradation models](../../docs/api/degradation-models.md).

## Stable public behavior

`breos.App` is the primary stable entry point. Native Naumann/Lam degradation
remains its default; BLAST is an explicit opt-in through
`degradation_engine="blast"` and a named `blast_model`. Unsupported
combinations raise rather than silently falling back to native behavior.

There is no chemistry selector. The App-level `battery_type` key was rejected
by strict validation from 0.3.4, with a targeted error from 0.4.0; 0.7.0
removed that error and the lower-level `BatteryConfig.battery_type` field,
which only ever accepted `"lfp"`. `App` now rejects the key as unknown.

Changes to these selectors, documented result/provenance fields, or default
behavior require focused `App` tests and an explicit changelog migration note.

## Default-change evidence

From 0.9.0 the residential-LFP default is `naumann_lam_field_calibrated_v2`.
`naumann_lam_field_calibrated_v1` and the unversioned
`naumann_lam_field_calibrated` select v1, the default before 0.9.0, so a
config that names a model gives the same results as before. The recorded
calibration evidence is:

| Native calibration | Full-fit RMSE | Leave-one-system-out result | Policy |
| --- | ---: | ---: | --- |
| v1 | 4.40 percentage points | mean RMSE 6.00 pp | Default before 0.9.0 |
| v2 | 4.65 percentage points | mean RMSE 5.49 pp; pooled RMSE 5.61 pp | Default from 0.9.0 |

The change does not meet the evidence bar below, and was made on these grounds:

- V1 fits all four calendar parameters to five field systems. The fit traded
  physics for in-sample error: an activation energy of 11.9 kJ/mol and a
  state-of-charge exponent of 0.10. Calendar ageing under v1 barely changes
  with temperature or with the state of charge the battery is held at, so v1
  misleads exactly the comparisons users run: climates, installation places,
  SOC windows and dispatch strategies that hold a high state of charge.
- V2 fixes those two parameters at the `naumann_lam` values: Lam's laboratory
  activation energy (38.5 kJ/mol) and an SOC exponent of 0.75 taken from
  Naumann, and fits only the rate and the time exponent. It fits the field
  systems almost as well and does better on the systems left out.

The case below compares the models on one residential system: Porto, hourly
PVGIS TMY weather, 14 Suntech 550 W modules (7.7 kWp), a 10 kWh battery,
5500 kWh a year on the bundled H0 profile, the Portuguese daily two-period
tariff and a 20-year horizon. For reference, Figgener et al. (2024) report an
average capacity loss of about 2 percentage points a year for the LFP home
storage systems in the field data set, over 3 to 8 years, with about
2 percentage points a year between similar systems.

| Model | SOH year 8 | First end of life | Replacements | SOH year 20 | NPV of savings |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native v1 | 74.7 % | 10.2 years | 1 | 70.7 % | 2498 EUR |
| Native v2 | 79.5 % | 14.4 years | 1 | 83.4 % | 2707 EUR |
| Native `naumann` (laboratory) | 84.8 % | not reached | 0 | 76.0 % | 7060 EUR |
| Native `naumann_lam` (laboratory) | 89.4 % | not reached | 0 | 81.0 % | 7143 EUR |
| BLAST `lfp_gr_sonymurata_3ah` | 90.1 % | not reached | 0 | 83.5 % | 7163 EUR |
| BLAST `lfp_gr_250ah_prismatic` | 91.9 % | not reached | 0 | 84.4 % | 7187 EUR |

Both field fits lose capacity faster than the field average over the observed
years; the field capacity tests they were fitted to read low for LFP
(Figgener et al. 2024). The laboratory and cell models lose it at or below
that average, and none of them reaches end of life in 20 years. The study
outcome turns on whether a replacement falls inside the horizon, so the
documentation asks users to run a laboratory or BLAST model beside the
default. No model here has a late-life knee or failures other than capacity
fade, and the field data stop at 8 years.

Further default changes still need the full evidence. Do not change the
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
warnings; they are not recommended pack settings. A model key does not supply
operating defaults: settings resolve as explicit user configuration over global
defaults.

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

The Python path remains the accuracy and calibration reference. The optional
Numba backend compiles the same within-day dispatch loop, keeps degradation in
Python, and is bit-identical to the reference on one toolchain (see
[Numba dispatch backend](numba-dispatch-backend.md)). The approximate
whole-project screening kernels that BREOS once had were removed in 0.6.0. A
faster path that does not reproduce the reference must not provide
calibration evidence or replace the reference path in published accuracy
comparisons.

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
[Public Release Checklist](../../maintainers/release-checklist.md); this policy does not duplicate that
release procedure.
