# Article 1 projected-optimization reproduction

This report compares four archived Article 1 Pareto candidates with the
corrected projected optimizer prepared for the BREOS 0.6.0 release line. The
new results deliberately do not reproduce known errors in the research run.

## Provenance

- Archived run: `dev/results/a1_july_rerun_tuxedo/moo_15min` in the private
  research repository.
- Earlier reference tag supplied for comparison: `a1_jun_26_submission`
  (`6af21cc61640da7fc003d79319527631ec7cbddd`). The later July rerun above is
  the Article 1 result set used for the numerical comparison.
- Research revision: `a0db6aae1e8d04a8260f51a34543b23bd82a1762`.
- Archived Pareto SHA-256:
  `5334b8361b2395f0f19b6839005964b0b61bfa0d00e5ea28f450cfb4cde0a225`.
- Archived weather SHA-256:
  `d2258dc7ea0d6432a6ddf69f748e67e88a36b235a906d5c6ab7da96e8e6911e0`.
- Public BREOS base revision:
  `8fd1f21c4f1335350d87f91d76246b3fd1184f6d` (`origin/develop`).
- Tested BREOS feature revision:
  `8dcf17574ca440f77b9092651035ae32b5e6f998`, with a clean tracked
  worktree recorded by the reproduction tool.
- Reproduction config SHA-256:
  `c44ba45c2d3d597275ef8fb66d7403551c6489af88545238f93c64c1b1f8a44f`.
- Bundled weather SHA-256 (compressed):
  `bf84e31b02ad9bf39f331a5ce8629b1ea8f80cd1597748e72a87a0fce56b4f15`.
- Bundled weather SHA-256 (uncompressed):
  `79af3f5edbf5040409506178b17e25a48ce1c426cb8eaff51991afb6e1199126`.
- External E-REDES RLP SHA-256:
  `23becc5a7bfc927b1f7604156e0e4953dcc6bb65268ca947b38db3dc4f2b28bc`.

The E-REDES profile is not redistributed. Pass the directory containing the
licensed file to the reproduction command. `reproduction.json` records the
fully resolved config, command line, Python and BREOS versions, exact source
revision, dirty-worktree flag, and all input hashes for every run.

The nested optimizer config lives here as
[`article1-projected-optimization.toml`](article1-projected-optimization.toml).
It is not an App/`breos run` config; `tools/reproduce_article1.py` is its public
entrypoint.

## Fixed candidates

| Candidate | Design (modules, battery, tilt, azimuth) | Archived GI | BREOS GI | Archived NPV | BREOS NPV |
| --- | --- | ---: | ---: | ---: | ---: |
| C1 | 6, 0 kWh, 30°, 190° | 41.056152% | 40.688173% | €5,424.43 | €5,374.16 |
| C2 | 9, 5 kWh, 25°, 185° | 64.768329% | 63.885473% | €3,779.25 | €3,621.66 |
| C3 | 9, 9 kWh, 35°, 190° | 78.683166% | 77.600575% | €2,642.02 | €2,456.26 |
| C4 | 9, 20 kWh, 45°, 175° | 89.390357% | 88.749742% | −€5,094.04 | −€5,263.66 |

[`fixed-candidate-comparison.csv`](fixed-candidate-comparison.csv) contains
the full-precision values and absolute and relative differences.

A two-process NSGA-II smoke run completed one generation with four evaluated
candidates. Its output used exactly two objectives (`Projected_Grid_Independence_%`
and `Projected_NPV_Eur`), retained projected ZEB only as a diagnostic, and
produced Pareto CSV SHA-256
`e458780b84d477bda5ed726dfdcb7b87a7a357bf0c97a54d6443126a98674645`.

## Full projected optimization

The exact Article 1 NSGA-II configuration completed all 40 configured
generations (2,050 evaluations) with seed 1 and eight worker processes. It
returned 100 unique nondominated designs using only projected GI and projected
NPV as objectives. Early stopping did not trigger before the generation cap.

The corrected front spans 40.825173% to 88.819590% projected GI and
−€5,280.60 to €5,379.67 projected NPV. Its maximum-NPV design has 6 modules,
no battery, 25° tilt, and 195° azimuth. Its maximum-GI design has 9 modules, a
20 kWh battery, 50° tilt, and 185° azimuth. A normalized closest-to-ideal
summary point has 9 modules, a 9 kWh battery, 30° tilt, and 190° azimuth, with
77.617510% projected GI and €2,454.04 projected NPV. This summary point is not
an additional optimization objective or a prescribed design choice.

The generated 100-row `pareto_results.csv` has SHA-256
`24dac6a404c3d17a5ee0f3de4589d3bec9a6bcca3c78eeb3f0504b1f803c8a98`.
The accompanying `reproduction.json` has SHA-256
`bdcc906dfab1f55324363f44dad4144a167193f9fea29ad2dd6b4c37685233e7`.
That JSON records the resolved configuration, environment, inputs, clean
source revision, worker count, objective names, generation count, and Pareto
digest. Generated outputs are deliberately not committed; rerun the command
below to create them in the chosen output directory.

The reproduction command also exports a plot-independent source bundle for
each fixed case: projected metrics, the yearly energy/degradation-state table,
and the financial ledger. C5 (4 modules, no battery, 35° tilt, 180° azimuth) is
included as the manuscript's low-investment benchmark. It was not part of the
four-candidate full-precision comparison above.

## Why the results differ

The archived run has a resolution error. Its TMY loader returned 8,760 hourly
rows even when the optimization requested a 15-minute resolution. The PV
calculation then constructed a 15-minute index and selected the nearest hourly
irradiance value at every step. For C1, this repetition produces 4,943.362 kWh
after a flat 96% inverter efficiency, which matches the archived 4,943.364 kWh.

The Article 1 reproduction uses clear-sky-aware interpolation and then
renormalizes each source hour's four GHI, DNI, and DHI values so their mean
equals the original hourly value. This preserves the source irradiance energy
while improving intra-hour solar shape. The energy-conserving option is
explicit and leaves the established BREOS resampling default unchanged.

BREOS uses the corrected explicit AC dispatch ledger. A finite inverter
rating applies the PVWatts part-load curve and clipping instead of an unbounded
flat 96% conversion. For C1, the first-year result is 5,182.097 kWh DC and
4,962.091 kWh usable AC, including 215.926 kWh conversion loss and 4.080 kWh
curtailed DC. Battery candidates additionally carry stored energy,
PV-origin energy, degradation state, and resistance growth across years.

The outer archived MOO provenance records `other_cost_per_module = 50`, but
the projection template that actually scored candidates resolved to €30 per
module. The public reproduction uses €30, matching the archived candidate
CAPEX (for example, €2,820.1968 for C1). This is an archived provenance defect,
not a reason to alter the current cost engine.

The bundled public TMY is a normalized, rounded form of the archived PVGIS
file and therefore has a different digest. Repeating C1 with the exact archived
weather changes the result by only +0.0145 percentage points GI and about
+€2.40 NPV. It does not explain the archived discrepancy.

## Commands

Run all fixed candidates:

```bash
uv run python tools/reproduce_article1.py \
  --rlp-directory /path/to/licensed/rlp \
  --output results/article1
```

Run the exact NSGA-II settings after reviewing the fixed-candidate comparison:

```bash
uv run python tools/reproduce_article1.py \
  --rlp-directory /path/to/licensed/rlp \
  --full-optimization \
  --n-procs 8 \
  --output results/article1
```

Run the field-v2 fixed cases and full optimization into a separate directory:

```bash
uv run python tools/reproduce_article1.py \
  --rlp-directory /path/to/licensed/rlp \
  --calendar-model naumann_lam_field_calibrated_v2 \
  --full-optimization \
  --n-procs 8 \
  --output results/article1-v2
```

Run the laboratory-parameter sensitivity in the same way:

```bash
uv run python tools/reproduce_article1.py \
  --rlp-directory /path/to/licensed/rlp \
  --calendar-model naumann_lam \
  --full-optimization \
  --n-procs 8 \
  --output results/article1-lab
```

Add `--candidate C2` and omit `--full-optimization` when only the fixed C2
degradation-model comparison is required.

Run the opt-in numerical regression locally:

```bash
BREOS_ARTICLE1_RLP_DIRECTORY=/path/to/licensed/rlp \
  uv run pytest -q tests/test_article1_reproduction.py
```
