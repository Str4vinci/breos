# Forthcoming publication reproduction

This directory contains the versioned configurations and documentation needed
to generate the forthcoming publication's source-data bundle with BREOS 0.6.

The repository keeps `article1` in internal file names and paths for
compatibility. User-facing descriptions call this work the forthcoming
publication.

Run the complete deterministic workflow with:

```bash
uv run python tools/run_article1.py
```

Use the [source-data runbook](runbook.md) for the Monte Carlo and verification
commands. Use the [manuscript audit](manuscript-v2-audit.md) to map each result
to the generated CSV or JSON source data.

## Scope

The workflow generates:

- projected C1-C5 fixed-design energy, degradation, economic, emissions, and
  LCOE tables;
- projected GI-NPV Pareto fronts at EUR 350, 500, and 711 per kWh of storage;
- hourly-resolution, H0-load, and battery-degradation sensitivities;
- orientation and historical-weather comparison source tables; and
- seeded historical-weather Monte Carlo tables for C1-C5.

Generated results belong under the ignored `results/article1/` directory. Do
not commit the licensed E-REDES profile or third-party validation data.

## Versioned inputs

[`article1-projected-optimization.toml`](article1-projected-optimization.toml)
defines the deterministic projected analyses. It uses a repeated TMY over 20
years, two projected objectives, explicit system constraints, battery-state
carry, and replacement costs from the yearly simulated ledger.

[`article1-montecarlo.toml`](article1-montecarlo.toml) defines the historical-
weather study. It uses 10,000 seeded 20-year trajectories and the manuscript's
uniform annual load multiplier from 0.95 through 1.05.

Both configurations identify the Suntech module and explicitly record its
1.134 m by 2.278 m frame. Every generated provenance file records the complete
resolved module parameters, including the module power temperature
coefficient of -0.34 %/°C and open-circuit-voltage coefficient of
-0.26 %/°C (-0.13 V/°C after rounding at the rated Voc).

Both primary configurations derive battery ambient temperature from weather
and apply the indoor-temperature model for a residential installation. The
`no-thermal-model/` configuration pair retains the manuscript's fixed 25 °C
assumption as a control. Use `--calendar-model` and `--config-dir` to select a
control run explicitly; these overrides require a separate output directory.

## External inputs

The publication workflow requires:

- the licensed `EREDES_2025_BTN_1000kwh_15min.csv` household profile;
- the historical Porto weather CSV used by the Monte Carlo and weather
  comparison.

`tools/preflight_article1_inputs.py` verifies the expected hashes and writes
an input manifest without running a simulation. The bundled Porto TMY is
version controlled under `validation/data/weather/`.

The public workflow excludes the measured, PVsyst, and Polysun comparison data
used for the forthcoming publication's Esposende validation. Those files remain
in the private study archive and are not required, copied, or verified by BREOS.
A reader can still run the BREOS side with the `esposende` location preset and
locally supplied load and weather inputs. Open-Meteo can supply historical
weather, but an exact comparison requires the same saved weather file and load
profile used by the study.

## Archived comparison

The private research reference is the forthcoming publication study run under
`dev/results/a1_july_rerun_tuxedo/moo_15min` at research revision
`a0db6aae1e8d04a8260f51a34543b23bd82a1762`. Its Pareto CSV SHA-256 is
`5334b8361b2395f0f19b6839005964b0b61bfa0d00e5ea28f450cfb4cde0a225`.

The archived workflow contained methodological issues that BREOS 0.6 does not
preserve: it repeated hourly irradiance in the nominal 15-minute run, used a
flat unbounded inverter conversion, and sampled normally distributed load in
the Monte Carlo despite the manuscript specifying a bounded uniform
distribution. The archived deterministic and Monte Carlo configurations also
used raw ambient weather for battery temperature. BREOS 0.6 instead models the
indoor thermal environment in the primary configurations and retains the
manuscript's fixed 25 °C assumption as a separate control. BREOS 0.6 also fixes
the final-hour Makima interpolation gap. These changes can alter every reported
result.

Earlier corrected values generated before the final module and resampling
fixes are intentionally not reported here. Add final numerical comparisons
only after the runbook has been executed from one clean release commit and
`tools/verify_article1_bundle.py` passes.

## Publication provenance

Each deterministic run writes `reproduction.json`; each context analysis and
Monte Carlo case writes `provenance.json`. These files record the BREOS
version and commit, dirty-worktree state, command, resolved configuration,
dependency versions, input hashes, output hashes, and resolved PV module.

Keep the complete generated result directory with the submitted manuscript
and deposit it with the publication data. Cite the released BREOS archive and
identify the licensed and third-party inputs separately.
