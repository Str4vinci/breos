# Article 1 reproduction

This directory contains the versioned configurations and documentation needed
to generate the Article 1 source-data bundle with BREOS 0.6.

Use the [source-data runbook](runbook.md) for the complete ordered command
sequence. Use the [manuscript audit](manuscript-v2-audit.md) to map each result
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
resolved module parameters, including the Article power temperature
coefficient of -0.34 %/°C.

## External inputs

The Article workflow requires:

- the licensed `EREDES_2025_BTN_1000kwh_15min.csv` household profile;
- the historical Porto weather CSV used by the Monte Carlo and weather
  comparison; and
- the monthly, weekly, and daily measured/PVsyst/Polysun comparison CSVs used
  for Figure 2.

`tools/preflight_article1_inputs.py` verifies the expected hashes and writes
an input manifest without running a simulation. The bundled Porto TMY is
version controlled under `validation/data/weather/`.

## Archived comparison

The private research reference is the Article run under
`dev/results/a1_july_rerun_tuxedo/moo_15min` at research revision
`a0db6aae1e8d04a8260f51a34543b23bd82a1762`. Its Pareto CSV SHA-256 is
`5334b8361b2395f0f19b6839005964b0b61bfa0d00e5ea28f450cfb4cde0a225`.

The archived workflow contained methodological issues that BREOS 0.6 does not
preserve: it repeated hourly irradiance in the nominal 15-minute run, used a
flat unbounded inverter conversion, and sampled normally distributed load in
the Monte Carlo despite the manuscript specifying a bounded uniform
distribution. BREOS 0.6 also fixes the final-hour Makima interpolation gap.
These changes can alter every reported result.

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
