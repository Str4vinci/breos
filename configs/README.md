# Configuration

You do **not** need this directory to run BREOS — the defaults (locations, costs,
emissions, PV modules, the bundled load profile) are packaged inside the
installed `breos` package. This folder holds runnable example configs to copy
and edit; `breos list` shows the packaged presets.

```
configs/
├── examples/      # runnable CLI configs for `breos run`, `sweep`, and `montecarlo`
└── optimization/  # nested configs for the Python optimization API
```

- **`examples/`** holds CLI configs: single-run inputs for `breos run`, plus
  dedicated `sweep` and Monte Carlo examples. Every file here validates with
  `breos validate-config`.
  `breos run` and `breos sweep` download a PVGIS TMY for the location unless a
  matching `weather/<location>_tmy_<years>_<source>.csv` file is in the
  directory you run from.
- **`optimization/`** holds configs for the optimization API. These use a
  different, nested shape and you load them from Python, so `breos run` and
  `breos validate-config` reject them.

## Running a simulation

```bash
# Run a packaged example
breos run --config configs/examples/quickstart.toml

# Check a config without running it
breos validate-config configs/examples/quickstart.toml

# Override any key on the command line
breos run --config configs/examples/pv-only.toml --battery-kwh 5

# Monte Carlo over weather years + demand (needs a multi-year weather file)
breos montecarlo --config configs/examples/montecarlo.toml --runs 100 --plots

# Parameter grid over a base scenario
breos sweep --config configs/examples/sweep.toml --output sweep_results.csv
```

### Monte Carlo

`breos montecarlo` runs the scenario as repeated multi-year projections,
resampling a weather year and a demand multiplier for each projection year. It
writes one row per run to `monte_carlo_results.csv` and a provenance JSON with
the resolved settings and input/output hashes. Pass `--collect-yearly` to also
write the per-run, per-year energy, degradation, and discounted-cost ledger
needed for cost envelopes. Pass `--plots` to generate
payback, NPV, grid-independence, final-SoH, and LCOE distributions in `plots/`.
It needs a multi-year historical weather CSV referenced by the `[montecarlo]`
section — BREOS does not bundle weather data. Drop your file in a local
`weather/` directory (git-ignored) and see
[`examples/montecarlo.toml`](examples/montecarlo.toml).

The established demand multiplier is normal with `load_uncertainty` as its
standard deviation. Set `load_distribution = "uniform"` to use
`[1 - load_uncertainty, 1 + load_uncertainty]`. Weather-year bounds
(`weather_start_year`, `weather_end_year`), demand-multiplier bounds
(`min_load_scale`, `max_load_scale`), and the worker count (`n_procs`) are
also `[montecarlo]` settings. The hourly-to-15-minute irradiance policy is the
top-level `irradiance_resampling` key, as for `breos run`.

The catalogue keys used in a config (`location`, `pv_module`, `cost_preset`,
`emissions_country`, `load_profile`) come from the packaged presets. List the
valid values with:

```bash
breos list locations
breos list modules
breos list cost-presets
breos list emissions
breos list load-profiles
```

## Example configs

| File | What it shows |
| --- | --- |
| [`quickstart.toml`](examples/quickstart.toml) | Minimal PV + battery run for Porto, as in the quickstart guide |
| [`pv-plus-battery.toml`](examples/pv-plus-battery.toml) | **Annotated reference** — the common keys with their defaults |
| [`pv-only.toml`](examples/pv-only.toml) | Baseline with no battery, to compare storage scenarios against |
| [`germany-berlin.toml`](examples/germany-berlin.toml) | Swapping location + cost preset + emissions factor together, on flat prices |
| [`east-west-roof.toml`](examples/east-west-roof.toml) | Multiple `[[pv_arrays]]` (split east/west roof) |
| [`bifacial-ground-mount.toml`](examples/bifacial-ground-mount.toml) | Opt-in infinite-sheds rear gain with explicit row geometry |
| [`recommended-pv.toml`](examples/recommended-pv.toml) | Higher-fidelity rooftop PV-model choices, set explicitly; the defaults stay unchanged |
| [`sweep.toml`](examples/sweep.toml) | Parameter grid over module count and battery size (`breos sweep`) |
| [`time-of-use-portugal.toml`](examples/time-of-use-portugal.toml) | Time-of-use `[tariff]` on a bundled Portuguese schedule; the battery dispatch is unchanged |
| [`custom-tariff-schedule.toml`](examples/custom-tariff-schedule.toml) | A schedule BREOS does not bundle, defined inline: periods, rules and explicit holidays |
| [`quarterly-tariff-berlin.toml`](examples/quarterly-tariff-berlin.toml) | The `germany-berlin.toml` system on calendar-month (quarterly) seasons with illustrative §14a-style windows |
| [`smart-charging-portugal.toml`](examples/smart-charging-portugal.toml) | The `time-of-use-portugal.toml` system with fixed-target grid charging off peak |
| [`tariff-comparison.toml`](examples/tariff-comparison.toml) | Three offers, with and without a battery, in one `breos sweep` |
| [`montecarlo.toml`](examples/montecarlo.toml) | Monte Carlo over weather years + demand (`breos montecarlo`) |
| [`external-rlp.toml`](examples/external-rlp.toml) | Using non-bundled, licensed load profiles |

Start from `pv-plus-battery.toml` to see the common knobs, and read the
[configuration key reference](../docs/getting-started/config-reference.md) for
every key BREOS accepts. Copy any example and edit it for your own scenario.

## Optimization configs

`breos run` picks one design and simulates it. The optimization API searches for
a design instead, and it takes a nested config that the CLI does not accept.

| File | What it shows |
| --- | --- |
| [`projected-optimization.toml`](optimization/projected-optimization.toml) | Projected-lifetime NSGA-II sizing over module count, battery size, tilt, and azimuth |

Load it from Python and pass it to
`breos.optimization.optimize_system_multi_objective`. Needs the pymoo extra
(`pip install "breos[optimization]"`). The full walkthrough is in the
[Optimization guide](../docs/getting-started/optimization.md).

## Notes

- Keep public examples on the bundled `load_profile = "demandlib_h0"` unless the
  example explicitly documents an external, user-licensed RLP directory.
- For external RLPs, use [`examples/external-rlp.toml`](examples/external-rlp.toml)
  as a template and put the licensed CSV files in a local directory such as
  `external_rlp/` (do not commit third-party RLPs).
- `breos run` configs are mostly flat key/value files (TOML or JSON).
  `[[pv_arrays]]` describes multiple arrays and `[costs]` holds explicit cost
  overrides. The `[sweep]` and `[montecarlo]` tables are read by their dedicated
  CLI commands; sweep entries can use quoted dotted keys such as
  `"costs.electricity_cost"`.
- BREOS rejects unknown top-level keys rather than silently applying
  defaults, so configs with nested model sections, inheritance, or
  simulation-type blocks are **not** accepted. Translate the values you need
  into the flat keys shown in `pv-plus-battery.toml` (and `[montecarlo]` for
  Monte Carlo studies).
