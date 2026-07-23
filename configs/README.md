# Configuration

You do **not** need this directory to run BREOS — the defaults (locations, costs,
emissions, PV modules, the bundled load profile) are packaged inside the
installed `breos` package. This folder exists so you can read those defaults and
keep your own runnable example configs.

```
configs/
├── base/        # editable copies of the packaged JSON presets (reference only)
└── examples/    # runnable CLI configs for `breos run`, `sweep`, and `montecarlo`
```

- **`base/`** mirrors the packaged presets (`locations`, `costs`, `emissions`,
  `financials`, `electricity`). Read them to see what BREOS ships and to copy
  values into your run config. The CLI always loads its own packaged copies, so
  editing files here is for reference — it does not change a run.
- **`examples/`** holds CLI configs — mostly single-run inputs for `breos run`,
  plus dedicated `sweep` and Monte Carlo examples.

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
resampling a weather year and a demand multiplier for each projection year, and
writes one row per run to `monte_carlo_results.csv`. Pass `--plots` to generate
payback, NPV, grid-independence, final-SoH, and LCOE distributions in `plots/`.
It needs a multi-year historical weather CSV referenced by the `[montecarlo]`
section — BREOS does not bundle weather data. Drop your file in a local
`weather/` directory (git-ignored) and see
[`examples/montecarlo.toml`](examples/montecarlo.toml).

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
| [`quickstart.toml`](examples/quickstart.toml) | Minimal happy-path run (Porto, PV + battery) |
| [`pv-plus-battery.toml`](examples/pv-plus-battery.toml) | **Annotated reference** — every available key with its default |
| [`pv-only.toml`](examples/pv-only.toml) | Baseline with no battery, to compare storage scenarios against |
| [`germany-berlin.toml`](examples/germany-berlin.toml) | Swapping location + cost preset + emissions factor together |
| [`east-west-roof.toml`](examples/east-west-roof.toml) | Multiple `[[pv_arrays]]` (split east/west roof) |
| [`sweep.toml`](examples/sweep.toml) | Parameter grid over module count and battery size (`breos sweep`) |
| [`montecarlo.toml`](examples/montecarlo.toml) | Monte Carlo over weather years + demand (`breos montecarlo`) |
| [`external-rlp.toml`](examples/external-rlp.toml) | Using non-bundled, licensed load profiles |

Start from `pv-plus-battery.toml` if you want to see the full set of knobs; copy
any example and edit it for your own scenario.

## Notes

- Keep public examples on the bundled `load_profile = "demandlib_h0"` (canonical
  key `"1"`) unless the example explicitly documents an external, user-licensed
  RLP directory.
- For external RLPs, use [`examples/external-rlp.toml`](examples/external-rlp.toml)
  as a template and put the licensed CSV files in a local directory such as
  `external_rlp/` (do not commit third-party RLPs).
- `breos run` configs are **flat** key/value files (TOML or JSON). The only
  recognised table for `breos run` is `[[pv_arrays]]`. The `[sweep]` and
  `[montecarlo]` tables are read by their dedicated CLI commands.
- Configs written for the research `pvbat` engine — with nested model sections,
  inheritance, or simulation-type blocks — are **not** compatible. BREOS
  rejects unknown top-level keys rather than silently applying defaults.
  Translate the values you need into the flat keys shown in
  `pv-plus-battery.toml` (and `[montecarlo]` for MC studies).
