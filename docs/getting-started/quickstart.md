# Quickstart

Run one simulation of a 10-module PV system with a 5 kWh battery for a house
in Porto, read the summary, then save the setup in a config file. You need
BREOS [installed](installation.md) and internet access: the run fetches a
typical weather year for Porto from PVGIS.

## Run your first simulation

```bash
breos run --location porto \
  --n-modules 10 \
  --annual-consumption-kwh 4000 \
  --battery-kwh 5 \
  --cost-preset residential-pt \
  --emissions-country pt
```

After a few seconds the terminal shows a summary of the run. Below a header
line with the BREOS version, the location and the weather source, it reads:

```text
System: 5.5 kWp PV (10 modules), 4.4 kW inverter, 5 kWh battery (DC)

Year 1 energy
  PV delivered (AC)             8143 kWh
  House load                    4000 kWh
  Self-consumed PV              3157 kWh       38.8 % of delivered PV
  Grid import                    839 kWh       79.0 % grid independence
  Grid export                   4986 kWh

Battery over 20 years
  Health at the end             86.2 %         replaced in year 16

Economics, 20 years, EUR, 3 % discount rate
  Investment                    7789 EUR
  Net present value             4888 EUR
  Discounted payback              10 years
  LCOE                         0.101 EUR/kWh
```

Your last digits can differ: PVGIS updates its typical year from time to
time. [Interpreting results](interpreting-results.md) explains each line.
Results of BREOS 0.6.2 and earlier are not directly comparable; see
[Migrating from BREOS 0.6.2](interpreting-results.md#migrating-from-breos-062).

## Save the setup in a config file

Put the same inputs in `quickstart.toml`, with the defaults the command above
used written out:

```toml
location = "porto"
n_modules = 10
annual_consumption_kwh = 4000
battery_kwh = 5.0
load_profile = "demandlib_h0"
cost_preset = "residential_pt"
emissions_country = "PT"
projection_years = 20
resolution = "h"
```

Check the file, then run it and keep the full results:

```bash
breos validate-config quickstart.toml
breos run --config quickstart.toml \
  --output result.json
```

`validate-config` checks the keys and prints the resolved choices without
fetching weather. `--output` writes every result to `result.json`: the
headline values, plus `yearly`, `monthly` and `financial` detail. The terminal
still shows the summary. Source checkouts include this file as
`configs/examples/quickstart.toml`.

To see every default BREOS fills in (module, tilt, inverter rating, losses),
add `--dry-run`. It prints the resolved configuration as JSON and does not
simulate.

To print the full results instead of the summary, pipe the command or add
`--json`:

```bash
breos run --config quickstart.toml \
  | jq .npv_savings
breos run --config quickstart.toml --json
```

## Try other inputs

The packaged locations, modules, cost presets, grid-emission countries and
load profiles are listed by:

```bash
breos list locations
breos list modules
breos list cost-presets
breos list emissions
breos list load-profiles
```

The same lists, with full details, live on the
[packaged options](options.md) page.

## The same run in Python

```python
import breos

app = breos.App({
    "location": "porto",              # preset or {"latitude": ..., "longitude": ..., "timezone": ...}
    "n_modules": 10,
    "annual_consumption_kwh": 4000,
    "battery_kwh": 5.0,               # 0 for no battery
    "cost_preset": "residential_pt",
    "emissions_country": "PT",
})

app.simulate()
result = app.result()

print(f"Grid independence: {result['grid_independence_pct']:.1f} %")
print(f"Usable AC production: {result['usable_ac_system_production_kwh']:.0f} kWh")
print(f"Payback: {result['payback_year']} years")
print(f"NPV savings: {result['npv_savings']:.0f} EUR")
print(f"CO2 avoided: {result['co2_avoided_total_lifetime_kg'] / 1000:.1f} t")
```

What just happened, end-to-end:

1. **Weather** — BREOS loaded a Typical Meteorological Year (TMY) for Porto
   from a local cache if present, otherwise fetched it from PVGIS.
2. **PV** — Ten panels at the configured tilt and azimuth produced a
   year-long DC power series. BREOS uses pvlib under the hood for solar
   position, irradiance transposition, cell temperature, and PV performance
   model pieces; see [PV](../api/pv.md) and [Resources](../resources.md) for
   further reading.
3. **Load** — A bundled demandlib-derived H0 residential profile scaled to
   4000 kWh/yr.
4. **Energy balance** — Per-timestep energy-flow accounting determined how
   much PV is self-consumed, stored, or exported.
5. **Battery degradation** — Twenty years of operation with field-calibrated
   calendar aging (v2) and Naumann's LFP cycle aging.
6. **Economics** — Costs derived from the `residential_pt` preset, NPV /
   LCOE / payback computed over the projection horizon.
7. **Emissions** — CO2 savings computed against the Portuguese grid
   intensity.

## Next steps

- See the [case examples](../gallery/index.rst) for stored runs that answer
  common questions: what a battery adds, east-west roofs, tariffs, battery
  replacement, Monte Carlo and the optimizer front.
- See the [how-to guides](../how-to/index.md) for custom coordinates, your
  own PV module, partial years, sweeps, 15-minute resolution, external load
  profiles and offline runs.
- See [Configuration](configuration.md) for every option `App` accepts.
- See [Interpreting results](interpreting-results.md) for what each result
  key means.
- See [Troubleshooting](troubleshooting.md) for weather access, preset,
  optional-dependency, cache, and runtime problems.
- See the [API reference](../api/index.md) for the lower-level functions
  the `App` calls under the hood.
