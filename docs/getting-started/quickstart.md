# Quickstart

The fastest way to run a BREOS simulation is the {py:class}`~breos.App`
facade. It takes a single config dict, runs the full PV + battery +
economics + emissions pipeline, and returns a plain JSON-serializable
result dict.

The example uses packaged defaults and public weather access. For real
projects, review [Required Inputs](inputs.md) before interpreting results.

## A 10-minute first run

The repository includes a runnable example config:

```bash
breos validate-config configs/examples/quickstart.toml
breos run --config configs/examples/quickstart.toml --dry-run
breos run --config configs/examples/quickstart.toml --output result.json
```

`validate-config` checks required keys and prints the resolved choices without
fetching weather or running a simulation. `--dry-run` writes the same resolved
configuration summary as JSON. A successful dry run shows the location,
timezone, module count, PV size, inverter AC rating, load profile, battery
capacity, cost preset, and emissions preset.

The final command runs the full simulation and writes a JSON object with
top-level keys such as `pv_production_kwh`, `grid_independence_pct`,
`self_consumption_pct`, `payback_year`, `npv_savings_eur`, `monthly`,
`financial`, and `yearly`. Exact values depend on the weather source and model
assumptions, so use the dry-run summary and [Required Inputs](inputs.md) to
decide which defaults are acceptable for your study.

To discover packaged option keys:

```bash
breos list locations
breos list modules
breos list cost-presets
breos list emissions
breos list load-profiles
```

## A minimal Python example

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

print(f"Grid independence: {result['grid_independence_pct']:.1f}%")
print(f"Payback: {result['payback_year']} years")
print(f"NPV savings: {result['npv_savings_eur']:,.0f} EUR")
print(f"CO2 avoided: {result['co2_avoided_total_kg']:,.0f} kg")
```

What just happened, end-to-end:

1. **Weather** — BREOS loaded a Typical Meteorological Year (TMY) for Porto
   from a local cache if present, otherwise fetched it from PVGIS.
2. **PV** — Ten panels at the configured tilt and azimuth produced a
   year-long DC power series.
3. **Load** — A bundled demandlib-derived H0 residential profile scaled to
   4000 kWh/yr.
4. **Energy balance** — Per-timestep dispatch decided how much PV is
   self-consumed, stored, or exported.
5. **Battery degradation** — Twenty years of operation with Naumann calendar
   aging and field-calibrated LFP cycle aging.
6. **Economics** — Costs derived from the `residential_pt` preset, NPV /
   LCOE / payback computed over the projection horizon.
7. **Emissions** — CO2 savings computed against the Portuguese grid
   intensity.

## Running from the command line

The same simulation can be run without writing Python:

```bash
breos run \
  --location porto \
  --n-modules 10 \
  --annual-consumption-kwh 4000 \
  --battery-kwh 5.0 \
  --cost-preset residential-pt \
  --emissions-country pt \
  --output result.json
```

Or with a TOML or JSON config file:

```bash
breos run --config configs/examples/quickstart.toml --output result.json
```

The CLI writes the same JSON-serializable dict that `App.result()` returns.

## Next steps

- See [Configuration](configuration.md) for every option `App` accepts.
- See [Interpreting results](interpreting-results.md) for what each result
  key means.
- See the [API reference](../api/index.md) for the lower-level functions
  the `App` calls under the hood.
