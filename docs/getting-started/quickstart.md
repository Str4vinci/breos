# Quickstart

The fastest way to run a BREOS simulation is the {py:class}`~breos.App`
facade. It takes a single config dict, runs the full PV + battery +
economics + emissions pipeline, and returns a plain JSON-serializable
result dict.

The example uses packaged defaults and public weather access. For real
projects, review [Required Inputs](inputs.md) before interpreting results.

## A 10-minute first run

Create a small `quickstart.toml` config:

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

Then validate it, inspect the resolved defaults, and run the simulation:

```bash
breos validate-config quickstart.toml
breos run --config quickstart.toml --dry-run
breos run --config quickstart.toml --output result.json
```

`validate-config` checks required keys and prints the resolved choices without
fetching weather or running a simulation. `--dry-run` writes the same resolved
configuration summary as JSON. A successful dry run shows the location,
timezone, module count, PV size, inverter AC rating, load profile, battery
capacity, cost preset, emissions preset, and resolved static PVWatts loss
components. The loss block applies any `pv_loss_overrides` and reports the
combined percentage, so you can verify shading/soiling/wiring assumptions before
fetching weather.

The final command runs the full simulation and writes a JSON object with
scalar headline keys plus `yearly`, `monthly`, and `financial` detail blocks.
It may fetch PVGIS TMY weather data and needs internet access unless a matching
local weather cache is present.

Source checkouts also include the same example at
`configs/examples/quickstart.toml`.

A representative run (PVGIS TMY for Porto, packaged defaults) produces
top-level values close to these:

```json
{
  "n_modules": 10,
  "pv_kwp": 5.5,
  "battery_kwh": 5.0,
  "pv_production_kwh": 8288.0,
  "grid_independence_pct": 80.2,
  "self_consumption_pct": 39.8,
  "total_investment_eur": 7788.9,
  "payback_year": 10,
  "npv_savings_eur": 5041.3,
  "battery_soh_end_pct": 70.6,
  "co2_avoided_total_kg": 20228.0
}
```

Exact numbers shift with the PVGIS TMY vintage and dependency versions, but a
plausible first run lands in the same neighborhood — roughly 8 MWh/yr of PV
production and 75–85% grid independence for this config. If your values are
far off, use the dry-run summary and [Required Inputs](inputs.md) to check
which defaults your run actually used.

To discover packaged option keys:

```bash
breos list locations
breos list modules
breos list cost-presets
breos list emissions
breos list load-profiles
```

The same lists, with full details, live on the
[packaged options](options.md) page.

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
   year-long DC power series. BREOS uses pvlib under the hood for solar
   position, irradiance transposition, cell temperature, and PV performance
   model pieces; see [PV](../api/pv.md) and [Resources](../resources.md) for
   further reading.
3. **Load** — A bundled demandlib-derived H0 residential profile scaled to
   4000 kWh/yr.
4. **Energy balance** — Per-timestep energy-flow accounting determined how
   much PV is self-consumed, stored, or exported.
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
breos run --config quickstart.toml --output result.json
```

The CLI writes the same JSON-serializable dict that `App.result()` returns.

## Next steps

- See [Recipes](recipes.md) for copy-paste configs: PV-only, east-west
  roofs, custom coordinates, 15-minute resolution, external load profiles,
  and offline runs.
- See [Configuration](configuration.md) for every option `App` accepts.
- See [Interpreting results](interpreting-results.md) for what each result
  key means.
- See the [API reference](../api/index.md) for the lower-level functions
  the `App` calls under the hood.
