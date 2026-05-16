# Configuration

The {py:class}`~breos.App` constructor accepts a single `config` dict. Only
three keys are strictly required:

- `location`
- `annual_consumption_kwh`
- `n_modules` — *or* `pv_arrays` for multi-array systems

Every other key has a sensible default.

## All keys

| Key | Default | Description |
|---|---|---|
| `location` | *required* | Preset key (e.g. `"porto"`, `"berlin"`) or `{"latitude": ..., "longitude": ..., "timezone": ...}` |
| `n_modules` | *required unless `pv_arrays` is set* | Number of PV modules |
| `pv_arrays` | `None` | List of arrays with `modules`, `module`, `tilt`, and `azimuth`. When present, the array module total overrides `n_modules` |
| `annual_consumption_kwh` | *required* | Annual electricity demand (kWh) |
| `battery_kwh` | `0.0` | Battery capacity in kWh (`0` = no battery) |
| `pv_module` | `None` | Module key from the built-in catalogue. `None` uses the first available |
| `load_profile` | `"6"` | Load profile type (see {py:func}`~breos.load_profile`) |
| `tilt` | auto | Tilt angle (degrees). Auto-estimated from latitude when `None` |
| `azimuth` | auto | Surface azimuth (degrees). Auto-set to 180 in the northern hemisphere |
| `resolution` | `"h"` | Time resolution (`"h"` or `"15min"`) |
| `projection_years` | `20` | Economic projection horizon |
| `cost_preset` | `None` | Cost preset key from `configs/costs.json` |
| `inflation_rate` | `0.02` | Annual electricity price inflation |
| `discount_rate` | `0.03` | Discount rate for NPV |
| `emissions_country` | `None` | Country code for CO2 calculations (`"PT"`, `"DE"`, `"ES"`, ...) |
| `pv_degradation_rate` | `0.005` | Annual PV degradation rate (0.5% / year) |
| `calendar_model` | `"naumann_lam_field_calibrated"` | Battery calendar aging model |
| `dc_coupled` | `True` | DC-coupled / hybrid inverter (vs AC-coupled) |
| `inverter_efficiency` | `0.96` | Inverter efficiency |
| `inverter_loading_ratio` | `1.25` | DC/AC oversizing ratio |
| `start_date` | `"2023-01-01"` | First simulation date |

## Custom location

Pass an explicit coordinate dict instead of a preset key:

```python
breos.App({
    "location": {
        "latitude": 41.1579,
        "longitude": -8.6291,
        "timezone": "Europe/Lisbon",
    },
    "n_modules": 10,
    "annual_consumption_kwh": 4000,
})
```

## Multi-array PV systems

For roofs with panels facing different directions, use `pv_arrays`. Each
array is simulated independently and its DC output combined before the
energy balance — east-west or pitched-roof layouts are not collapsed into
one representative tilt/azimuth:

```python
breos.App({
    "location": "porto",
    "annual_consumption_kwh": 4000,
    "pv_arrays": [
        {"modules": 8, "module": "Erlangen_445W", "tilt": 10, "azimuth": 90},
        {"modules": 8, "module": "Erlangen_445W", "tilt": 10, "azimuth": 270},
    ],
})
```

When `pv_arrays` is set, `n_modules` is computed from the array totals and
any explicit `n_modules` key is ignored.

## Cost and emissions presets

Built-in presets live in `configs/costs.json` and `configs/emissions.json`.
Pass the key:

```python
breos.App({
    "location": "porto",
    "n_modules": 10,
    "annual_consumption_kwh": 4000,
    "cost_preset": "residential_pt",
    "emissions_country": "PT",
})
```

For full control, build a {py:class}`~breos.CostParams` and
{py:class}`~breos.EmissionsParams` yourself and call the lower-level
functions — see [Building custom pipelines](../api/index.md) once that
guide lands, or browse the [Cost analysis API](../api/cost-analysis.md).
