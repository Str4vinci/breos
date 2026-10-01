# Run at 15-minute resolution

Hourly weather is reconstructed at quarter-hour steps under the
`irradiance_resampling` policy (see
[Hourly weather at 15-minute resolution](../getting-started/configuration.md#hourly-weather-at-15-minute-resolution)),
and the bundled H0 profile has a native 15-minute variant. An external profile supplied only
as hourly means is interpolated between hour midpoints and keeps each hour's
mean exactly. Simulations take correspondingly
longer:

```toml
location = "porto"
n_modules = 10
annual_consumption_kwh = 4000
battery_kwh = 5.0
resolution = "15min"
cost_preset = "residential_pt"
emissions_country = "PT"
```
