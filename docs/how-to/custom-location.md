# Simulate any site by coordinates

Any site works without a packaged preset — pass coordinates and an IANA
timezone instead of a location key:

```toml
location = { latitude = 48.2082, longitude = 16.3738, timezone = "Europe/Vienna" }
n_modules = 12
annual_consumption_kwh = 4500
battery_kwh = 5.0
cost_preset = "residential_de"
emissions_country = "AT"
```

Tilt and azimuth are auto-estimated from the latitude when not set. There is
no Austrian cost preset yet, so this example borrows the German one — replace
it with your own tariffs for real economics.
