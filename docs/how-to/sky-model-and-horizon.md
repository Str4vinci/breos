# Choose a sky model and a terrain horizon

## Anisotropic sky-diffusion model

The default `isotropic` transposition underestimates plane-of-array
irradiance on clear days. Switch to an anisotropic model — here Perez — to
capture circumsolar and horizon brightening. No extra weather inputs are
needed; see [Sky-diffusion model](../getting-started/configuration.md#sky-diffusion-transposition-model):

```toml
location = "porto"
n_modules = 10
annual_consumption_kwh = 4000
cost_preset = "residential_pt"
emissions_country = "PT"
transposition_model = "perez"
surface_type = "grass"          # or a numeric albedo, e.g. albedo = 0.2
```

`surface_type` (or a numeric `albedo`) sets the ground reflectance that feeds
the ground-diffuse component; a snowy or sandy foreground (`"snow"`, `"sand"`)
raises annual yield further. Leave both unset to keep pvlib's 0.25 default.

From the CLI, the equivalent flag is `--transposition-model perez`
(alias `--sky-model`).

## Apply a custom terrain horizon

Use azimuth/elevation pairs to model far-horizon direct-beam obstruction
without a 3D scene:

```toml
location = "porto"
n_modules = 8
annual_consumption_kwh = 4000

horizon_profile = [
  [0, 4],
  [60, 9],
  [120, 14],
  [180, 3],
  [270, 6],
]
```

The profile is circular: BREOS interpolates from the last point back to the
first across north, so a repeated `360` endpoint is unnecessary. With a fresh
PVGIS fetch, configuring the profile automatically requests provider data
without PVGIS's own horizon. Cached weather must have a matching provenance
sidecar that explicitly records `not_applied`; legacy CSVs are rejected because
their horizon treatment cannot be established safely. The normalized profile
and number of shaded timesteps appear under
`provenance.weather.horizon.profile`.
