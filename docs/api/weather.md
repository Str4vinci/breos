# Weather

Sources, loaders, and resampling utilities for solar irradiance and
temperature time series.

Local weather loading and PVGIS/NSRDB TMY helpers use the core install.
Open-Meteo historical fetching requires `breos[weather]`.

The NSRDB helper and legacy downsampling/CSV converters are scheduled for
removal in 0.6.0. See [Deprecations for 0.6.0](../deprecations.md).

## Horizon provenance

Weather returned by BREOS records a `horizon` object in
`DataFrame.attrs["breos_weather_metadata"]`. Its `status` is one of:

- `applied`: the provider already applied a terrain-horizon profile;
- `not_applied`: the provider explicitly returned unshaded irradiance;
- `unknown`: BREOS cannot establish whether terrain shading is present.

PVGIS TMY requests apply the provider's default horizon by default. Pass
`use_horizon=False` to request unshaded PVGIS irradiance. This switch is
useful when a separate shading model will be applied later; it does not itself
apply a user-defined profile.

When a weather fetcher saves a CSV, BREOS also writes a versioned sidecar named
`<weather-file>.csv.metadata.json`. The sidecar includes the CSV's SHA-256
digest, so provenance is only restored when it still matches the weather data.
Missing, malformed, legacy, or stale sidecars are safe to load, but their
horizon status is `unknown`. BREOS never infers horizon treatment from a
filename such as `pvgis-sarah3`.

## Loading from local files

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.weather.load_weather
   breos.weather.parse_weather_filename
   breos.weather.read_epw_file
```

## Fetching from external APIs

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.weather.fetch_tmy_weather_data
   breos.weather.fetch_tmy_nsrdb
   breos.weather.fetch_weather_data
```

## Resampling

Convert between hourly and 15-minute resolutions. The 15-minute path uses
Makima interpolation on clearness indices rather than raw irradiance so
sunrise / sunset transitions stay physically consistent.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.weather.resample_to_15min
   breos.weather.resample_to_hourly
   breos.weather.resample_tmy_to_15min
```

## Helpers

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.weather.extract_ambient_temperature
   breos.weather.preload_weather_by_year
```
