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
useful when a separate shading model will be applied; it does not itself apply
a user-defined profile.

At the App level, configure an inline circular horizon as azimuth/elevation
pairs:

```toml
horizon_profile = [
  [0, 4],
  [90, 12],
  [180, 3],
  [270, 7],
]
```

Azimuth is clockwise from north and both values are degrees. BREOS linearly
interpolates across north, including between the last and first point. It
zeros DNI while the apparent sun elevation is on or below that line and
removes the corresponding direct-horizontal component from GHI. DHI is kept:
the v1 profile models far-horizon beam obstruction, not diffuse sky-view loss.

When this key is active and no cached weather matches, the App automatically
requests fresh PVGIS data with `use_horizon=False`. It refuses weather marked
`applied` (double-counting) or `unknown` (unsafe to decide). A local CSV is
therefore eligible only when its valid metadata sidecar records
`horizon.status = "not_applied"`.

Monte Carlo weather files do not yet carry equivalent per-file horizon
provenance, so `horizon_profile` is rejected by that workflow rather than
being applied to weather whose prior treatment is unknown.

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
sunrise / sunset transitions stay physically consistent. Set
`preserve_irradiance_energy=True` to renormalize each source hour's four GHI,
DNI, and DHI values to the original hourly mean. This opt-in mode is useful
when the source values represent hourly averages; the default keeps the
established interpolation output.

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
