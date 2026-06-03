# Weather

Sources, loaders, and resampling utilities for solar irradiance and
temperature time series.

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
