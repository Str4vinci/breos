# Run offline with cached weather

When the config uses a location *preset key*, BREOS scans a `weather/`
directory in the current working directory before fetching from PVGIS, and
silently reuses a file named `<location>_tmy_<year0>_<year1>_<source>.csv`, or
the same name gzip-compressed as `.csv.gz`. Seed the cache once while online:

```python
from pathlib import Path
from breos.weather import fetch_tmy_weather_data

Path("weather").mkdir(exist_ok=True)
tmy, _ = fetch_tmy_weather_data(
    latitude=41.1579,
    longitude=-8.6291,
    timezone="Europe/Lisbon",
)
tmy.to_csv("weather/porto_tmy_2005_2023_pvgis-sarah3.csv")
```

This manual CSV export is compatible with existing offline runs, but it has no
provenance sidecar and is therefore loaded with an `unknown` horizon status.
Calling `fetch_tmy_weather_data(..., save_to_file=True)` writes both the CSV
and its digest-bound `.csv.metadata.json` sidecar. Keep the two files together
and rename both with the same CSV basename if you adapt the generated filename
to a location preset.

Subsequent runs from the same working directory work without network access
(the log line `Found local weather file` confirms the cache hit). Custom
coordinate-dict locations always fetch; delete or rename the file to force a
fresh fetch. The filename's year part only needs to match the pattern; it is
metadata, not a lookup key.

If the directory holds more than one TMY file for the location, for example a
PVGIS and an NSRDB export for Porto, BREOS does not pick one: the run stops and
lists the candidates. Set `weather_source` to the filename's source part to
choose one:

```python
App({"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000,
     "weather_source": "pvgis-sarah3"})
```

or `breos run --config config.toml --weather-source pvgis-sarah3` from the
command line. A `weather_source` with no matching file is an error rather than
a PVGIS fetch. The file that was used, with its SHA-256 digest and the parsed
filename (including the source), is recorded under `provenance.weather`.

The file is restamped onto the year of `start_date` and must then cover that
whole calendar year. A file missing its first or last rows raises `ValueError`
that names the file and the missing span, rather than simulating a shorter
year. With a [`[period]`](partial-year.md), the file needs to cover
only the window.

The repository's `validation/data/weather/` directory holds PVGIS TMYs for a
few preset locations as `.csv.gz` files, with the Porto file's sidecar. The
[case examples](../gallery/index.rst) ran on them: copy one, with its sidecar
if it has one, into `weather/` to reproduce a case offline.
