# Load profiles

Bundled demandlib-derived H0 examples plus utilities for scaling and time
alignment. BREOS also supports user-supplied BDEW, E-REDES, REE, and custom
CSV files through the `rlp_directory` argument.

For public examples, use `profile_type="demandlib_h0"`, the bundled profile.
Other profile keys are treated as external data and require local files that
users are licensed to use. See
[Load Profile Data](../legal/load-profile-data.md).

The bundled H0 files were generated for 2023. For another study year, BREOS
matches each target day to the nearest source day of the same H0 type:
weekday, Saturday, or Sunday. It searches across the year boundary and uses
28 February as the seasonal anchor for 29 February, then selects a source day
of the leap day's actual type. A selected shape can be up to four calendar
days from the target's month and day. On up to six days a year, that
source day falls in the neighbouring demandlib season: for example, 21 and
22 March 2026 get winter shapes, and 14 May 2025 gets a summer shape. BREOS
scales the resulting profile to the requested annual consumption after
alignment. The default 2023 study year retains its original rows. A
`demandlib_h0` file supplied through `rlp_directory` uses the year on its
dated first row by the same rule. That row must be 1 January 00:00; an
undated `demandlib_h0` file raises `ValueError`. Dated E-REDES files follow
the same rule with E-REDES day classes; see [E-REDES profiles](#e-redes-profiles).
Other external profile families, undated files and `custom` profiles follow
their positional calendar rule. Project years replay the study year's
calendar; they do not advance the load and tariff weekdays each year. Monte
Carlo builds the load for its `target_year` by the same rule.

## External profile files

Non-bundled standard profiles are still supported by the public API. Put the
licensed CSVs in a local directory and pass `rlp_directory`:

```python
from breos.load_profiles import load_profile

load = load_profile(
    "eredes_btn_c",
    annual_consumption_kwh=4000,
    freq="15min",
    rlp_directory="external_rlp",
)
```

For full `breos.App` or CLI runs, use the same directory through config:

```toml
load_profile = "eredes_btn_c"
rlp_directory = "external_rlp"
resolution = "15min"
```

Expected filenames are documented in [Load Profile Data](../legal/load-profile-data.md).
Your own CSV loads with `load_profile("custom", ..., profile_file="meter.csv",
profile_unit="kW")`.

## E-REDES profiles

E-REDES BTN profiles distinguish three day classes: working day, Saturday, and
Sunday or holiday. When an E-REDES file has timestamps, BREOS takes its source
year from them, not from the filename. Each day of the study year then takes
the nearest source day of the same class. The search starts at the same month
and day in the source year and tries one day earlier, then one day later, and
so on. It wraps across New Year, and 29 February starts from 28 February in a
common source year. This is the demandlib H0 rule with E-REDES classes. A file
in its own year loads unchanged. Alignment works on whole civil days before
the timezone is applied, and annual scaling comes after it.

BREOS types a day as Sunday/holiday if it is a Sunday or one of Portugal's
nationwide statutory holidays that year. This applies in both the source year
and the study year. The holidays are those of Labour Code Article 234: 1 January, Good
Friday, Easter Sunday, 25 April, 1 May, Corpus Christi, 10 June, 15 August,
5 October, 1 November, and 1, 8 and 25 December. Corpus Christi, 5 October,
1 November and 1 December were suspended in 2013, 2014 and 2015, so they are
working days (or Saturdays) in those years. A holiday on a Saturday takes the
Sunday/holiday class. Carnival Tuesday, municipal holidays, and bridge days
are not holidays for this rule. E-REDES does not publish a per-date holiday
flag, so this is BREOS's reading of its three day classes. Tariff schedules
keep their own holiday lists.

A dated E-REDES file must start at 1 January 00:00, the start of its first
interval, and hold exactly that calendar year. An undated E-REDES file is
placed by position.

To make the files from the E-REDES publication
(`Perfil_Consumo_Injecao_E-REDES_<year>.csv`), use the converter in the
BREOS source tree:

```bash
python tools/convert_eredes_profiles.py Perfil_Consumo_Injecao_E-REDES_2026.csv --output-dir external_rlp
```

It writes `EREDES_<year>_BTN_1000kwh_15min.csv` and
`EREDES_<year>_BTN_1000kwh_hourly.csv` with the columns
`DateTime,BTN A - Wh,BTN B - Wh,BTN C - Wh`. The year comes from the
publication's dates. The converter does the following:

- It reads the Latin-1 file and its four header rows, finds the BTN A, B and
  C columns by their labels, and drops blank rows.
- It checks that the dates cover one complete calendar year, that each
  weekday matches its date, that every value is finite and non-negative, and
  that there is no missing or repeated quarter-hour.
- It changes each interval end to an interval start. The publication stamps
  each quarter-hour at its end, in Portuguese legal time (`00:15` to `24:00`).
  `24:00` is the next midnight, and each end moves back 15 minutes.
- It puts every date on 96 civil quarter-hours. The fall-back hour is listed
  twice in the publication, with its second occurrence marked `a`; the
  converter keeps the second, standard-time occurrence. The spring-forward
  hour is not in the publication; the converter interpolates it linearly.
- It multiplies kWh by 1000 to give Wh. Each hourly value is the sum of its
  four quarter-hours, so the two files have the same days in the same phase.

The clock changes move each file's annual energy by a few Wh in 1,000 kWh;
BREOS scales every profile to `annual_consumption_kwh` anyway. App
provenance records the converted file and its SHA-256, not the publication
it came from. The converter does not download data, and BREOS does not
bundle E-REDES files.

## Profile registry

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.load_profiles.ProfileSpec
   breos.load_profiles.resolve_profile_key
   breos.load_profiles.resolve_profile_file
```

## Loading and scaling

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.load_profiles.load_profile
   breos.load_profiles.scale_to_annual_consumption
```

## Repairing measured input

An explicit, reported repair step for measured load and PV power series, run
before a simulation. See [Repairing measured data](../getting-started/inputs.md#repairing-measured-data).
The same names are available from `breos.io`.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.repair.repair_series
   breos.repair.InputRepairReport
   breos.repair.RepairEvent
```
