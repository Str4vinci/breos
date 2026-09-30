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
undated `demandlib_h0` file raises `ValueError`. Other external profile
families still follow their positional calendar rule. Project years replay
the study year's calendar; they do not advance the load and tariff weekdays
each year. Monte Carlo builds the load for its `target_year` by the same
rule.

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
