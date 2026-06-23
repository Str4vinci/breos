# Load profiles

Bundled demandlib-derived H0 examples plus utilities for scaling and time
alignment. BREOS also supports user-supplied BDEW, E-REDES, REE, and custom
CSV files through the `rlp_directory` argument.

For public examples, use `profile_type="demandlib_h0"` or the equivalent
canonical key `"1"`. Other profile keys are treated as external data and
require local files that users are licensed to use. See
[Load Profile Data](../legal/load-profile-data.md).

## External profile files

Non-bundled standard profiles are still supported by the public API. Put the
licensed CSVs in a local directory and pass `rlp_directory`:

```python
from breos.load_profiles import load_profile

load = load_profile(
    "6",
    annual_consumption_kwh=4000,
    freq="15min",
    rlp_directory="external_rlp",
)
```

For full `breos.App` or CLI runs, use the same directory through config:

```toml
load_profile = "6"
rlp_directory = "external_rlp"
resolution = "15min"
```

Expected filenames are documented in [Load Profile Data](../legal/load-profile-data.md).

## Loading and scaling

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.load_profiles.load_profile
   breos.load_profiles.scale_to_annual_consumption
```

## Alignment

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.load_profiles.align_load_to_pv
```
