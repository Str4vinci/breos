# Load Profile Data

BREOS can read several residential load-profile formats, but the public package only bundles profiles whose redistribution posture is appropriate for a general open-source release.

This is a project policy note, not legal advice.

## Bundled with BREOS

- `h0SLP_demandlib_1000kwh_hourly.csv`
- `h0SLP_demandlib_1000kwh_15min.csv`

These profiles were generated with demandlib, which documents itself as MIT-licensed free software. They provide a safe default for examples, tests, and installed package behavior.

## Why some profiles are not bundled

BREOS still supports E-REDES BTN, REE 2.0TD, direct BDEW H0 exports, and custom CSVs when the caller passes `rlp_directory` to {py:func}`breos.load_profiles.load_profile`.

Those files are not bundled in the public package because the source terms reviewed for this release do not provide a clear redistribution grant for package publication:

- BDEW publishes Standardlastprofile downloads, but its site terms reserve copyright rights and limit downloads/copies to private, non-commercial use unless permission is granted.
- E-REDES terms reviewed through the public website are restrictive enough that redistribution in an OSS package should be treated as unconfirmed.
- REE publishes legal terms that reserve intellectual-property rights and do not clearly authorize republishing derived CSV datasets.

That does not mean users cannot use those profiles. It means BREOS does not redistribute the CSV files. Users can still download or obtain the profiles themselves, keep them outside the package, and point BREOS at that local directory if their source terms permit their use case.

## Using external profiles

Create a local directory that is not committed to this repository, for example:

```text
external_rlp/
  EREDES_2025_BTN_1000kwh_hourly.csv
  EREDES_2025_BTN_1000kwh_15min.csv
```

Then reference it from Python:

```python
from breos.load_profiles import load_profile

load = load_profile(
    "6",
    annual_consumption_kwh=4000,
    freq="15min",
    rlp_directory="external_rlp",
)
```

Or from the BREOS app/CLI config:

```toml
location = "porto"
n_modules = 10
annual_consumption_kwh = 4000
load_profile = "6"
rlp_directory = "external_rlp"
resolution = "15min"
```

```bash
breos run --config configs/examples/external-rlp.toml
```

## Expected external filenames

BREOS selects files by profile key and requested resolution:

| Profile key | Profile family | Hourly filename | 15-minute filename | Notes |
|---|---|---|---|---|
| `"4"` | E-REDES BTN A | `EREDES_2025_BTN_1000kwh_hourly.csv` | `EREDES_2025_BTN_1000kwh_15min.csv` | File must contain `BTN A - Wh`. |
| `"5"` | E-REDES BTN B | `EREDES_2025_BTN_1000kwh_hourly.csv` | `EREDES_2025_BTN_1000kwh_15min.csv` | File must contain `BTN B - Wh`. |
| `"6"` / `"eredes_btn_c"` | E-REDES BTN C | `EREDES_2025_BTN_1000kwh_hourly.csv` | `EREDES_2025_BTN_1000kwh_15min.csv` | File must contain `BTN C - Wh`. |
| `"7"` | BDEW H0 2025 | `bdew_h0_2025_15min.csv` | `bdew_h0_2025_15min.csv` | Native 15-minute file; BREOS can downsample to hourly. |
| `"8"` / `"ree_2.0td"` | REE 2.0TD | `REE_2026_2.0TD_1000kwh_hourly.csv` | `REE_2026_2.0TD_1000kwh_15min.csv` | Generic single-column CSV format. |

For E-REDES, the same CSV can contain BTN A, BTN B, and BTN C columns; BREOS chooses the column based on the profile key.

## If redistribution permission is granted

If an operator explicitly allows redistribution, keep a copy of the permission or license text with the release record, update `ATTRIBUTIONS.md`, add the files back under `breos/data/rlp/`, and add a test that the installed wheel can load the profile without `rlp_directory`.

## Practical rule

Use `load_profile = "1"` for public examples and packaged defaults. Use other profile keys only in private or downstream projects after obtaining the source files under terms that permit the intended use.
