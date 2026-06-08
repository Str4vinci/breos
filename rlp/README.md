# External Load Profiles

BREOS ships its default H0 reference load profiles **inside the package**
(`breos/data/rlp/`), so you do not need any files here to run a basic simulation.

This directory documents how to supply your **own** load profiles — E-REDES, REE,
BDEW, or custom CSVs — which BREOS does not redistribute. Point BREOS at a local
directory of your licensed files via `rlp_directory` (Python API) or
`--rlp-directory` (CLI). You are responsible for ensuring your source's terms
permit your use.

Expected external filenames:

| Profile key | Filename(s) |
|---|---|
| `4`, `5`, `6` | `EREDES_2025_BTN_1000kwh_hourly.csv` and/or `EREDES_2025_BTN_1000kwh_15min.csv` |
| `7` | `bdew_h0_2025_15min.csv` |
| `8` | `REE_2026_2.0TD_1000kwh_hourly.csv` and/or `REE_2026_2.0TD_1000kwh_15min.csv` |

Example:

```bash
breos run --config configs/examples/external-rlp.toml --rlp-directory external_rlp
```
