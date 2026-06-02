# Load Profile Files

This directory keeps only redistributable example RLPs in the public repository:

- `h0SLP_demandlib_1000kwh_hourly.csv`
- `h0SLP_demandlib_1000kwh_15min.csv`

BREOS still supports external E-REDES, REE, BDEW, and custom CSV profiles through `rlp_directory`. Users must provide those files from a source they are licensed to use.

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
