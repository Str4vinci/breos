# PCoE PV system test-bed validation

Result: the PCoE file is useful for a BREOS thermal check and for testing the consistency of its recorded electrical channels. The default BREOS Faiman model has a positive temperature bias against this reference. A PVsyst freestanding sensitivity run performs better, but the mounting type is not documented.

At `Gpoa >= 200 W/m²`, the default BREOS Faiman model produced a bias of `+2.694 °C`, mean absolute error of `3.161 °C`, root mean square error of `3.821 °C`, and Pearson correlation of `0.972` over `12,708` observations. The PVsyst freestanding sensitivity run produced `+0.976 °C` bias, `1.986 °C` MAE, `2.619 °C` RMSE, and `0.976` correlation.

## Scope

The validation uses the single downloaded PCoE CSV. The [Zenodo record](https://zenodo.org/records/15779578) describes it as photovoltaic system and weather data from a test-bed at the Smart Energy Infrastructure PHAETHON CoE, University of Cyprus.

The file contains 15-minute records for one year, from June 1, 2015 through May 31, 2016. The thermal reference is `Tmod`. The thermal inputs are `Gpoa`, `Tamb`, and `WS`. The file also contains `Vdc`, `Idc`, `Pdc`, and `Pac`, so the run checks the identity `Pdc ≈ Vdc × Idc` and describes the recorded `Pac/Pdc` relationship.

This run does not claim a full PV power or inverter validation. The downloaded directory contains no module nameplate, inverter rating, site location, timezone, or separate data dictionary.

## Method

1. Read `pcoe_pv_system_testbed.csv` and parse `time_stamp` with the documented day-first format in the local dataset mapping.
2. Convert all measurement columns to numeric values.
3. Run the BREOS Faiman model with its default parameters, `u0=25.00` and `u1=6.84`.
4. Run BREOS's `pvsyst-freestanding` model as a mounting sensitivity. The model uses BREOS's default module efficiency of `0.20` because the file has no module efficiency field.
5. Compare each temperature prediction with `Tmod` at `Gpoa` thresholds of `0`, `50`, `200`, and `400 W/m²`.
6. Repeat the comparison for stable 15-minute periods where the absolute `Gpoa` change is at most `25 W/m²`.
7. Compare the equilibrium prediction from the previous 15-minute inputs with the current `Tmod`.
8. Group temperature errors by month at the primary threshold of `Gpoa >= 200 W/m²`.
9. Compare `Pdc` with `Vdc × Idc`. Describe `Pac/Pdc` only when `Pdc > 100 W`, because the file has a measured nighttime AC baseline and no inverter nameplate.

## Thermal results

| Gpoa threshold | Model | Observations | Bias (°C) | MAE (°C) | RMSE (°C) | r |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | BREOS Faiman default | 35,136 | 1.577 | 1.873 | 2.552 | 0.992 |
| 0 | BREOS PVsyst freestanding | 35,136 | 0.913 | 1.425 | 1.898 | 0.994 |
| 50 | BREOS Faiman default | 15,659 | 2.308 | 2.842 | 3.543 | 0.980 |
| 50 | BREOS PVsyst freestanding | 15,659 | 0.831 | 1.843 | 2.460 | 0.984 |
| 200 | BREOS Faiman default | 12,708 | 2.694 | 3.161 | 3.821 | 0.972 |
| 200 | BREOS PVsyst freestanding | 12,708 | 0.976 | 1.986 | 2.619 | 0.976 |
| 400 | BREOS Faiman default | 9,941 | 3.062 | 3.407 | 4.040 | 0.965 |
| 400 | BREOS PVsyst freestanding | 9,941 | 1.126 | 2.055 | 2.723 | 0.967 |

At the primary threshold, the stable-period results are:

| Analysis | Model | Observations | Bias (°C) | MAE (°C) | RMSE (°C) | r |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Stable current inputs | BREOS Faiman default | 2,904 | 3.396 | 3.559 | 4.281 | 0.974 |
| Stable current inputs | BREOS PVsyst freestanding | 2,904 | 1.092 | 1.976 | 2.668 | 0.974 |
| Equilibrium inputs from previous 15 minutes | BREOS Faiman default | 12,708 | 2.361 | 2.866 | 3.599 | 0.976 |
| Equilibrium inputs from previous 15 minutes | BREOS PVsyst freestanding | 12,708 | 0.618 | 2.277 | 2.965 | 0.968 |

The stable-period filter does not reduce the Faiman error at this threshold. The PCoE reference behaves differently from the Sandia reference, and the positive Faiman bias persists even when irradiance changes slowly. The PVsyst freestanding model reduces the bias, but the result is conditional on an unverified mounting assumption.

Monthly bias at `Gpoa >= 200 W/m²` is:

| Month | Observations | Faiman bias (°C) | PVsyst freestanding bias (°C) |
| --- | ---: | ---: | ---: |
| January | 752 | 2.337 | 0.565 |
| February | 883 | 2.191 | 0.774 |
| March | 1,112 | 2.294 | 2.210 |
| April | 1,146 | 2.635 | 0.896 |
| May | 1,201 | 2.211 | 1.207 |
| June | 1,228 | 2.416 | 0.819 |
| July | 1,306 | 2.675 | 0.972 |
| August | 1,246 | 3.109 | 1.219 |
| September | 1,120 | 2.901 | 0.877 |
| October | 954 | 2.980 | 0.292 |
| November | 927 | 3.486 | 1.121 |
| December | 833 | 3.190 | 0.314 |

## Electrical channel checks

The recorded `Pdc` is internally consistent with `Vdc × Idc`:

| Population | Observations | Bias of `Vdc × Idc - Pdc` (W) | MAE (W) | RMSE (W) | r | Errors above 10 W |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All rows | 35,136 | 0.184 | 0.305 | 2.011 | 1.000 | 35 |
| `Pdc > 100 W` | 14,469 | 0.370 | 0.640 | 3.103 | 1.000 | 35 |

For `Pdc > 100 W`, the median `Pac/Pdc` ratio is `0.919`, with a fifth to ninety-fifth percentile range of `0.838` to `0.927`. Two rows have `Pac > Pdc`. When `Gpoa = 0`, the median `Pac` is `4.744 W`, and `555` rows still have non-zero `Pdc`. Treat the raw AC channel as a measured signal with a baseline until its meter boundary is confirmed.

## Interpretation

This dataset exposes a clear difference between the two thermal presets. BREOS Faiman tracks the annual variation well, but its temperature prediction is consistently high relative to `Tmod` in the daytime subset. The PVsyst freestanding preset gives a smaller error under the same inputs. That result supports testing the preset further, not changing the default, because the file does not document the mounting construction or the sensor location.

The electrical channels are suitable for a data-quality check. They are not enough to run the BREOS inverter curve independently. The file has no inverter AC rating, so the validation does not fit or assume one.

## Limits

- `Tmod` is a module-temperature measurement. Its sensor location is not documented, while BREOS thermal helpers return cell temperature.
- Measured `Gpoa` is supplied directly. This check does not test solar position, transposition, or GHI/DHI decomposition.
- The file has solar angles and extraterrestrial irradiance, but no verified site coordinates or timezone. No geometry was invented.
- No module parameters support a defensible CEC single-diode comparison.
- No inverter rating supports an independent PVWatts inverter comparison. `Pac/Pdc` is descriptive only.
- All `35,136` rows have finite values and unique timestamps at exact 15-minute spacing. The local file has no separate data dictionary, so units remain based on the local column mapping and observed ranges.

## Provenance

- Source record: [Zenodo record 15779578](https://zenodo.org/records/15779578)
- Source DOI: `10.5281/zenodo.15779578`
- Local source file: `$BREOS_VALIDATION_DATA/pcoe_pv_testbed/pcoe_pv_system_testbed.csv`
- Local MD5: `35386ee96c1a57d10f8d73dfacee858b`, matching the Zenodo record
- BREOS worktree: `/tmp/breos-article1-0.6.0`
- BREOS commit: `f62f4f5bf3c14140ab189d35ea2885e6fcc60c6b`
- BREOS worktree status at run time: clean
- Python version: `3.13.14`
- NumPy version: `2.4.6`
- pandas version: `3.0.5`
- pvlib version: `0.15.2`
- BREOS package metadata version: `0.6.0`
- Input hashes: [`input_manifest.sha256`](input_manifest.sha256)
- Configuration and configuration hash: [`run_config.json`](run_config.json), `ded938c9620edfada9da8ae40b015a108d642b9c87d818ccd1be6ce0e04e491f`
- Driver hash and output hashes: [`provenance.json`](provenance.json)

## Reproduce

From the clean article worktree, run:

```text
BREOS_VALIDATION_ROOT=/path/to/article-worktree \
  BREOS_VALIDATION_DATA=/path/to/datasets \
  python tools/validation/recovery/pcoe/drivers/pcoe_validate.py --force
```

The driver refuses to replace existing generated files unless you pass `--force`. It writes [`thermal_metrics.csv`](thermal_metrics.csv), [`thermal_sensitivity_metrics.csv`](thermal_sensitivity_metrics.csv), [`monthly_bias.csv`](monthly_bias.csv), [`electrical_checks.json`](electrical_checks.json), [`dataset_facts.json`](dataset_facts.json), and [`provenance.json`](provenance.json).
