# Sandia/IEA PVPS Task 13 thermal validation

Result: the default BREOS Faiman model follows the measured back-of-module temperature well enough to support further work with this dataset. This run is a thermal component check, not a full weather-to-PV validation.

At `Gpoa >= 200 W/m²`, the default BREOS path produced a mean bias of `-0.038 °C`, mean absolute error of `2.141 °C`, root mean square error of `2.993 °C`, and Pearson correlation of `0.970` over `26,023` observations.

## Scope

The downloaded collection contains twelve monthly five-minute CSV files and a module-characterization workbook. The official dataset description also lists Pmax, Isc, Voc, a full power matrix, spectral response, IAM, and outdoor electrical measurements. This run uses only the thermal slice. The reference is `Tbom`, the measured back-of-module temperature. BREOS is run through its Faiman temperature helper with the measured `Gpoa`, `AIR_TEMP`, and `WIND_SPEED` inputs.

The dataset is the [IEA PVPS Task 13 module validation dataset](https://pvpmc.sandia.gov/datasets/iea-pvps-task-13-module-validation-dataset/). The source page describes one year of five-minute outdoor operating data with module electrical measurements, irradiance, ambient conditions, and wind speed.

## Method

1. Read all twelve semicolon-delimited CSV files in filename order.
2. Strip whitespace from the CSV headers. The files contain leading spaces after delimiters.
3. Parse the day-first `Date` and `Time` columns into a timestamp index and sort the combined data.
4. Convert the four required input columns to numeric values.
5. Keep rows with non-null `Gpoa`, `AIR_TEMP`, `WIND_SPEED`, and `Tbom`, with non-negative `Gpoa` and wind speed.
6. Compare the measured `Tbom` with two Faiman runs:
   - `breos_faiman_default`: the current BREOS defaults, `u0=25.00` and `u1=6.84`.
   - `faiman_sandia_u0_u1`: the `U0=29.84` and `U1=3.44` values in the downloaded module-characterization workbook, used as a sensitivity check.
7. Recalculate the default model for stable periods, where the previous valid record is five minutes earlier and the absolute `Gpoa` change is at most `25 W/m²`.
8. Recalculate the default model from the previous five-minute inputs to show the effect of thermal lag.
9. Group the default-model errors by month at the primary `Gpoa` threshold.

The primary threshold is `Gpoa >= 200 W/m²`, which reduces the influence of low-light sensor and thermal offsets. The other thresholds are retained to show sensitivity to that choice.

## Results

| Gpoa threshold | Model | Observations | Bias (°C) | MAE (°C) | RMSE (°C) | r |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | BREOS Faiman default | 34,834 | 0.040 | 1.891 | 2.721 | 0.985 |
| 0 | Workbook U0/U1 sensitivity | 34,834 | -0.797 | 2.063 | 2.790 | 0.986 |
| 50 | BREOS Faiman default | 31,686 | -0.018 | 1.992 | 2.832 | 0.980 |
| 50 | Workbook U0/U1 sensitivity | 31,686 | -0.929 | 2.187 | 2.905 | 0.982 |
| 200 | BREOS Faiman default | 26,023 | -0.038 | 2.141 | 2.993 | 0.970 |
| 200 | Workbook U0/U1 sensitivity | 26,023 | -1.077 | 2.380 | 3.076 | 0.973 |
| 400 | BREOS Faiman default | 21,055 | -0.010 | 2.111 | 2.921 | 0.966 |
| 400 | Workbook U0/U1 sensitivity | 21,055 | -1.133 | 2.397 | 3.023 | 0.970 |

At the primary threshold, the diagnostics are:

| Analysis | Model | Observations | Bias (°C) | MAE (°C) | RMSE (°C) | r |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Stable current inputs | BREOS Faiman default | 17,553 | -0.026 | 1.699 | 2.246 | 0.983 |
| Stable current inputs | Workbook U0/U1 sensitivity | 17,553 | -1.111 | 2.039 | 2.444 | 0.986 |
| Equilibrium inputs from previous five minutes | BREOS Faiman default | 25,727 | -0.168 | 1.845 | 2.484 | 0.979 |

The default RMSE falls from `2.993 °C` to `2.246 °C` on stable periods. Using the equilibrium prediction from the previous five minutes gives `2.484 °C` RMSE. Both checks indicate that irradiance changes and thermal inertia contribute to the annual error.

Monthly default-model bias at `Gpoa >= 200 W/m²` is:

| Month | Observations | Bias (°C) |
| --- | ---: | ---: |
| January | 1,656 | 0.672 |
| February | 2,060 | 0.828 |
| March | 2,499 | 0.841 |
| April | 2,152 | 0.610 |
| May | 2,445 | 0.042 |
| June | 2,334 | -0.538 |
| July | 2,710 | -1.059 |
| August | 3,057 | -1.068 |
| September | 2,813 | -0.826 |
| October | 1,757 | 0.011 |
| November | 735 | 0.978 |
| December | 1,805 | 0.912 |

## Interpretation

The current BREOS Faiman defaults have near-zero annual bias against this reference and preserve the measured time variation with high correlation. The annual bias hides a seasonal pattern. Bias is about `-1.07 °C` in July and August and about `+0.98 °C` in November.

The workbook coefficients produce a cooler prediction and have worse bias, MAE, and RMSE at the primary threshold. They have slightly better correlation, `0.973` versus `0.970`, so calling them unconditionally worse would be inaccurate. They should not replace the current defaults based on this result alone.

## Limits

- `Tbom` is a back-of-module measurement, while the BREOS helper returns cell temperature. The comparison therefore includes a sensor-boundary difference.
- Measured `Gpoa` is supplied directly. Geometry is not needed for this thermal component check. The run does not test GHI/DHI-to-POA transposition, solar position, or irradiance decomposition.
- The official dataset includes electrical reference data, but the downloaded characterization workbook does not provide the Vmp and Imp values needed by the current BREOS `PVModuleParams` interface. A separate electrical-component adapter may still be worthwhile.
- The combined input contains `36,449` rows and no duplicate timestamps. There are `34,834` rows in the thermal comparison after filtering. Most consecutive records are five minutes apart, but the files contain long gaps, so this is not treated as a complete regular time series.

## Provenance

- BREOS worktree: `/tmp/breos-article1-0.6.0`
- BREOS commit: `f62f4f5bf3c14140ab189d35ea2885e6fcc60c6b`
- BREOS worktree status at run time: clean
- pvlib version: `0.15.2`
- Python version: `3.13.14`
- NumPy version: `2.4.6`
- pandas version: `3.0.5`
- BREOS package metadata version: `0.6.0`
- Source directory: `$BREOS_VALIDATION_DATA/sandia_iea_pvps_task13`
- Input hashes: [`input_manifest.sha256`](input_manifest.sha256)
- Configuration and configuration hash: [`run_config.json`](run_config.json), `4c872b61fb2f6923b1d4027a2d26c806e29be3f4910a24003d5a9bebb95394a4`
- Driver hash and output hashes: [`provenance.json`](provenance.json)

## Reproduce

From the clean article worktree, run:

```text
BREOS_VALIDATION_ROOT=/path/to/article-worktree \
  BREOS_VALIDATION_DATA=/path/to/datasets \
  python tools/validation/recovery/sandia_task13/drivers/sandia_thermal_validate.py --force
```

The driver refuses to replace existing generated files unless you pass `--force`. It writes [`thermal_metrics.csv`](thermal_metrics.csv), [`thermal_sensitivity_metrics.csv`](thermal_sensitivity_metrics.csv), [`monthly_bias.csv`](monthly_bias.csv), [`dataset_facts.json`](dataset_facts.json), and [`provenance.json`](provenance.json).
