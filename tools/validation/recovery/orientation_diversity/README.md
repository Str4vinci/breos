# Orientation-diversity PV data validation

Result: the workbook is internally consistent and contains useful orientation signal, but it cannot support a defensible BREOS weather-to-PV run from the downloaded files alone.

The workbook has `7,080` rows covering `295` numbered days and `24` hours per day. The `Total PV Power` column equals the sum of the five panel-power columns in every row. A held-out orientation-response screen reduced aggregate RMSE from `159.126 W` to `131.821 W` and raised R² from `0.830` to `0.883` when the supplied incidence-angle cosine values were added to common solar radiation.

## Source and scope

The [Zenodo record for the dataset](https://zenodo.org/records/19245713) describes hourly PV generation and weather data from five rooftop panels with different orientations. It lists the record as CC BY 4.0 and reports `7,080` hours of data.

The local directory contains `PV_Data.xlsx` and `README.txt`. The workbook has one data sheet with no header row and one description sheet. The driver takes the header from row 2 of the description sheet and the values from the data sheet.

The workbook provides day and hour numbers, common weather fields, solar radiation, five incidence-angle cosine columns, five panel-power columns, and total PV power. It does not provide calendar dates, a timezone, site coordinates, panel tilt or azimuth, GHI, DNI, DHI, module parameters, or inverter parameters.

## BREOS fit

The current BREOS production path was not run. That path needs a `DatetimeIndex`, a location, panel geometry, irradiance components, and module parameters. The local workbook does not provide those inputs as verified values.

The run instead performs two checks:

1. It checks the data structure and the exact panel-to-total power identity.
2. It measures whether the supplied incidence-angle cosine values improve a held-out power screen.

The second check is not a BREOS model. It uses ordinary least squares through the origin on days `1` through `206`, then evaluates days `207` through `295` where solar radiation exceeds `20 W/m²`.

The common-radiation baseline uses:

```text
power = coefficient × Solar Radiation
```

The orientation model uses:

```text
power = coefficient_1 × Solar Radiation
       + coefficient_2 × Solar Radiation × max(AOI cosine, 0)
```

The screen uses the same weather row for all five panels. It does not infer a panel tilt, azimuth, location, or module rating.

## Held-out orientation screen

The test set contains `984` daylight rows. The metrics compare measured panel power with the fitted screen predictions.

| Panel | Model | Bias (W) | MAE (W) | RMSE (W) | r | R² |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PV-1 | Common radiation | -2.700 | 20.520 | 26.044 | 0.939 | 0.878 |
| PV-1 | Radiation plus AOI cosine | -3.505 | 21.249 | 26.562 | 0.941 | 0.873 |
| PV-2 | Common radiation | -8.982 | 36.541 | 46.082 | 0.810 | 0.643 |
| PV-2 | Radiation plus AOI cosine | -2.475 | 23.319 | 30.918 | 0.938 | 0.839 |
| PV-3 | Common radiation | -4.928 | 34.946 | 43.701 | 0.861 | 0.737 |
| PV-3 | Radiation plus AOI cosine | -2.833 | 25.338 | 33.134 | 0.934 | 0.849 |
| PV-4 | Common radiation | -1.879 | 28.140 | 35.223 | 0.903 | 0.814 |
| PV-4 | Radiation plus AOI cosine | -3.136 | 24.006 | 30.705 | 0.933 | 0.859 |
| PV-5 | Common radiation | 2.420 | 19.280 | 25.508 | 0.944 | 0.889 |
| PV-5 | Radiation plus AOI cosine | 1.260 | 18.924 | 24.816 | 0.947 | 0.895 |
| Total | Common radiation | -16.068 | 126.314 | 159.126 | 0.912 | 0.830 |
| Total | Radiation plus AOI cosine | -10.690 | 101.591 | 131.821 | 0.948 | 0.883 |

The AOI feature helps panels 2 through 5 in this split. It slightly worsens PV-1 RMSE and R². The aggregate improvement shows that the workbook contains orientation-related information, but it does not validate BREOS transposition or electrical output.

## Data checks

- All `7,080` numeric rows have finite numeric values.
- The data contains `295` days with exactly `24` rows per day.
- No `Day` and `Hour` pair is duplicated.
- `Total PV Power` matches the sum of `PV-1 Power` through `PV-5 Power` exactly. The maximum absolute difference is `0 W`.
- Each panel has one positive-power row where `Solar Radiation` is zero. The shared row is day `159`, hour `19`, with total power `84 W`. Treat that row as an anomaly until the source definition is checked.
- The local column called `Dewpoint` ranges from `7` to `94`, while `Air Temperature` ranges from `-14` to `36`. Do not use `Dewpoint` as a temperature input without confirming its meaning.

## Limits and next action

This dataset is useful for testing an orientation-aware data adapter. It is not ready for an independent BREOS production comparison. Recover the source location, timezone, panel tilt and azimuth, module nameplate, and irradiance definitions before running `calculate_multi_array_production`.

The source values also need a decision about whether panel power is W or Wh. The Zenodo description allows either interpretation. The current screen preserves the source values and does not convert them.

## Provenance

- Source record: [Zenodo record 19245713](https://zenodo.org/records/19245713)
- Source DOI: `10.5281/zenodo.19245713`
- Local source directory: `$BREOS_VALIDATION_DATA/orientation_diversity_pv`
- Input hashes: [`input_manifest.sha256`](input_manifest.sha256)
- BREOS worktree: `/tmp/breos-article1-0.6.0`
- BREOS commit: `f62f4f5bf3c14140ab189d35ea2885e6fcc60c6b`
- BREOS worktree status at run time: clean
- Python version: `3.13.14`
- NumPy version: `2.4.6`
- pandas version: `3.0.5`
- BREOS package metadata version: `0.6.0`
- Configuration and configuration hash: [`run_config.json`](run_config.json), `bf059326c7055da48a87706a7b2674ba9778766f8282bc58a87e89a20d83d16b`
- Driver hash and output hashes: [`provenance.json`](provenance.json)

## Reproduce

From the clean article worktree, run:

```text
BREOS_VALIDATION_ROOT=/path/to/article-worktree \
  BREOS_VALIDATION_DATA=/path/to/datasets \
  python tools/validation/recovery/orientation_diversity/drivers/orientation_screen.py --force
```

The driver refuses to replace existing generated files unless you pass `--force`. It writes [`orientation_screen_metrics.csv`](orientation_screen_metrics.csv), [`data_integrity.json`](data_integrity.json), [`dataset_facts.json`](dataset_facts.json), and [`provenance.json`](provenance.json).
