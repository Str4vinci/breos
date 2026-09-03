# Reunion Island microgrid validation

Result: this collection is useful for BREOS component and integration checks, but it does not validate the complete microgrid simulation. The default BREOS Faiman model overpredicts the measured under-panel temperature, and the default lumped battery-temperature helper underpredicts the measured battery temperature. The differences are large enough to require boundary and parameter investigation before either model is used as a validated representation of this installation.

## Source and scope

The collection is the [data in an experimental stand-alone microgrid](https://pmc.ncbi.nlm.nih.gov/articles/PMC10568554/) from Roche Plate, Mafate, Reunion Island. The corresponding [Zenodo dataset DOI is 10.5281/zenodo.8186303](https://doi.org/10.5281/zenodo.8186303). The local README identifies meteo, PV-regulator, inverter, battery, and three-house load files.

This run uses:

- `Gincl`, ambient temperature, wind speed, and `Tpv` for a PV thermal-component check;
- `Ubat`, `Ibat`, ambient temperature, and `Tbat` for the BREOS lumped battery-temperature helper; and
- inverter, house-main, and submeter channels for boundary and data-integrity checks.

It does not claim to validate a complete weather-to-PV-to-battery simulation. The files do not provide a battery SOC trace, nominal battery capacity, PV module electrical parameters, or inverter nameplate data. `Gincl` is already measured plane-of-array irradiance, so solar-position and irradiance-transposition code are not exercised.

## Data handling

The local meteo file contains `1,048,573` data rows and ends at `2020-12-29 18:18:00`, while the plant and load files continue to May 2021. The file has exactly `1,048,576` lines including its metadata and header. This is an incomplete meteo year and is treated as such.

The meteo records are 20 seconds apart through June. During July, three successive readings share each minute timestamp. The runner averages numeric values at exact duplicate timestamps and then averages the meteo data to one-minute bins. This prevents July from receiving three times the weight of the other months and aligns it with the one-minute plant and load files. The raw meteo file has `44,618` duplicate timestamp groups, of which `44,616` contain three rows.

House 2 has `108,257` missing values in its main channel, `Input5`. All exact duplicate timestamps in the plant and load files are averaged before analysis. Long gaps remain visible in the recorded facts and are not interpolated.

## Method

1. Parse the semicolon-delimited meteo file after its two metadata rows. Parse the plant file after its metadata row. Parse house 2 dates day-first and the other house dates in ISO order.
2. Convert source `Gincl` from kW/m² to W/m² by multiplying by `1,000`.
3. Average exact duplicate timestamps, then use one-minute means for the meteo thermal comparison.
4. Run the BREOS Faiman default and a PVsyst freestanding sensitivity with module efficiency `0.20` against measured `Tpv`.
5. Report thresholds at `0`, `50`, `200`, and `400 W/m²`. The primary threshold is `Gincl >= 200 W/m²`.
6. Define stable inputs as consecutive one-minute records with an absolute `Gincl` change no greater than `25 W/m²`. Also evaluate Faiman using inputs from five minutes earlier when those records exist.
7. Aggregate the battery telemetry to hourly means, split terminal power using the source convention that positive current is discharge and negative current is charge, and run the BREOS default lumped thermal helper with its default efficiencies and thermal resistance.
8. Compare each house's named component channels with its named main channel. Compare `Pout` with the sum of the three available house main channels only when all three are present.

## PV thermal results

At the primary threshold, model error is prediction minus measured `Tpv`:

| Model | Observations | Bias (°C) | MAE (°C) | RMSE (°C) | r |
| --- | ---: | ---: | ---: | ---: | ---: |
| BREOS Faiman default | 108,986 | 8.882 | 9.978 | 11.162 | 0.623 |
| BREOS PVsyst freestanding sensitivity | 108,986 | 6.651 | 8.066 | 9.009 | 0.708 |

For the default Faiman model, stable current inputs produce bias `9.839 °C`, MAE `10.702 °C`, and RMSE `11.642 °C` over `81,858` observations. Using the equilibrium inputs from five minutes earlier gives bias `8.690 °C`, MAE `9.627 °C`, and RMSE `10.788 °C` over `108,952` observations. Stability and a short lag adjustment do not remove the main discrepancy.

The default Faiman monthly bias at the primary threshold ranges from `7.693 °C` in November to `9.826 °C` in July. This is not an annual cancellation of opposing seasonal errors.

`Tpv` is described as the surface temperature under the roof-mounted PV panels. BREOS returns a cell-temperature model output. The comparison therefore combines a physical-boundary difference with any mismatch in mounting, ventilation, sensor placement, and thermal parameters. The result is evidence for calibration and boundary investigation, not evidence that the default model is accurate for this installation.

Detailed threshold, stability, lag, and monthly results are in [`thermal_metrics.csv`](thermal_metrics.csv), [`thermal_sensitivity_metrics.csv`](thermal_sensitivity_metrics.csv), and [`thermal_monthly.csv`](thermal_monthly.csv).

## Battery thermal results

Using hourly means over the period where measured ambient temperature and battery telemetry overlap, the default BREOS lumped battery-temperature helper produced:

| Analysis | Observations | Bias (°C) | MAE (°C) | RMSE (°C) | r |
| --- | ---: | ---: | ---: | ---: | ---: |
| All common hourly records | 5,655 | -3.292 | 3.870 | 4.399 | 0.640 |
| Absolute battery power at least 100 W | 5,288 | -3.223 | 3.841 | 4.395 | 0.627 |

The model uses the BREOS default charge and discharge efficiencies, `sqrt(0.95)`, and thermal resistance `0.05 K/W`. `Tbat` is a dataset battery-temperature channel, not an independently specified cell-temperature measurement. The helper is documented as quasi-steady and suitable for hourly or longer steps; this check follows that limitation. The result supports further battery thermal calibration but does not validate SOC, dispatch, capacity derating, or degradation.

See [`battery_thermal_metrics.csv`](battery_thermal_metrics.csv).

## Electrical and load checks

`Pout` is the inverter-output channel, while `Ppv1` and `Ppv2` are solar-regulator outputs. The dataset does not establish that these are the same electrical boundary, so the runner reports their ranges and recorded-sample energy separately instead of treating `Pout / (Ppv1 + Ppv2)` as inverter efficiency.

On `342,479` rows with a non-null `Pout` and all three house main channels, `Pout * 1,000` minus the sum of house mains has bias `-38.8 W`, MAE `66.7 W`, RMSE `123.2 W`, and correlation `0.715`. The recorded-sample sums are `1,889.9 kWh` for `Pout`, `4,047.0 kWh` for `Ppv1 + Ppv2`, and `1,894.8 kWh` for the three house mains on their common rows. These are not annual energy yields because the series contain gaps and do not share complete coverage.

The component-to-main load checks are:

| House | Main channel | Valid rows | Bias of component sum minus main (W) | MAE (W) | RMSE (W) | r | Within 5 W |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `Input4` | 510,640 | 30.8 | 39.1 | 84.4 | 0.799 | 36.5% |
| 2 | `Input5` | 403,908 | -16.3 | 41.4 | 65.8 | 0.776 | 27.7% |
| 3 | `Input1` | 502,255 | 4.9 | 6.8 | 12.1 | 0.995 | 67.9% |

The reconciliation is strongest for house 3. It is not exact for any house, and house 2's main channel has a substantial missing interval. These checks are useful for building a BREOS input adapter, not for asserting that the submeter groups exhaust each house's consumption.

Detailed values are in [`electrical_checks.json`](electrical_checks.json) and [`load_checks.csv`](load_checks.csv). Dataset coverage and missing-value facts are in [`dataset_facts.json`](dataset_facts.json).

## Verdict

Worth keeping for BREOS validation, with a component-focused scope:

- Use it to investigate and calibrate PV thermal boundary assumptions. The current Faiman default does not match this under-panel target closely.
- Use it to test the battery thermal helper after specifying the measured-temperature boundary and pack parameters.
- Use the house and plant channels to test timestamp alignment, unit conversion, missing-data handling, and adapter logic.
- Do not use this run as validation of the complete BREOS weather-to-PV-to-battery chain. A follow-up would need the missing SOC/capacity and electrical metadata, plus a complete meteo export.

## Provenance

- BREOS worktree: `/tmp/breos-article1-0.6.0`
- BREOS commit: `f62f4f5bf3c14140ab189d35ea2885e6fcc60c6b`
- BREOS worktree status at run time: clean
- Python: `3.13.14`
- NumPy: `2.4.6`
- pandas: `3.0.5`
- BREOS package metadata version: `0.6.0`
- Source directory: `$BREOS_VALIDATION_DATA/reunion_island_microgrid`
- Input hashes: [`input_manifest.sha256`](input_manifest.sha256)
- Configuration and configuration hash: [`run_config.json`](run_config.json), `9e33ce2a47a021e72090c2c29b64d835c2d8a6b6438351a72c3620ce528c0a31`
- Driver and output hashes: [`provenance.json`](provenance.json)

## Reproduce

From the clean article worktree:

```text
BREOS_VALIDATION_ROOT=/path/to/article-worktree \
  BREOS_VALIDATION_DATA=/path/to/datasets \
  python tools/validation/recovery/reunion_microgrid/drivers/reunion_validate.py --force
```

The driver refuses to replace existing generated files unless `--force` is supplied. It writes the CSV and JSON files listed above and records input, configuration, driver, and output hashes.
