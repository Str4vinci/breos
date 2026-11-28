# External validation

BREOS 0.8.0 was checked against five public measured datasets. Each check
isolates one part of the model. None of the datasets supports one unqualified
weather-to-AC accuracy score, so each result is reported at the measurement
boundary that its source data can support.

| Dataset | Component checked | Main result |
|---|---|---|
| NIST Gaithersburg Ground Array | Module, temperature, and inverter calculations from measured plane-of-array irradiance | The inverter calculation differed from the independent AC meter by `-0.06%`, with hourly `r = 0.99998`. |
| DKA Solar Centre, Alice Springs | Fixed-plane irradiance transposition | All tested models were within `1.8%` of annual measured plane-of-array irradiance. Perez reproduced the measured west-to-north energy ratio within `0.60%`. |
| IEA PVPS Task 13 module dataset | Faiman module-temperature calculation | At plane-of-array irradiance of at least `200 W/m2`, bias was `-0.038 C`, RMSE was `2.993 C`, and `r = 0.970` over `26,023` records. |
| UCY PHAETHON PV test-bed | Faiman module-temperature calculation | At plane-of-array irradiance of at least `200 W/m2`, the default model had a bias of `+2.694 C`, RMSE of `3.821 C`, and `r = 0.972` over `12,708` records. |
| Reunion Island microgrid | Faiman temperature and the lumped battery-temperature helper | Diagnostic only. Both defaults disagree with the measured sensors by several degrees, which points to boundary and parameter differences. |

The checks were rerun on the 0.8.0 release candidate at commit
`419254a5effe51d9d037472bc785d135bdeda5b0`. The displayed headlines are
unchanged to their shown precision from the preceding 0.8 development replay.
The DKA Solar Centre orientation-pair comparison was rerun for 0.8.1 with the
missing-irradiance rule described in its section.
Raw third-party data and generated measured-data results are not committed to
this repository. The original drivers, input manifests, and run records are
preserved in the [BREOS 0.6.2 archive](https://doi.org/10.5281/zenodo.22938914)
under `validation/external/` and `tools/validation/recovery/`.

## NIST Gaithersburg Ground Array

The NIST Ground Array is a 271 kW grid-connected system with co-located
irradiance, weather, temperature, DC, and AC measurements. The system has no
battery. NIST describes the array, instruments, channel definitions, and data
quality in Matthew Boyd's 2017 paper, [Performance Data from the NIST
Photovoltaic Arrays and Weather Station](https://doi.org/10.6028/jres.122.040).
NIST publishes the measurements under [data DOI
10.18434/M3S67G](https://doi.org/10.18434/M3S67G) and provides a separate
[data dictionary](https://www.nist.gov/system/files/documents/2017/10/04/datadictionary_supplementalcontent.pdf).

The component check used measured plane-of-array irradiance, ambient
temperature, and wind speed. This removes irradiance decomposition and
transposition from the comparison. Measured DC power was also passed directly
to the BREOS PVWatts inverter calculation and compared with the independent
revenue-grade AC meter.

For the inverter calculation, annual bias was `-0.06%`, hourly normalized RMSE
was `0.91%`, and hourly correlation was `0.99998`. For plane-of-array
irradiance above `400 W/m2`, the default Faiman calculation had a temperature
bias of `+2.19 C`, RMSE of `4.39 C`, and correlation of `0.9635` against the
mean of seven backsheet sensors.

This check does not validate decomposition or transposition. Array outages and
snow-affected days were identified from the independent measurements and were
reported separately instead of being fitted as system losses.

## DKA Solar Centre, Alice Springs

The [DKA Solar Centre](https://dkasolarcentre.com.au/) publishes open five-minute
weather and PV measurements from its Alice Springs demonstration systems. The
[data download page](https://dkasolarcentre.com.au/download?location=alice-springs)
provides the measurements and metadata. The [technology
list](https://www.dkasolarcentre.com.au/locations/alice-springs/technologies)
identifies the fixed north, east, west, and horizontal arrays used for the
orientation experiment. The centre also publishes [data-quality and equipment
change notes](https://dkasolarcentre.com.au/download/notes-on-the-data/p6).

Two tests isolate transposition from the rest of the PV model. The first
compares modelled plane-of-array irradiance with a co-planar pyranometer at
20 degrees tilt and north-facing azimuth. Every tested transposition model was
within `1.8%` of measured annual plane-of-array irradiance.

The second test compares two otherwise matched BP Solar arrays, one facing
north and one facing west, over 2144 screened days from 2009 through 2014.
Perez reproduced the measured west-to-north energy ratio within `0.60%`.
Hay-Davies and Reindl were within `0.30%`, and isotropic and Klucher were
within `1.29%`. The Hay-Davies ratio error varies from year to year with a
standard deviation of `0.86` percentage points over 2009 and 2011–2014, so the
result bounds the transposition error but does not establish one model as
universally best.

Five source samples in the 2009–2020 input have missing global horizontal
irradiance and positive diffuse irradiance:

| Sample (ACST) | Measured diffuse irradiance (W/m2) | In the 2009–2014 window |
|---|---:|---|
| 2012-12-14 11:45 | 867.3 | yes |
| 2014-02-05 12:55 | 644.8 | yes |
| 2014-02-05 13:00 | 636.1 | yes |
| 2015-08-04 10:05 | 310.6 | no |
| 2017-10-24 17:05 | 58.3 | no |

Global irradiance cannot be filled at these samples without imputing it, and a
zero fill leaves the diffuse fraction undefined. Each civil day that holds such
a sample is therefore excluded from the comparison: from both modelled and
measured energy, for every transposition model, before the day and array
screens. The source readings are kept unchanged. This removes two days from
the 2009–2014 comparison and one day each from the 2012, 2014, 2015 and 2017
annual checks. Model output on every other day is identical to the earlier
replay, checked step by step for Hay-Davies on both arrays.

The earlier replay filled the missing global values with zero, so Klucher, whose
diffuse model divides by global irradiance, could not run. For every other
model, the exclusion changes the west-to-north ratio error by at most `0.00134`
percentage points and each array's energy bias by at most `0.004` percentage
points. The 2016 irradiance-only comparison has no such samples and is
unchanged.

The dual-axis array is excluded from this evidence. Its tracking geometry is
not equivalent to the fixed-plane comparison and exposed a separate limitation
in the current dual-axis path.

## IEA PVPS Task 13 module dataset

The [IEA PVPS Task 13 module validation
dataset](https://pvpmc.sandia.gov/datasets/iea-pvps-task-13-module-validation-dataset/)
was produced by IEA PVPS Task 13 and measured at SUPSI PVLab, Switzerland. Sandia's
PV Performance Modeling Collaborative hosts it; Sandia did not produce it.
It contains one year of five-minute outdoor measurements, including
plane-of-array irradiance, ambient temperature, wind speed, and back-of-module
temperature. The dataset accompanies [IEA PVPS report
T13-20:2020](https://iea-pvps.org/key-topics/climatic-rating-of-photovoltaic-modules/),
*Climatic Rating of Photovoltaic Modules: Different Technologies for Various
Operating Conditions*, ISBN `978-3-907281-08-6`.

The check passed measured plane-of-array irradiance, ambient temperature, and
wind speed to the BREOS Faiman temperature calculation. At irradiance of at
least `200 W/m2`, the default coefficients produced `-0.038 C` mean bias,
`2.141 C` mean absolute error, `2.993 C` RMSE, and `0.970` correlation over
`26,023` records.

The reference is back-of-module temperature, while BREOS calculates cell
temperature. Thermal inertia also affects the five-minute comparison. The
result therefore supports the default model at its component boundary. It is
not a full electrical PV validation.

## UCY PHAETHON PV test-bed

The test-bed data are published by George Makrides, University of Cyprus, as
the [*Solar forecasting and digital twin
dataset*](https://doi.org/10.5281/zenodo.15779578) (2025, CC BY 4.0). They come
from a PV system at the Smart Energy Infrastructure, PHAETHON Centre of
Excellence, University of Cyprus, and cover one year of 15-minute records from
June 2015 through May 2016. Cite the dataset as the UCY PHAETHON test-bed. The
file inside the record is named `PCoE-Dataset`, but PCoE usually means NASA
Ames' Prognostics Center of Excellence in the PV literature.

The check passed measured plane-of-array irradiance, ambient temperature, and
wind speed to the BREOS Faiman temperature calculation and compared the result
with the recorded module temperature. At irradiance of at least `200 W/m2`, the
default coefficients produced `+2.694 C` mean bias, `3.161 C` mean absolute
error, `3.821 C` RMSE, and `0.972` correlation over `12,708` records. A PVsyst
freestanding sensitivity run produced `+0.976 C` bias and `2.619 C` RMSE. The
mounting type is not documented, so neither run is preferred on physical
grounds. The recorded DC channels also satisfy `Pdc = Vdc x Idc`, with
`0.184 W` mean bias over `35,136` rows.

The dataset has no module nameplate, inverter rating, site location, or
timezone, so it does not support an electrical PV or inverter validation.

## Reunion Island microgrid

The [experimental stand-alone microgrid
dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC10568554/) from Roche Plate,
Mafate, Reunion Island, is published under [DOI
10.5281/zenodo.8186303](https://doi.org/10.5281/zenodo.8186303). It includes
weather, PV-regulator, inverter, battery, and household load channels.

At inclined irradiance of at least `200 W/m2`, the default Faiman model
overpredicted the measured under-panel surface temperature by `+8.882 C`, with
`11.162 C` RMSE and `0.623` correlation over `108,986` one-minute records. The
reference is a surface temperature under a roof-mounted array, while BREOS
calculates cell temperature, so the result mixes a boundary difference with
mounting and ventilation effects. Over `5,655` hourly records, the default
lumped battery-temperature helper underpredicted the measured battery
temperature by `-3.292 C`, with `4.399 C` RMSE.

These results are evidence for calibration and boundary investigation. They
do not validate either model for this installation, and the dataset has no
SOC trace or nameplate data for a dispatch or degradation check.

## PySAM PVWatts v8 on the same weather

This check compares BREOS with another model, not with measurements. It shows
that BREOS reproduces PVWatts when both use the same inputs. It does not show
that either model is accurate.

PySAM 8.0.0 (PVWatts v8) and BREOS at commit
`bd6c74b14b2229a7bf34cffc829b1cddc70ec8f1` ran on the seven hourly PVGIS
typical weather years in `validation/data/weather/`. Both used the 4 kWp
south-facing reference system from `validation/locations.json`: fixed open
rack, albedo `0.2`, DC/AC ratio `1.2`, `96%` nominal inverter efficiency and
`14.08%` aggregate loss. Each engine kept its own module, optical and
temperature models. BREOS ran with Perez transposition and Marion diffuse IAM,
which are the defaults from 0.9.0. Annual AC energy:

| Site | PySAM (kWh) | BREOS Perez + Marion | BREOS isotropic, no diffuse IAM |
|---|---:|---:|---:|
| Porto | 6243 | `-0.09%` | `-3.31%` |
| Berlin | 4208 | `+0.13%` | `-3.90%` |
| Golden, Colorado | 7393 | `-0.45%` | `-3.43%` |
| Nairobi | 6105 | `-0.43%` | `+1.32%` |
| New Delhi | 6637 | `+0.11%` | `-2.04%` |
| Melbourne | 6094 | `-0.16%` | `-3.64%` |
| São Paulo | 5490 | `-0.03%` | `-0.39%` |

With Perez transposition but no diffuse IAM, BREOS was `0.63%` to `2.03%`
above PySAM. Hourly AC RMSE was 27 to 73 W.

The same replay ran PySAM's irradiance model on the 2016 DKA Solar Centre
plane-of-array comparison. With the same sky model, the annual bias of PySAM
and BREOS differed by at most `0.02` percentage points. Perez had the smallest
bias against the pyranometer (`-0.48%`), and isotropic the largest (`-1.76%`).

The benchmark scripts and outputs are not committed to this repository, like
the measured-data drivers above.

## Evidence not used for the release claim

The Esposende field comparison is part of the upcoming publication, but it is
not a release validation dataset. Its weather is not measured at the
site, and the measurement boundary of its annual series is not verified.

Two other inspected datasets are excluded. The orientation-diversity workbook
has no dates, site coordinates, or panel geometry, so BREOS cannot run on it.
The HKUST metadata has no usable tilt or azimuth, so that check fitted an
effective orientation and an output scale on earlier years before evaluating
2023. It is a calibrated forward check, not an independent test.

## Reproducing the upcoming publication

The configurations, drivers, and run records for the upcoming publication
were removed from the repository after 0.6.2. They are preserved in the
[BREOS 0.6.2 archive](https://doi.org/10.5281/zenodo.22938914) and at the
[`v0.6.2` tag](https://github.com/Str4vinci/breos/tree/v0.6.2), under
`validation/article1/` and `tools/`. Reproduce the upcoming publication's
numbers from that release, not from a later version.
