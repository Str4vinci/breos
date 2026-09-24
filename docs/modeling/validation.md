# External validation

BREOS 0.6.0 was checked against five public measured datasets. Each check
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

The checks were run from clean BREOS commit
`f62f4f5bf3c14140ab189d35ea2885e6fcc60c6b`. Raw third-party data and generated
results are not committed to this repository. The drivers, input manifests,
and run records are preserved in the
[BREOS 0.6.2 archive](https://doi.org/10.5281/zenodo.22938914) under
`validation/external/` and `tools/validation/recovery/`.

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
north and one facing west, over 2,146 screened days from 2009 through 2014.
Perez reproduced the measured west-to-north energy ratio within `0.60%`.
Hay-Davies and Reindl were within `0.30%`, and isotropic was within `1.29%`.
The year-to-year variation of the measured ratio was `0.86` percentage points,
so the result bounds the transposition error but does not establish one model
as universally best.

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

## Evidence not used for the release claim

The Esposende field comparison is part of the upcoming publication, but it is
not a release validation dataset. Its weather is not measured at the
site, and its annual series comes from a hybrid inverter with a battery inside
an unverified measurement boundary.

Two other inspected datasets are excluded. The orientation-diversity workbook
has no dates, site coordinates, or panel geometry, so BREOS cannot run on it.
The HKUST metadata has no usable tilt or azimuth, so that check fitted an
effective orientation and an output scale on earlier years before evaluating
2023. It is a calibrated forward check, not an independent test.
