# External measured-data checks

BREOS 0.6.0 was checked against three public measured datasets. Each check
isolates a part of the photovoltaic model. None of the datasets supports one
unqualified weather-to-AC accuracy score, so the results are reported at the
measurement boundary that the source data can support.

| Dataset | Component checked | Main result |
|---|---|---|
| NIST Gaithersburg Ground Array | Module, temperature, and inverter calculations from measured plane-of-array irradiance | The inverter calculation differed from the independent AC meter by `-0.06%`, with hourly `r = 0.99998`. |
| DKA Solar Centre, Alice Springs | Fixed-plane irradiance transposition | All tested models were within `1.8%` of annual measured plane-of-array irradiance. Perez reproduced the measured west-to-north energy ratio within `0.60%`. |
| IEA PVPS Task 13 module dataset | Faiman module-temperature calculation | At plane-of-array irradiance of at least `200 W/m2`, bias was `-0.038 C`, RMSE was `2.993 C`, and `r = 0.970` over `26,023` records. |

The checks were run from clean BREOS commit
`f62f4f5bf3c14140ab189d35ea2885e6fcc60c6b`. Raw third-party data and generated
results are not committed to this repository.

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

The Sandia PV Performance Modeling Collaborative publishes the [IEA PVPS Task
13 module validation dataset](https://pvpmc.sandia.gov/datasets/iea-pvps-task-13-module-validation-dataset/).
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

## Evidence not used for the release claim

The Esposende field comparison remains part of the forthcoming publication,
but it is not a release validation dataset. Its weather is not measured at the
site, and its annual series comes from a hybrid inverter with a battery inside
an unverified measurement boundary.

Other inspected datasets either lack the geometry and equipment metadata
needed for an independent PV calculation or test data ingestion rather than a
physical BREOS model. They are not included in the 0.6.0 validation claim.
