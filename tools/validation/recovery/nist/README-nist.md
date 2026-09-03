# External validation: NIST Gaithersburg Ground array, 2016

An independent check of the BREOS PV chain against a fully documented,
third-party instrumented array, chosen to avoid the two weaknesses of the
Esposende comparison: an unverified measurement boundary and weather recorded
30 km from the site.

## Source

NREL PVDAQ `system_id=4902` (`NIST_Ground_1`) on the OEDI data lake, no
credentials required:

```
https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/pvdata/system_id=4902/year=2016/month=<m>/day=<d>/system_4902__date_2016_<mm>_<dd>.csv
```

Documentation: Boyd (2017), *Performance Data from the NIST Photovoltaic Arrays
and Weather Station*, J. Res. NIST 122:40, doi:10.6028/jres.122.040; channel
units from the NIST data dictionary, https://www.nist.gov/file/391591.

271 kW DC, 1152 Sharp NU-U235F2, 20 deg tilt, azimuth 180, one PV Powered
PVP260kW inverter, open field, no battery. Both the module and the inverter are
present in the CEC databases under their exact model names, so no parameter is
guessed.

## Facts established from the raw data, not assumed

* **Timestamps are Local Standard Time (EST, UTC-5), no DST.** Confirmed twice:
  by the paper, and independently by the measured POA window (04:40 to 19:23 on
  2016-06-15) against computed sunrise 04:42 and sunset 19:36.
* **Pyranometers are published as raw millivolts** with no sensitivity constant.
  The only irradiance in engineering units is plane-of-array (silicon reference
  cell). There is no usable GHI for this array, so the chain is driven from
  measured POA.
* **The inverter's own metering is unusable in 2016.** `InvPAC_kW` integrates to
  *negative* monthly energy in April (-40 MWh) and October (-56 MWh), and is
  absent for most of June and July. The independent revenue-grade AC meter is
  used instead, cross-checked against its own cumulative kWh counter: they agree
  within 0.1 % in all twelve months, with 100 % coverage.
* **DC reference is the sum of the seven combiner-box shunts.** Zero at night,
  six channels identical to 0.1 %, the seventh at 0.861 of the others, matching
  its 12-of-14 source circuits. Implied inverter efficiency 0.9598 against the
  CEC datasheet's 0.9636.
* **28 of 366 days are not normal operation** -- multi-day outages in April and
  October, and snow cover during the January 2016 blizzard. They are screened
  and reported, not absorbed into a loss coefficient.

## Reproducing

```
python tools/validation/nist_build.py    --raw-dir <dir of day CSVs> --out nist_ground_2016.csv.gz
python tools/validation/nist_validate.py --data nist_ground_2016.csv.gz
PYTHONPATH=tools/validation python tools/validation/nist_analysis.py \
    --data nist_ground_2016.csv.gz --outdir results/
```
