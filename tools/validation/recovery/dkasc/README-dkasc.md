# External validation: DKASC Alice Springs

The second external validation of the BREOS PV chain, chosen specifically to
close the gap the NIST Gaithersburg run could not: **transposition**.

NIST publishes its pyranometers as raw millivolts with no sensitivity constant,
so the only irradiance available in engineering units there was already
plane-of-array. Decomposition and transposition were never exercised. DKASC
logs Global Horizontal *and* Diffuse Horizontal radiation in W/m2, so the whole
chain runs: GHI, DHI -> DNI -> transposition -> POA -> DC -> AC.

## Source

Desert Knowledge Australia Solar Centre, Alice Springs, Northern Territory
(-23.7624, 133.8745, 545 m). Open data, no credentials:
<https://dkasolarcentre.com.au/download?location=alice-springs>, one CSV per
"source id", 5-minute resolution, 2008 to present.

The numeric filename prefix is the DKASC source id. The id -> meter mapping is
confirmed two independent ways: by the column names inside the site-wide
`Alice_Springs_2025.csv` export, which spell out `<id>_DKA_<meter>_<phase>`, and
by the per-source pages on dkasolarcentre.com.au.

| id | meter | array | rating | modules | orientation | commissioned |
|---|---|---|---|---|---|---|
| 100 | DKA-M1 A | 16A | 1.98 kW | 12 x BP 3165J poly | 20 deg, azimuth 0 (north) | 2008-11-11 |
| 81 | DKA-M2 A | 16D | 1.98 kW | 12 x BP 3165J poly | 20 deg, azimuth 270 (west) | 2008-11-11 |
| 84 | DKA-M5 B | 12 | 5.1 kW | 30 x BP 4170N mono | 20 deg, azimuth 0 | 2008-11-11 |
| 92 | DKA-M6 B | 13 | 5.25 kW | 30 x Trina TSM-175DC01 mono | 20 deg, azimuth 0 | 2009-01-08 |
| 91 | DKA-M9 B | 1A | 10.5 kW | 60 x Trina TSM-175DC01 mono | dual-axis tracking | 2013-08-14 |
| 101 | weather station | - | - | - | tilted sensor at 20 deg, azimuth 0 | - |

Two further sources were downloaded and are handled by `dkasc_build.py` but are
not analysed, and one supplied mapping turned out to be wrong:

* **214 = site 32**, Canadian Solar 5.3 kW poly, fixed, 2016 -- confirmed. Its
  record starts 2016-11-09, matching that commissioning date.
* **59 is not site 38.** Site 38 (Q CELLS 5.9 kW mono, 2017) is on M19
  *B*-phase; source 59 is M19 *C*-phase and carries an array this run did not
  identify. The data corroborates the correction independently: source 59 reads
  exactly zero until 2018, so it cannot be a 2017 commissioning either.

Site 16 is DKASC's deliberate orientation experiment: four physically identical
2 kW BP arrays commissioned the same day, differing only in azimuth (16A north,
16B flat, 16C east, 16D west). 16A and 16D are the controlled pair used here.
In the southern hemisphere north is the sun-facing orientation, so 16D is the
off-axis array whose output is far more sensitive to the sky-diffuse model.

**Modules are not in the CEC database.** BP Solar no longer exists and the Trina
DC01 series predates the current listing, so unlike the NIST run these
parameters come from manufacturer datasheets. Each is cross-checked against the
array area DKASC publishes independently, and agrees to better than 0.2
percentage points of module efficiency in every case.

## Facts established from the raw data, not assumed

Run `dkasc_facts.py` to reproduce all of these.

* **Timestamps are ACST (UTC+9:30), no daylight saving.** Fitting measured GHI
  against clear-sky GHI, on one fixed sample set chosen from measurements alone
  and least-squares scaled per candidate, gives 120.3 W/m2 RMSE at +9:30 against
  137.1 (+9:00), 142.1 (+10:00) and 199.4 (+10:30, i.e. ACDT). Daylight saving
  is excluded outright. Alice Springs is in `Australia/Darwin`, permanently
  +9:30, which is the site's own zone rather than a stand-in.
  The mask must not be built from a candidate offset's own clear-sky series;
  doing so lets each offset pick the samples that suit it and ranks +9:00 first.
* **The stamp labels the centre of the 5-minute averaging window.** The same fit
  restricted to low-sun clear samples minimises at -0.25 min; interval-start
  would give +2.5 and interval-end -2.5. Solar position is therefore evaluated
  *at* the stamp, which is BREOS's default `interval-start` setting. This
  differs from the NIST array, whose logger stamped at interval end.
* **The tilted pyranometer is co-planar with the fixed arrays.** Fitting its
  orientation from the data alone gives 21 deg / azimuth 359, against DKASC's
  published 20 deg / azimuth 0. The published geometry is what the validation
  uses; the fit is only a check on it -- and on the hemisphere convention, since
  a sign error would have placed it near 180.
* **Channels are already in engineering units**, so unlike NIST there is no
  millivolt channel to convert. What they do carry is out-of-range spikes and
  sentinel values -- GHI to -986 and +2726 W/m2, wind to -1742 m/s -- which are
  screened rather than modelled.
* **Two independent metering channels agree.** Integrated `Active_Power` and the
  differenced cumulative `Active_Energy` counter agree within 0.3-1.0 % for
  every array-year from 2009 to 2020, and diverge sharply from 2021 (to 0.83 by
  2024) as the 5-minute record starts dropping samples the counter still
  accrues. That divergence is what puts 2021 onward out of bounds.
* **The site master meter is unusable.** Source 96 (`MasterMeter1`) fails the
  same cross-check outright, with power-to-counter ratios of -0.55 to -0.82 in
  six separate years, because its counter is a net meter that runs backwards.
  Its 5-minute power channel is 52 % absent in 2016 as well. Per-array meters
  are used throughout; the master meter is never a generation reference.
* **Wind speed ends permanently in late October 2016.** It is interpolated over
  short gaps and median-filled beyond them; the sensitivity to that fill is
  reported, and it is common to both arrays of the pair in any case.

## Screening

Two failure modes exist and they pull the bias in opposite directions, so they
are screened separately:

* **Weather-station outage** -- the pyranometers read identically zero for whole
  days while the arrays keep producing. 2016 contains one contiguous 18-day
  block, 18 June to 5 July. Any irradiance-driven model predicts zero on those
  days, so leaving them in *understates* the bias. This is the mirror image of
  NIST, where the array was down and the model over-predicted.
* **Array outage or curtailment** -- the array underperforms a valid irradiance
  record. 2016-01-30 is depressed for every array on the site at once.

The array screen normalises by each array's own median measured/model ratio
before thresholding. An absolute threshold would cut the low tail of a
distribution the model's own bias has already shifted, quietly flattering the
result.

## Reproducing

```
python tools/validation/dkasc_facts.py --raw-dir <dir of DKASC CSVs>

python tools/validation/dkasc_build.py --raw-dir <dir> \
    --out dkasc_2016.csv.gz --start-year 2016 --end-year 2016 \
    --sources 100 81 84 92 91 96
python tools/validation/dkasc_build.py --raw-dir <dir> \
    --out dkasc_2009_2020.csv.gz --start-year 2009 --end-year 2020 \
    --sources 100 81 84 92 91

PYTHONPATH=tools/validation python tools/validation/dkasc_transposition.py \
    --gti-data dkasc_2016.csv.gz --pair-data dkasc_2009_2020.csv.gz --outdir results/
PYTHONPATH=tools/validation python tools/validation/dkasc_analysis.py \
    --data dkasc_2009_2020.csv.gz --outdir results/ \
    --ladder-years 2009:2014 --tracker-years 2014:2016
```

Built artefacts live outside the repo, in the session scratchpad. The project's
dependency set has no parquet engine, so frames are written as gzipped CSV.

Findings: `FINDINGS-dkasc.md`.
