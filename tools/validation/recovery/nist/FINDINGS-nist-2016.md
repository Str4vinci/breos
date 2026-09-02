# External validation of the BREOS PV chain: NIST Gaithersburg, 2016

Findings from the first external validation run. Written as a handoff document:
it assumes no prior context from the session that produced it.

- **Worktree:** `/tmp/breos-external-validation`
- **Branch:** `validate/external-pv-datasets`, branched from `origin/develop` (`e58f6e6`)
- **Harness commit:** `a508747`
- **Environment:** `UV_CACHE_DIR=/tmp/breos-ext-uv uv run --offline python …` (pvlib 0.15.2, pandas 3.0.5, Python 3.13.14)
- **Report:** https://claude.ai/code/artifact/52b998cc-28e5-4632-a60b-c9d19b23abc5

---

## Correction and follow-up added on 2026-08-29

The first run used the flattened OEDI PVDAQ copy. That copy exposes the Ground
pyranometers as raw millivolts without calibration constants. The report
incorrectly generalized that limitation to the NIST source dataset.

The canonical NIST bulk archives contain calibrated thermopile channels in
W/m². They include Ground GHI and POA, plus weather-station GHI, DNI, and DHI.
The original POA-driven PV-chain results remain valid. The statements that
NIST had no usable GHI and could not support a transposition test do not.

A follow-up downloaded and audited the complete 2016 one-minute Ground,
weather-station logger 1, and weather-station logger 2 archives. It then
compared local measurements with Open-Meteo, four GHI decomposition models,
and five transposition models. Section 5 records that work. Section 6 relates
it to the independent DKASC Alice Springs transposition test.

## 1. Why this array

The Esposende comparison for the forthcoming publication could not be closed
for two reasons. Neither is now recoverable. The measured series was a Growatt
portal field of unverified definition, with a battery inside the metering
boundary. The weather came from Porto Airport, roughly 30 km from the site.

The NIST Ground array removes both. Grid-tied, no battery, metered independently
twice over, and instrumented with co-located irradiance, ambient temperature, wind
and seven module backsheet RTDs.

**Configuration** (Boyd 2017, J. Res. NIST 122:40, doi:10.6028/jres.122.040):

| | |
|---|---|
| Rated DC | 271 kW (1152 × Sharp NU-U235F2, 235 W) |
| Tilt / azimuth | 20° / 180°, open field, 0.67 m above ground |
| Inverter | 1 × PV Powered PVP260kW, 260 kW AC |
| Strings | 12 modules/string, 96 source circuits, 7 combiner boxes |
| Location | 39.1319 °N, −77.2141 °E, 138 m |
| Commissioned | July 2012 |

Both `Sharp_NU_U235F2` and `PV_Powered__PVP260KW__480V_` exist in the CEC
databases under those exact names, so **no module or inverter parameter was
guessed**.

**Source data.** The first run used this open PVDAQ copy:

```
https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/pvdata/
    system_id=4902/year=2016/month=<m>/day=<d>/system_4902__date_2016_<mm>_<dd>.csv
```

366 daily CSVs, 525,847 one-minute rows, 99.8 % complete.

---

## 2. Facts established from the raw data, not assumed

These four checks changed the analysis. Three of them contradict the channel a
first pass would reach for.

1. **Timestamps are Local Standard Time (EST, UTC−5), no DST.** NIST's
   publication confirms this fact. An independent check compares the measured
   irradiance window on 2016-06-15, from 04:40 to 19:23, with computed sunrise
   at 04:42 and sunset at 19:36. The daylight-saving reading is off by a full
   hour.

2. **The PVDAQ pyranometer channels are unusable, but the NIST bulk channels
   are calibrated.** The PVDAQ copy publishes the Eppley PSP and Kipp & Zonen
   CMP11 as raw millivolts without sensitivity constants. Its
   `irradiance_ghi_*` and `irradiance_poa_*` labels are also unreliable. The
   canonical NIST bulk archives provide `Pyra1_Wm2_Avg` and
   `Pyra2_Wm2_Avg` in W/m². The weather-station archives also provide
   calibrated GHI, DNI, and DHI. The first run did not ingest those archives,
   so it could not use the calibrated channels.

3. **The inverter's own metering is corrupt in 2016.** `InvPAC_kW` integrates to
   *negative* monthly energy in April (−40 MWh) and October (−56 MWh), and is
   absent for most of June and July. `InvPDC_kW` fails the same way. Neither is
   usable.

4. **Two independent references survive scrutiny and were used instead:**
   - *AC.* The revenue-grade meter (`PwrMtrP_kW`, PVDAQ `ac_power_meter_1864_5`).
     Cross-checked against its own cumulative kWh counter
     (`ac_power_meter_1864_4`): they agree within **0.1 % in all twelve months**,
     100 % coverage. Annual 371,271 kWh integrated / 371,628 kWh counter.
   - *DC.* The sum of the seven combiner-box shunts. Zero at night; six channels
     identical to 0.1 %; the seventh at 0.861 of the others, matching its 12-of-14
     source circuits. Annual 386,816 kWh, 100 % coverage.
   - Implied inverter efficiency **0.9598**, against the CEC datasheet's 0.9636.

**Outage screening.** Of the 366 days, 28 are not normal operation. They include
multi-day outages in April and October, plus snow cover during the January 2016
blizzard. They cost **4.23 %** of modelled annual output. They are excluded from
the headline figures and reported separately, not absorbed into a loss
coefficient. Including them drops daily correlation from 0.998 to 0.960, which
is what first exposed them.

---

## 3. Method

Because the first PVDAQ extraction had no calibrated GHI, the chain is driven
from **measured plane-of-array irradiance** (`RefCell1_Wm2`) plus measured
ambient temperature and wind speed.

This is a deliberate trade, and it cuts both ways:

- It removes decomposition, transposition and weather-station-distance error from
  the comparison entirely. This isolates the module → thermal → inverter
  sub-chain, which is exactly where the Esposende residual lives.
- It means **transposition is not exercised at all**. The Perez-versus-isotropic
  question that matters most at Esposende is untested by this run.

---

## 4. Results

### Stage by stage

| Stage | Result |
|---|---|
| Inverter alone (measured DC → BREOS PVWatts inverter → measured AC) | **−0.06 %**; hourly r = 0.99998, hourly nRMSE 0.91 % |
| Cell temperature vs mean of 7 backsheet RTDs (POA > 400 W/m²) | `faiman` **+2.19 °C**, RMSE 4.39 °C, r 0.9635 |
| | `pvsyst-freestanding` +2.76 °C, RMSE 5.18 °C |
| | `sapm-open-rack-glass-glass` +5.57 °C, RMSE 6.82 °C |

Backsheet temperature runs a couple of degrees below true cell temperature, so
BREOS's default Faiman model is close to unbiased against the quantity it predicts.
The SAPM open-rack preset is clearly too hot for this mounting.

### Loss ladder: annual AC bias, 338 normal days

| Loss configuration | Stack | Bias | Daily r |
|---|---|---|---|
| Module only | 0 % | **+5.65 %** | 0.998 |
| Module only, 4 years' age at 0.5 %/yr | 0 % (+2.23 % deg.) | **+3.32 %** | 0.998 |
| No availability, no shading | 8.75 % | −3.51 % | 0.998 |
| No availability | 11.5 % | −6.43 % | 0.998 |
| `DEFAULT_PVWATTS_LOSSES` | 14.1 % | −9.26 % | 0.998 |

**The array's true effective system loss is 5.35 %.** BREOS's 14.1 % default
overstates it by ~8.7 percentage points and makes BREOS under-predict a
well-maintained array by 9.26 %.

Daily correlation is 0.998 at every rung: the loss assumption moves the level,
never the shape.

### Residual structure is flat

| Regime | Days | Bias |
|---|---|---|
| Clear (k<sub>t</sub> > 0.85) | 148 | −3.56 % |
| Mixed (0.5–0.85) | 98 | −2.97 % |
| Overcast (≤ 0.5) | 92 | −4.74 % |

Monthly bias stays between −1.8 % and −5.2 % with no seasonal trend. There is
**no clear-day concentration**, which is the opposite of Esposende.

### Measurement floor

Two co-located calibrated POA sensors (`RefCell1_Wm2`, `SEWSPOAIrrad_Wm2`) differ
by 0.64 % in annual total and shift the resulting bias by 1.65 percentage points.
**A bias below roughly 1.7 % is not resolvable at this site**, whatever the model
does.

### Sanity checks

| | |
|---|---|
| Annual POA | 1677.2 kWh/m² |
| Measured DC | 386,816 kWh, 85.1 % of the STC-efficiency ceiling |
| Measured AC | 371,271 kWh |
| Specific yield | 1370 kWh/kWp |

---

## 5. Weather-chain follow-up

### Sources and time treatment

The follow-up used the official NIST PV Data portal and retained all calibrated
channels needed for a weather-chain comparison. The three 2016 archives each
contain 366 daily files and 527,040 one-minute rows.

- Canonical source: https://pvdata.nist.gov/
- Dataset DOI: https://doi.org/10.18434/M3S67G

| Source | SHA-256 |
|---|---|
| Ground | `f12498294ad3bec41150a9263ac28344563bc24896d907a8b05576d18ae1abf8` |
| Weather station logger 1 | `fc63b434da6f86c0b8dfe96195f99ea15b4f7c0b34076d7407836652901efa1b` |
| Weather station logger 2 | `49de8481f17af242be0c79f6b5513686fbb5d7be4baf975a00347444586c8375` |

NIST records one-minute averages in fixed EST, UTC−5, without daylight saving.
The follow-up converted them to UTC and formed right-labelled hourly means.
Each retained channel needed at least 50 valid minutes per hour. Exact NIST
sentinels were masked before aggregation. No irradiance value was interpolated.

The calculation treated the one-minute NIST stamp as the interval end. NIST
does not state this label basis as clearly as Open-Meteo states its hourly
convention. The remaining ambiguity is one minute and cannot reproduce the
one-hour mismatch found in the earlier Open-Meteo treatment.

Open-Meteo's unsuffixed radiation fields were treated as preceding-hour means
with right-edge labels. Solar position was calculated 30 minutes before each
label. The `*_instant` fields were evaluated at their labels and reported as a
separate counterfactual.

### The local measurement floor is not small

The Ground array and weather station are 0.729 km apart. Open-Meteo maps both
coordinates to the same radiation grid cell. The local instruments do not read
the same annual irradiation.

| Paired measurement | Annual difference | Hourly r |
|---|---:|---:|
| Weather-station thermopile GHI vs Ground thermopile GHI | +4.50 % | 0.9975 |
| Weather-station 20° silicon POA vs Ground silicon POA | −2.22 % | 0.9981 |
| Ground silicon POA vs Ground thermopile POA | −1.73 % | 0.9996 |

These differences combine sensor response, calibration, horizon, and local
weather. They do not isolate distance alone. They do show that a gridded source
cannot represent all local variation, even across less than one kilometre.

### Open-Meteo gets the annual total closer than the components

The interval-mean Open-Meteo fields were compared with exact overlapping NIST
hours. DNI and DHI use the weather-station instruments because the Ground array
does not measure those components.

| Open-Meteo quantity | Annual bias | Hourly r |
|---|---:|---:|
| GHI vs Ground thermopile GHI | +2.84 % | 0.9618 |
| DNI vs weather-station DNI | +11.46 % | 0.8924 |
| DHI vs weather-station DHI | −22.10 % | 0.8919 |
| Components plus Perez vs Ground thermopile POA | +2.35 % | 0.9567 |
| Provider GTI vs Ground thermopile POA | −1.12 % | 0.9576 |

The local measured components reconstruct local GHI to +1.94 %, with hourly
r = 0.99984. The larger and opposite Open-Meteo DNI and DHI errors cancel after
transposition. Annual POA therefore looks much better than the component and
hourly errors that produced it.

Explicit ERA5 and ERA5-Seamless runs give almost the same answer. Their Perez
POA biases are +2.48 % and +2.33 %. Open-Meteo Best Match changed nine
2016-12-31 radiation rows when the query window was extended, with a maximum
GHI difference of 70 W/m². Explicit model selection is the reproducible choice.

### Decomposition changes annual POA by about one percentage point

Ground measured GHI was decomposed and transposed with Perez. The target is the
co-located Ground thermopile POA.

| Decomposition | POA bias | Hourly r |
|---|---:|---:|
| Boland | −3.96 % | 0.9978 |
| Dirint | −3.10 % | 0.9985 |
| DISC | −2.68 % | 0.9990 |
| Erbs | −3.19 % | 0.9983 |

The annual range is 1.28 percentage points. Repeating the decomposition on
Open-Meteo GHI gives a 1.60 percentage-point range. Open-Meteo supplied
components produce about 1.5 % more modelled AC than Open-Meteo GHI plus
Dirint, with all other PV inputs fixed.

### Transposition changes level more than shape at this site

The weather-station GHI, DNI, and DHI were transposed to its co-planar 20° south
silicon sensor. The comparison uses effective irradiance because the sensor has
a silicon spectral and angular response.

| Transposition | Annual bias | Hourly r |
|---|---:|---:|
| Isotropic | +1.23 % | 0.99897 |
| Hay-Davies | +3.13 % | 0.99910 |
| Reindl | +3.18 % | 0.99909 |
| Perez | +4.17 % | 0.99920 |
| Perez-Driesse | +4.21 % | 0.99920 |

Perez raises annual effective irradiation by about 2.9 % relative to isotropic
and improves correlation slightly. This result does not select isotropic as a
canonical model. The target is a silicon sensor on a white-gravel roof with a
documented southeast obstruction. It establishes a site-specific model spread.

### Annual PV agreement selects the wrong weather treatment

Absolute PV-to-meter biases include snow, outages, and system behaviour. The
follow-up did not remove observations by model residual. The PV results are
descriptive and were not used to choose a weather treatment.

| Weather treatment | AC bias vs meter | Daily r |
|---|---:|---:|
| Ground GHI plus Dirint and local met | −3.75 % | 0.900 |
| Weather-station components plus Perez and local met | +2.52 % | 0.864 |
| Open-Meteo components plus Perez and local met | +2.46 % | 0.794 |
| Open-Meteo GHI plus Dirint and local met | +0.95 % | 0.794 |

Open-Meteo GHI plus Dirint gives the closest annual total and the weakest daily
tracking. Local Ground GHI gives a worse annual bias and much better daily
correlation. Annual energy alone cannot identify the best weather treatment
because unrelated errors cancel.

On the same 8,699 hours, substituting Open-Meteo temperature and wind for local
measurements increases modelled AC by 0.78 %. Meteorological inputs matter, but
they do not explain a three-percentage-point difference alone in this test.

### Provenance and verification

The canonical follow-up result is
`/tmp/nist-weather-chain-results-final-v5-20260829/`. Its processed NIST input
hash is
`6b7d4e26030dc4fad5059c5407deae0cb8155a93290740cdbf6d9eeae05e3f47`.
The Open-Meteo Best Match, ERA5, and ERA5-Seamless hashes are recorded in
`provenance.json`. The resolved-configuration hash is
`b3bc4d708551cb51f8371a594449e270a945707b01c3f7fe24ff54b76641818c`.
The run used BREOS commit `c82a32a8d606d66164132024f47428b82fed4307`.
Its provenance records `worktree_dirty: true` because the follow-up scripts
were untracked. The deterministic outputs are diagnostic evidence, not release
artifacts, until those scripts are reviewed and committed.

Two independent runs produced byte-identical data files. Thirteen focused
tests passed. Ruff check and formatting passed. The full BREOS suite reported
1,030 passed, 8 skipped, 2 deselected, and one failure. The failure attempted a
live PVGIS request in a network-restricted environment and is unrelated to the
weather-chain work.

## 6. Cross-read against DKASC and Esposende

The independent DKASC test closes the transposition question with a cleaner
orientation experiment. It uses both a co-planar tilted pyranometer and a
physically matched north and west array pair. See `FINDINGS-dkasc.md` for the
full analysis.

| DKASC result | Bound |
|---|---:|
| All models vs co-planar annual POA | within 1.8 pp |
| All models vs measured west-to-north ratio | within 1.5 pp |
| Isotropic-to-Perez span on co-planar POA | 1.3 pp |
| Year-to-year floor on the orientation ratio | 0.86 pp |

Perez leads the co-planar-instrument test. Hay-Davies and Reindl lead the
orientation-ratio test. Their separation is smaller than the measurement floor,
so the data support a bound rather than one winner. Isotropic is last or
near-last in both legs.

DKASC also shows that decomposition is the larger input uncertainty. Replacing
measured DHI with Dirint, DISC, or Erbs changes annual POA by 1.7 to 3.9
percentage points and doubles or triples the five-minute RMSE. The NIST
follow-up reaches the same qualitative conclusion: component handling and
weather-source errors can cancel in annual POA while remaining visible hour by
hour.

The DKASC dual-axis result is separate from fixed-tilt transposition. BREOS
overpredicts the dual-axis array by 23 % to 29 % because the dual-axis branch
points each tracker at the sun but has no inter-tracker shading or backtracking
term. `gcr` and `backtrack` apply only to the single-axis branch. This is a
dual-axis modelling limitation, not an explanation for Esposende.

The DKASC data audit also corrected two easy ways to contaminate the result.
The timezone check uses one measurement-only clear-sample mask for every
candidate offset. Rebuilding the mask for each offset incorrectly selects
UTC+9:00, while the common mask selects ACST at UTC+9:30. The source audit also
confirms source 214 as site 32 and rejects source 59 as site 38. Neither source
enters the transposition analysis.

Both runs used the same BREOS default 14.1 % stack, so removing it puts them on a
common footing.

| Site | As configured | Loss-free equivalent | Driven from |
|---|---|---|---|
| Esposende 2024 | +8.57 % | **+26.4 %** | GHI, 30 km away |
| NIST Ground 2016 | −9.26 % | **+5.65 %** | measured POA, on-site |

The Esposende reference series sits roughly **20 percentage points** lower, relative
to a loss-free BREOS run, than a fully documented array with verified metering does.
Transposition and decomposition error at Esposende could account for a few points of
that; they cannot account for twenty.

**What this establishes:** BREOS's module, thermal and inverter physics are accurate
to about +5.7 % before any loss stack, and about +3.3 % once array age is allowed
for. The Esposende residual is several times larger than anything this validation
can attribute to the model.

**What it does not establish:** it does not identify what the Growatt portal field
measured, and it does not recover the PVsyst project.

---

## 7. Limits

- **The original POA-driven run does not test transposition.** The bulk-archive
  follow-up and the independent DKASC experiment now cover that stage.
- One site, one year, one module technology: mono-silicon, 20° tilt, open field,
  Mid-Atlantic climate.
- The POA sensor is a **silicon reference cell**, not a thermopile. Its spectral and
  angular response resemble the modules', which suits this comparison, but it
  partially embeds incidence-angle behaviour that would otherwise be modelled.
- The 5.35 % figure applies to *this array*, with cleaned radiometers,
  research-grade maintenance, and an age of four years. It is not a general
  recommendation for the default stack.
- Array age is inferred from the July 2012 construction date; the 0.5 %/yr rate is
  BREOS's assumption, not a NIST measurement.

---

## 8. Reproducing

```bash
cd /tmp/breos-external-validation
export UV_CACHE_DIR=/tmp/breos-ext-uv

uv run --offline python tools/validation/nist_build.py \
    --raw-dir <dir of day CSVs> --out nist_ground_2016.csv.gz

uv run --offline python tools/validation/nist_validate.py \
    --data nist_ground_2016.csv.gz

PYTHONPATH=tools/validation uv run --offline python tools/validation/nist_analysis.py \
    --data nist_ground_2016.csv.gz --outdir results/
```

Built artefacts from this run live outside the repo, in the session scratchpad:
`nist_ground_2016.csv.gz` (28 MB consolidated frame), `results/sweep.csv`,
`results/regimes.csv`, `results/monthly.csv`, `results/excluded_days.csv`.

**Note:** the project's dependency set has no parquet engine, so the consolidated
frame is written as gzipped CSV.
