# External validation of the BREOS PV chain: DKASC Alice Springs

Findings from the second external validation run, chosen to close the one gap
the NIST Gaithersburg run could not: **transposition**. Written as a handoff
document; it assumes no prior context from the session that produced it.

- **Worktree:** `/tmp/breos-external-validation`
- **Branch:** `validate/external-pv-datasets`, branched from `origin/develop` (`e58f6e6`)
- **Environment:** `UV_CACHE_DIR=/tmp/breos-ext-uv uv run --offline python …` (pvlib 0.15.2, pandas 3.0.5, Python 3.13.14)
- **Companion:** `FINDINGS-nist-2016.md`, `README-dkasc.md`

---

## 1. Why this array

The first NIST run established that BREOS's module, thermal, and inverter
physics are accurate to about +5.7 % before any loss stack. That run used the
flattened PVDAQ copy, where the pyranometer channels are raw millivolts without
calibration constants. The canonical NIST bulk archives do contain calibrated
GHI and POA channels. A later follow-up used those channels and is recorded in
`FINDINGS-nist-2016.md`.

The first NIST run did not exercise decomposition or transposition. Going into
the DKASC work, Perez versus isotropic was the largest open modelling choice in
BREOS, worth a suspected 5 % of annual energy at Esposende. BREOS's default is
`isotropic`.

DKASC logs Global Horizontal and Diffuse Horizontal radiation in W/m², which
makes the whole chain testable, and it offers two independent ways to test it.

**A co-planar reference instrument.** The weather station carries a tilted
pyranometer at 20° / azimuth 0, the same plane as every fixed array on site.
Modelled POA can be compared straight against it, with no PV model in the way.

**A controlled orientation pair.** Site 16 is DKASC's deliberate orientation
experiment: four physically identical 2 kW BP arrays commissioned the same day,
differing only in azimuth. Two of them are the pair used here.

| | 16A | 16D |
|---|---|---|
| Source id | 100 (`DKA-M1 A`) | 81 (`DKA-M2 A`) |
| Modules | 12 × BP 3165J, 165 W poly | 12 × BP 3165J, 165 W poly |
| Rating | 1.98 kW | 1.98 kW |
| Tilt / azimuth | 20° / **0° (north)** | 20° / **270° (west)** |
| Inverter | SMA SB 2500 | SMA SB 2500 |
| Commissioned | 2008-11-11 | 2008-11-11 |

In the southern hemisphere north is the sun-facing orientation, so 16D is the
off-axis array. A west-facing plane takes a far larger share of its annual
energy as sky diffuse, which is exactly the term the transposition models
disagree about. One irradiance record has to reproduce both.

The pair's real value is that **module parameters, inverter efficiency, the loss
stack, array age and site soiling are common to both arrays and divide out of
the west/north energy ratio.** That ratio is the headline metric below, and it
is far more robust than either absolute bias.

Location −23.7624, 133.8745, 545 m. Site 12 (30 × BP 4170N mono, 5.1 kW) and
site 13 (30 × Trina TSM-175DC01 mono, 5.25 kW), both 20° / north, are used as
independent controls, and site 1A (two DEGERenergie dual-axis trackers, 60
Trina modules, 10.5 kW) checks the tracking path.

**Unlike the NIST run, module parameters are not sourced from the CEC database.**
BP Solar no longer exists and the Trina DC01 series predates the current
listing, so all three module definitions come from manufacturer datasheets. Each
is cross-checked against the array area DKASC publishes independently and agrees
to better than 0.2 percentage points of module efficiency. The uncertainty is
largest exactly where it does least damage: it cancels in the north/west ratio,
which is the number this run turns on.

---

## 2. Facts established from the raw data, not assumed

Reproduce with `dkasc_facts.py`. Three of them contradict what a first pass
would assume, and one corrects the source-ID mapping this run was handed.

1. **Timestamps are ACST (UTC+9:30), no daylight saving.** Fitting measured GHI
   against clear-sky GHI on 103,888 bright samples from clear days, with each
   candidate least-squares scaled so the statistic measures curve shape rather
   than the clear-sky model's own calibration, gives **120.3 W/m² RMSE at
   +9:30** against 137.1 (+9:00), 142.1 (+10:00) and 199.4 (+10:30, i.e. ACDT).
   The two nearest candidates bracket +9:30 near-symmetrically, which is what a
   true minimum there looks like, and daylight saving is excluded outright.
   Fact 2 corroborates it independently: the sub-interval sweep finds an
   *interior* minimum inside a ±6-minute window, which cannot happen unless the
   hour offset is already right. Alice Springs sits in `Australia/Darwin`,
   permanently +9:30, so this is the site's own zone rather than a stand-in.

   One methodological note, because it nearly went the other way. An earlier
   draft of this check rebuilt its "clear sample" mask from each candidate
   offset's own clear-sky series. That lets every offset select the samples that
   suit it, and it ranked +9:00 first. The mask is now built from measurements
   alone and shared across all four candidates.

2. **The stamp labels the *centre* of the 5-minute averaging window.** The same
   fit, restricted to the 33,742 low-sun clear samples where a timing error
   produces the largest signal, minimises at **−0.25 min**. Interval-start would
   minimise at +2.5 and interval-end at −2.5, and both sit on the wrong side of
   a smooth minimum (63.8 W/m² at the centre against 64.4 and 64.7 at ∓2.5 min).
   The margin is only about 1 % of RMSE, so this is a modest effect at
   5-minute resolution rather than a dramatic one. Solar position is evaluated
   *at* the stamp, which is BREOS's default `interval-start` setting. The NIST
   follow-up treats its one-minute stamp as the interval end. NIST does not
   document that label basis as clearly as DKASC documents its convention, so
   the one-minute NIST ambiguity remains recorded rather than presented as a
   verified source fact.

3. **The tilted pyranometer is co-planar with the fixed arrays.** Fitting its
   orientation from the data alone gives **21° / azimuth 359°**, against DKASC's
   published 20° / azimuth 0. The published geometry is what the validation
   uses. The fit is only a check on it, and on the hemisphere convention: a
   sign error would have put the fit near 180°, not near 0°.

4. **Channels are already in engineering units.** Unlike the PVDAQ copy used
   in the first NIST run, there is no millivolt channel to convert. The raw
   columns do carry out-of-range spikes and sentinel values. Across the
   2008–2025 record, GHI reaches −986 and +2726 W/m², wind reaches −1742 m/s,
   and relative humidity exceeds 100 % on 2,198 samples in 2016. The build
   screens these values rather than modelling them.

5. **Two independent metering channels agree, until they don't.** Integrated
   `Active_Power` and the differenced cumulative `Active_Energy` counter agree
   within **0.3–1.0 % for every array-year from 2009 to 2020**, then diverge
   sharply from 2021 (to 0.83 by 2024) as the 5-minute record starts dropping
   samples the counter still accrues. That divergence is what puts 2021 onward
   out of bounds; it is not visible in either channel alone.

6. **The site master meter is unusable, and the obvious channel is again the
   wrong one.** Source 96 (`MasterMeter1`) fails the cross-check outright, with
   power-to-counter ratios of −0.55 to −0.82 in six separate years: it is a net
   meter whose counter runs backwards. Its 5-minute power channel is also 52 %
   absent in 2016. The per-array meters are used throughout and the master meter
   is never used as a generation reference. This is the NIST lesson repeating in
   a different costume, and the cross-check is what catches it.

7. **Wind speed ends permanently in late October 2016.** It is interpolated over
   short gaps and median-filled beyond them. It is common to both arrays of the
   pair in any case, so it cannot affect the ratio.

8. **One supplied source-ID mapping was wrong.** Source 214 is confirmed as site
   32 (Canadian Solar 5.3 kW poly, 2016). Source 59 is not site 38: site 38
   (Q CELLS, 2017) sits on M19 *B*-phase, while 59 is M19 *C*-phase. The data
   corroborates this independently. Source 59 reads exactly zero until 2018, so
   it cannot be a 2017 commissioning either. Neither source is used in the
   analysis.

### Screening

Two failure modes exist and they pull the bias in opposite directions, so
they are screened separately rather than with one rule.

- **Weather-station outage.** The pyranometers read identically zero for whole
  days while the arrays keep producing normally. 2016 contains one contiguous
  **18-day block, 18 June to 5 July**. Any irradiance-driven model predicts zero
  on those days, so leaving them in *understates* the bias. For 16A in 2016,
  screening moved the answer from +6.98 % to +10.78 % and daily correlation from
  0.866 to 0.993. The array screen accounts for one day of that; the rest is the
  weather outage. This is the mirror image of NIST, where the *array* was down and the
  model over-predicted.
- **Array outage or curtailment.** 2016-01-30 is depressed for every array on
  the site at once, so it reads as a site-wide event, not an array fault.

The array screen normalises by each array's own median measured/model ratio
before thresholding. An absolute threshold would cut the low tail of a
distribution that the model's own bias has already shifted. At this site that
would have quietly discarded 12 ordinary days for 16A and flattered the result.
Over 2009–2014 the screen removes 11 of 2191 days on the weather side and 8–23
days per array.

---

## 3. The 16A fault, and the window the pair test runs in

**16A stopped being 16D's twin in 2015.** This is the single most important
thing to know before reading any transposition number from this site.

Measured quarterly energy ratios, with no model anywhere in them. Arrays 12 and
13 are independent north-facing arrays of different manufacture, scaled by DC
rating.

| period | 12/16A | 13/16A |
|---|---|---|
| 2009–2014 | 0.96–1.03 | 0.96–1.11 |
| 2015Q4 | 1.062 | 1.099 |
| 2017–2020 | 1.07–1.13 | 1.06–1.19 |

The modelled loss-free bias says the same thing, and quantifies it as an ageing
rate:

| array | implied degradation |
|---|---|
| 12 (BP mono, north) | 0.61 %/yr |
| **16D (BP poly, west)** | **0.64 %/yr** |
| 13 (Trina mono, north) | 1.23 %/yr |
| **16A (BP poly, north)** | **1.77 %/yr** |

16A ages nearly three times faster than the physically identical array standing
next to it. Run the pair test through that fault and it reports a 5–9 %
transposition failure that does not exist. **The pair test is therefore run over
2009–2014**, and that window is chosen from the model-free table above, so it
cannot have been selected to favour any transposition model.

---

## 4. Results

### Leg A: modelled POA against the co-planar pyranometer

2016, 348 usable days, 46,707 daytime samples. No PV model involved. Measured
2232.8 kWh/m²; every model gives r = 0.993.

| transposition | bias | RMSE |
|---|---|---|
| klucher | **+0.04 %** | 44.3 W/m² |
| perez-driesse | −0.47 % | 43.3 W/m² |
| perez | −0.48 % | 43.3 W/m² |
| king | −0.56 % | 44.4 W/m² |
| reindl | −1.02 % | 43.6 W/m² |
| haydavies | −1.06 % | 43.7 W/m² |
| **isotropic** *(BREOS default)* | **−1.76 %** | 44.8 W/m² |

Read the RMSE column alongside the bias. Klucher's +0.04 % is a cancellation,
not accuracy: its RMSE is the second worst in the table, so it overestimates the
sky-diffuse term and happens to offset a general shortfall. Perez has both the
lowest bias among the physically-motivated models and the lowest RMSE. The span
from BREOS's `isotropic` default to Perez is **1.3 percentage points** on a
sun-facing 20° plane.

Perez coefficient sets span −0.34 % (`albany1988`) to −1.16 % (`phoenix1988`),
except `osage1988` at −3.18 % with a visibly worse RMSE (54.7 W/m²). That is the
one set that should not be used here. Albedo 0.20 to 0.30 moves every model by
the same +0.28 pp, because the ground-reflected term is identical at a common
tilt.

### Leg B: the controlled north/west pair

2009–2014, 2146 normal days. Measured west/north energy ratio **0.8724**.
Loss case `no-availability-no-shading` (8.75 % stack).

| transposition | 16A north | 16D west | spread | **W/N ratio error** |
|---|---|---|---|---|
| haydavies | +1.90 % | +2.19 % | 0.30 pp | **+0.29 %** |
| reindl | +1.93 % | +2.23 % | 0.30 pp | **+0.30 %** |
| perez-driesse | +2.49 % | +3.07 % | 0.58 pp | +0.56 % |
| perez | +2.46 % | +3.07 % | 0.61 pp | +0.60 % |
| klucher | +2.95 % | +4.28 % | 1.32 pp | +1.29 % |
| **isotropic** *(default)* | +1.11 % | +2.42 % | 1.30 pp | +1.29 % |
| king | +2.31 % | +3.83 % | 1.52 pp | +1.49 % |

**Every model gets both orientations right to within 1.5 percentage points.**

### Measurement floor

The year-to-year scatter of the Hay-Davies ratio error across the six window
years is **0.86 pp** (2010 excluded: its *measured* W/N of 0.852 is itself an
outlier against 0.872–0.884 in every other year). Including 2010 the scatter is
2.15 pp.

So the ~1 pp gap separating {haydavies, reindl} from {isotropic, klucher, king}
is at the edge of what this site can resolve, and **the 0.3 pp gap between
Hay-Davies and Perez is not resolvable at all.** DKASC has no redundant
co-located irradiance sensor, so unlike NIST the floor cannot be measured by
differencing two pyranometers; DKASC's own stated instrument accuracy is ±8 % or
±10 W/m² per reading.

### Decomposition costs more than transposition

DNI from the measured GHI/DHI closure is a geometric identity, not a model. The
alternatives estimate it from GHI alone, which is the ordinary case, and at
Esposende the actual case.

| DNI source | isotropic | haydavies | perez | RMSE (perez) |
|---|---|---|---|---|
| **closure** (GHI + DHI measured) | −1.76 % | −1.06 % | −0.48 % | **43.3 W/m²** |
| dirint | −3.50 % | −2.75 % | −2.19 % | 72.8 W/m² |
| disc | −4.22 % | −3.45 % | −2.90 % | 85.6 W/m² |
| erbs | −5.71 % | −4.95 % | −4.40 % | 95.4 W/m² |

**Losing the measured diffuse component costs 1.7–3.9 pp of annual POA and
doubles to triples the 5-minute RMSE, two to three times more than the entire
choice of transposition model.** Model choice is the smaller lever at this site;
the quality of the irradiance input is the larger one.

### Loss ladder, 2009 to 2014, Hay-Davies, DNI from the measured closure

Annual AC bias on screened days, ~2160 days per array.

| array | `breos-default` 14.1 % | `no-availability` 11.5 % | `no-avail-no-shading` 8.75 % | `module-only` 0 % | daily r |
|---|---|---|---|---|---|
| 16A north | −4.17 % | −1.17 % | +1.92 % | **+11.69 %** | 0.992 |
| 16D west | −3.91 % | −0.89 % | +2.22 % | **+12.05 %** | 0.987 |
| 12 north | −2.73 % | +0.31 % | +3.44 % | **+13.34 %** | 0.993 |
| 13 north | −7.47 % | −4.58 % | −1.61 % | **+7.80 %** | 0.986 |

Daily correlation is 0.986–0.993 at every rung: as at NIST, the loss assumption
moves the level and never the shape.

### Effective system loss

The stack that would zero each array's loss-free bias:

| array | effective system loss |
|---|---|
| 13 (Trina mono) | 7.2 % |
| 16A (BP poly, north) | 10.5 % |
| 16D (BP poly, west) | 10.8 % |
| 12 (BP mono, north) | 11.8 % |
| *NIST Ground 2016* | *5.35 %* |
| *BREOS `DEFAULT_PVWATTS_LOSSES`* | *14.1 %* |

DKASC's arrays sit between NIST's exceptionally well-maintained research
array and the BREOS default, and closer to the default than NIST was. That is
the expected direction for a desert site: dust loading is real and persistent in
a way it is not in Maryland. The 14.1 % default still overstates every array
here, but by 2–7 points rather than NIST's 8.7.

The spread across arrays (7.2 % to 11.8 %) is wider than the physics warrants
and mostly reflects datasheet rather than CEC module parameters. Site 13's
7.2 % is as likely to be an optimistic Trina nameplate as a genuinely cleaner
array.

### Residual structure, flat again

| array | clear (k<sub>t</sub>>0.85) | mixed | overcast |
|---|---|---|---|
| 16A | +1.76 % | +2.68 % | +3.28 % |
| 16D | +1.91 % | +4.09 % | +1.94 % |
| 12 | +3.79 % | +2.49 % | −3.00 % |
| 13 | −0.98 % | −3.43 % | −11.40 % |

**No clear-day concentration.** That is the same result as NIST and the
opposite of Esposende, where the residual lives on clear days. Alice Springs is overwhelmingly
clear (1632 of ~2160 days), so the clear-sky column is where the statistical
weight is, and it is where the bias is smallest for three of the four arrays.

Monthly bias carries a mild seasonal signature on the west array that the north
arrays do not share: negative in autumn and winter, positive in spring and
summer, roughly ±5 pp about its mean in 2009. That is the residue of the off-axis
geometry, and it is the one place a better sky model would visibly help.

### The tracking path, the one large failure found

Site 1A, two DEGERenergie 5000NT dual-axis trackers, 60 Trina modules, 10.5 kW,
over 2014–2016 on ~1025 screened days:

| transposition | bias | daily r |
|---|---|---|
| isotropic | **+23.0 %** | 0.979 |
| haydavies | **+27.5 %** | 0.977 |
| perez | **+28.7 %** | 0.978 |

Sweeping `dual_axis_max_tilt` from 90° to 60° changes this by less than 0.5 pp,
so a mechanical elevation limit is not the explanation. Daily correlation stays
at 0.977–0.979: the shape is right and the level is badly wrong.

Two measurements locate the cause. First, the array's measured yield is only
**+12.9 %** above the co-located fixed 20° north array, 2003 against 1774
kWh/kWp/yr. BREOS predicts the fixed arrays to within a few per cent over the
same period, so a 23 to 29 % error on the tracker is specific to the tracking
path rather than shared with the rest of the chain. Second, the over-prediction
is a clean function of solar elevation:

| sun elevation | model / measured | share of measured energy |
|---|---|---|
| 60–90° | 1.12 | 26.5 % |
| 45–60° | 1.19 | 26.1 % |
| 30–45° | 1.29 | 27.1 % |
| 20–30° | 1.51 | 11.4 % |
| 10–20° | **1.71** | 7.1 % |
| 0–10° | 1.45 | 1.8 % |

That is the signature of **inter-tracker shading**: two trackers side by side,
each held normal to the sun, shade one another whenever the sun is low, and the
error grows exactly as the sun drops.

`breos/solar.py` confirms it directly. The dual-axis branch sets
`surface_tilt = clip(zenith, 0, dual_axis_max_tilt)` and
`surface_azimuth = sun_azimuth` and stops there. **`gcr`, `backtrack` and
`cross_axis_tilt` are consumed only by `pvlib.tracking.singleaxis` in the
single-axis branch.** BREOS's dual-axis model is an isolated tracker that is
never shaded by anything, and at a real two-tracker installation that is worth
23–29 % of annual energy.

This is far larger than any error in the fixed-tilt chain, and it is the one
result here that is a defect rather than a bound. It is scoped to dual-axis:
single-axis arrays do get backtracking and `gcr`.

---

## 5. What the transposition test concluded

**BREOS's fixed-tilt transposition stage is validated within the bounds of
these two tests. It does not explain the Esposende residual.**

- Against a co-planar reference instrument, every model lands within 1.8 % of
  measured annual POA, and the span from BREOS's `isotropic` default to Perez,
  the best of the physically-motivated models, is **1.3 percentage points**.
- Against the controlled north/west pair, every model reproduces the measured
  west/north energy ratio to within **1.5 percentage points**, over six years
  and 2146 screened days.
- The two legs do not crown the same winner. Perez leads Leg A (−0.48 % against
  Hay-Davies's −1.06 %); Hay-Davies and Reindl lead Leg B (+0.29 %, +0.30 %,
  against Perez's +0.60 %). What both legs agree on is that **isotropic is last
  or near-last in each**, and that Hay-Davies and Perez cannot be separated from
  one another.
- **A model was not chosen after seeing the answer.** The array geometry is
  DKASC's published 20°/0° and 20°/270° throughout, and the 2009–2014 window
  comes from a model-free measured-ratio table.

**Where it is inconclusive, and it partly is.** The 0.86 pp year-to-year scatter
means this site cannot separate Hay-Davies from Perez, and only barely separates
either from isotropic. The honest statement is a bound, not a ranking:
**transposition-model choice is worth about 1 to 1.5 percentage points of annual
energy at 20° tilt in a clear-sky desert climate, and no model tested is wrong
by more than that.**

That bound does not transfer unchanged to Esposende. DKASC's diffuse fraction is
24 % (511 of 2130 kWh/m² in 2016) at 20° tilt. That is the least discriminating
combination there is, because the sky-diffuse term the models disagree about is
both small and close to isotropic under a clear desert sky. A cloudier site at a
steeper tilt would show a wider spread, and DKASC cannot measure how much wider.
What DKASC does establish is that **the models are not broken**, including on a
90°-off-axis plane where they are under most stress.

### What to change

Two things follow, one small and one not.

**Move the default off `isotropic`, to either `haydavies` or `perez`.** Both
beat isotropic in both legs and neither needs any extra input, since Hay-Davies
and Perez consume the same GHI/DNI/DHI that isotropic does. Hay-Davies gains
0.7 pp on measured POA and 1.0 pp on the north/west ratio; Perez gains 1.3 pp
and 0.7 pp. Choosing between those two is beyond what this site can resolve, so
the case for moving is that isotropic is consistently the worst of the three and
the alternatives are free, not that DKASC picked a winner. Avoid the `osage1988`
Perez coefficient set, the one clear outlier at 3.2 % low with a visibly worse
RMSE.

**Give the dual-axis path a self-shading term, or document that it has none.**
This is the real defect found here, and 23 to 29 % dwarfs everything else in
this document. Single-axis arrays already get `backtrack` and `gcr`; dual-axis
silently ignores both.

**The larger lever is the irradiance input, not the model.** Replacing measured
DHI with a decomposition model costs 1.7–3.9 pp of annual POA and doubles to
triples the sub-hourly RMSE, two to three times the entire transposition
question.

---

## 6. Cross-read against NIST and Esposende

All three on a common footing, since all used BREOS's 14.1 % default stack.

| Site | As configured | Loss-free equivalent | Driven from |
|---|---|---|---|
| Esposende 2024 | +8.57 % | **+26.4 %** | GHI, 30 km away |
| NIST Ground 2016 | −9.26 % | **+5.65 %** | measured POA, on-site |
| DKASC 13 north | −7.47 % | **+7.80 %** | on-site GHI+DHI, full chain |
| DKASC 16A north | −4.17 % | **+11.69 %** | on-site GHI+DHI, full chain |
| DKASC 16D west | −3.91 % | **+12.05 %** | on-site GHI+DHI, full chain |
| DKASC 12 north | −2.73 % | **+13.34 %** | on-site GHI+DHI, full chain |

DKASC's loss-free biases (+7.8 % to +13.3 %) sit above NIST's +5.65 %, which is
what six-to-fourteen-year-old modules on datasheet parameters at a dusty desert
site should do. **Esposende's +26.4 % still sits roughly 13 to 20 points above
any fully documented array**, now including one where the entire GHI → POA path
has been exercised end to end.

**This closes the fixed-tilt transposition hypothesis.** The first NIST run did
not test transposition because its PVDAQ input lacked calibrated GHI. The later
NIST bulk-archive follow-up found a 2.9 pp isotropic-to-Perez spread against a
co-planar silicon sensor, while the two DKASC legs bound the difference at
about 1 to 1.5 pp against thermopile POA and the measured orientation ratio.
The site-specific spreads differ, but neither approaches the roughly 20 pp
loss-free Esposende residual.

---

## 7. Limits

- **Module parameters are from datasheets, not the CEC database.** BP Solar is
  defunct and the Trina DC01 series is delisted. Each is cross-checked against
  DKASC's published array area to better than 0.2 pp of efficiency, but the
  7.2 %–11.8 % spread in implied system loss is more likely parameter
  uncertainty than four genuinely different arrays.
- **One site, one climate, one tilt.** 20°, clear desert, 24 % diffuse fraction.
  This is the regime in which transposition models agree most, so the 1.3 pp
  bound is a best case, not a general one.
- **The pair test rests on six years.** 2015 onward is unusable for it because
  16A developed a fault, and 2021 onward is unusable for anything because the
  5-minute power record starts dropping samples.
- **2010 is an outlier on the measurement side.** Its measured W/N is 0.852
  against 0.872–0.884 everywhere else. It is retained in the pooled figures and
  reported separately rather than dropped.
- **No redundant irradiance sensor.** Unlike NIST, the measurement floor cannot
  be established by differencing two co-located instruments; it is inferred from
  year-to-year scatter (0.86 pp) and DKASC's stated ±8 % / ±10 W/m².
- **Wind is absent after October 2016** and median-filled. It is common-mode
  across the pair, so it cannot affect the ratio, but it does affect the
  absolute biases in later years.
- **Soiling is not separated from degradation.** The 0.61–1.77 %/yr rates are
  net of whatever DKASC's cleaning regime does, which is not published here.
- **The tracker result is one installation.** The +23–29 % is measured against a
  two-tracker site whose row spacing is not published; it bounds the cost of
  having no self-shading term, but it is not a general correction factor.
- **Array 13 shows a winter shortfall** (39 abnormal days in 2016, concentrated
  May–July, and −11.4 % in overcast conditions) that this run did not diagnose.
  Row-to-row shading at low winter sun elevation is the obvious candidate.

---

## 8. Reproducing

```bash
cd /tmp/breos-external-validation
export UV_CACHE_DIR=/tmp/breos-ext-uv

uv run --offline python tools/validation/dkasc_facts.py --raw-dir ~/Downloads

PYTHONPATH=tools/validation uv run --offline python tools/validation/dkasc_build.py \
    --raw-dir ~/Downloads --out dkasc_2016.csv.gz \
    --start-year 2016 --end-year 2016 --sources 100 81 84 92 91 96
PYTHONPATH=tools/validation uv run --offline python tools/validation/dkasc_build.py \
    --raw-dir ~/Downloads --out dkasc_2009_2020.csv.gz \
    --start-year 2009 --end-year 2020 --sources 100 81 84 92 91

PYTHONPATH=tools/validation uv run --offline python tools/validation/dkasc_transposition.py \
    --gti-data dkasc_2016.csv.gz --pair-data dkasc_2009_2020.csv.gz --outdir results/
PYTHONPATH=tools/validation uv run --offline python tools/validation/dkasc_analysis.py \
    --data dkasc_2009_2020.csv.gz --outdir results/ \
    --ladder-years 2009:2014 --tracker-years 2014:2016
```

Built artefacts live outside the repo, in the session scratchpad:
`dkasc_2016.csv.gz` (9 MB), `dkasc_2009_2020.csv.gz` (87 MB), and under
`results/`: `transposition_leg_a.csv`, `transposition_leg_b.csv`,
`transposition_pair_drift.csv`, `transposition_pair_drift_measured.csv`,
`transposition_decomposition.csv`, `loss_ladder.csv`, `effective_loss.csv`,
`degradation_by_year.csv`, `degradation_fit.csv`, `regimes.csv`, `monthly.csv`,
`tracker.csv`.

**Note:** the project's dependency set has no parquet engine, so consolidated
frames are written as gzipped CSV.
