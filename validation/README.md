# BREOS validation suite

A standing benchmark that pins BREOS's PV chain against independent
references at seven sites on four continents, and turns any unintended model
drift into a CI failure. This is repo-side tooling — nothing in here ships in
the package.

## Layout

| Path | What it is | Committed? |
|---|---|---|
| `locations.json` | The validation sites + one shared 4 kWp system spec | yes |
| `data/weather/*.csv.gz` | PVGIS TMY hourly weather per site — the input BREOS runs on | yes (gitignore exception) |
| `data/references/*.json` | External reference results + every request parameter | yes |
| `baselines/breos_baseline.json` | BREOS's own outputs, snapshotted — the drift-test anchor | yes |
| `results/` | Freshly generated outputs | no (gitignored) |
| `REPORT.md` | Generated comparison report snapshot | yes |

## Workflow

```bash
uv run python validation/fetch_references.py         # fetch weather + references (network)
uv run python validation/run_breos.py                # run BREOS on the checked-in weather
uv run python validation/compare.py                  # table + REPORT.md
uv run pytest tests/test_validation_drift.py         # what CI runs
```

`run_breos.py --write-baseline` regenerates the drift-test anchor. Do that
**only** for intentional model changes, and note it in the changelog — the
whole point of the baseline is that it does not move silently.

## Reference sources

- **PVGIS v5.3 PVcalc** (`re.jrc.ec.europa.eu/api/v5_3/PVcalc`) — the EU
  JRC's own PV model (Huld et al. 2010) on the full multi-year SARAH3/ERA5
  satellite record. No API key. Worldwide coverage; this is the reference
  that works for every site. The TMY weather BREOS runs on comes from the
  same database, so this is the closest thing to a same-weather comparison
  among the public calculators.
- **NREL PVWatts v8** (`developer.nrel.gov/api/pvwatts/v8.json`) — the
  NREL/SAM reference implementation. `NREL_API_KEY` env var (free at
  https://developer.nrel.gov/signup/), falls back to `DEMO_KEY`. Coverage:
  NSRDB for the Americas and parts of Asia; station-based `intl` data
  elsewhere (the fetcher records station distance and marks references
  \>200 km as untrusted, which excludes them from CI assertions).
- **PVGIS TMY** (via `breos.weather.fetch_tmy_weather_data`) — the checked-in
  hourly weather inputs, trimmed to the five columns BREOS reads
  (`ghi`/`dni`/`dhi`/`temp_air`/`wind_speed`), rounded to physical precision
  and gzipped (~90 KB per site). The committed file is the deterministic
  input the drift test recomputes from — never regenerate it casually; a new
  fetch means a new baseline. If the suite outgrows in-repo data (more sites,
  sub-hourly, measured series), move it to a release asset or data repo with
  pinned checksums.

Deviations against these references bundle **weather-source differences with
model differences** (TMY vs multi-year record vs NSRDB stations). That is why
the CI bands are loose (±10% PVGIS, ±15% PVWatts) — they catch gross errors.
The *tight* guarantee is the self-baseline regression (0.1%), which catches
any unintended change to transposition, IAM, thermal, DC, loss, or inverter
modeling the moment it lands.

## What the numbers mean

BREOS is run twice per site: with its shipped default transposition
(`isotropic`, systematically low on tilted planes) and with `perez` (what
PVGIS and PVWatts effectively use). Comparing both against PVGIS quantifies
the isotropic penalty per climate — evidence for the "recommended model
profile" roadmap item.

Alignment choices, so the three models see the same system: BREOS's default
PVWatts loss stack (~14.1%) is passed to PVWatts as `losses` with
`inv_eff=96`; PVGIS gets the combined `1 − (1−losses)·η_inv` (~17.5%) since
its single loss figure includes the inverter. Free-standing mount everywhere
(BREOS Faiman default ↔ PVWatts `array_type=0` ↔ PVGIS `mountingplace=free`),
albedo 0.2, same tilt/azimuth per site.

## Extending

- **Add a site**: add it to `locations.json`, run the three commands above,
  commit weather + reference + baseline + report.
- **Measured-data track (future)**: NREL PVDAQ (US systems,
  https://data.openei.org/submissions/4568) and DKA Solar Centre Alice
  Springs (https://dkasolarcentre.com.au) provide free measured PV time
  series for validating against reality rather than other models.
- **Same-weather oracle (future)**: run SAM/PySAM offline on the checked-in
  TMY CSVs to remove the weather-source confound entirely; store its outputs
  as another reference block.

## Data licensing

PVGIS data © European Union, 2001–2024, reuse permitted with attribution
(CC BY 4.0). PVWatts results courtesy of the NREL developer API under its
[terms of service](https://developer.nrel.gov/docs/terms/).
