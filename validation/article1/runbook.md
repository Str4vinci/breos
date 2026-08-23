# Generate the Article 1 source data

This runbook generates the BREOS source tables for the Article 1 manuscript.
The commands write generated files under the ignored `results/` directory.
They do not write to the research repository.

## Set the external input paths

Set these paths before you start:

```bash
export BREOS_A1_RLP_DIR=/path/to/licensed/rlp
export BREOS_A1_HISTORICAL_WEATHER=/path/to/porto_historical_2005_2024_openmeteo.csv
```

`BREOS_A1_RLP_DIR` must contain
`EREDES_2025_BTN_1000kwh_15min.csv`. BREOS records the SHA-256 digest of both
external inputs. Do not commit the licensed E-REDES file.

Run all commands from the BREOS repository root at the final release commit.
Commit all tracked changes before the scientific runs so each provenance file
records `tracked_worktree_dirty` as `false`.

## Generate the fixed base cases

Generate the C1-C5 base-case tables at €500/kWh:

```bash
uv run python tools/reproduce_article1.py \
  --rlp-directory "$BREOS_A1_RLP_DIR" \
  --output results/article1/base-v1
```

This command writes `fixed_candidates.csv`, a `reproduction.json` file, and
the yearly, financial, and metric files for each candidate.

## Generate the battery-cost fronts

Run the €350, €500, and €711 per kWh projected optimizations:

```bash
uv run python tools/reproduce_article1.py \
  --rlp-directory "$BREOS_A1_RLP_DIR" \
  --battery-cost 350 \
  --battery-cost 500 \
  --battery-cost 711 \
  --skip-fixed \
  --full-optimization \
  --n-procs 8 \
  --output results/article1/battery-cost-sensitivity
```

The command creates one `battery-cost-*` directory per scenario. Each
directory contains the Pareto front and replayed maximum-NPV, knee, and
maximum-GI designs. Replacement costs use the selected capacity cost. The
fixed €350 installation charge applies only to the initial installation.

## Generate the timestep and load-profile sensitivities

Run the hourly projected optimization:

```bash
uv run python tools/reproduce_article1.py \
  --rlp-directory "$BREOS_A1_RLP_DIR" \
  --resolution h \
  --skip-fixed \
  --full-optimization \
  --n-procs 8 \
  --output results/article1/hourly-v1
```

Evaluate C2 with the packaged H0 profile:

```bash
uv run python tools/reproduce_article1.py \
  --rlp-directory "$BREOS_A1_RLP_DIR" \
  --load-profile h0 \
  --candidate C2 \
  --output results/article1/load-profile-h0
```

Compare the H0 result with C2 under `results/article1/base-v1`.

## Generate the degradation sensitivities

Run the field-v2 optimization and C2 replay:

```bash
uv run python tools/reproduce_article1.py \
  --rlp-directory "$BREOS_A1_RLP_DIR" \
  --calendar-model naumann_lam_field_calibrated_v2 \
  --candidate C2 \
  --full-optimization \
  --n-procs 8 \
  --output results/article1/field-v2
```

Run the laboratory-parameter optimization and C2 replay:

```bash
uv run python tools/reproduce_article1.py \
  --rlp-directory "$BREOS_A1_RLP_DIR" \
  --calendar-model naumann_lam \
  --candidate C2 \
  --full-optimization \
  --n-procs 8 \
  --output results/article1/laboratory
```

Use the three C2 `yearly_summary.csv` files from the base, field-v2, and
laboratory runs for the degradation comparison.

## Generate the orientation source table

Generate the tilt-azimuth grid and continuous optimum for Figure 3:

```bash
uv run python tools/reproduce_article1_context.py \
  --output results/article1/orientation \
  orientation
```

The command evaluates one module over the manuscript's 10-90° tilt and
100-260° azimuth bounds. It writes both module production and area-normalized
production.

## Generate the weather-comparison source table

Generate the 2005-2023 monthly source data for Figure 6:

```bash
uv run python tools/reproduce_article1_context.py \
  --output results/article1/weather-comparison \
  weather-comparison \
  --historical-weather-file "$BREOS_A1_HISTORICAL_WEATHER"
```

The command writes the per-year monthly values and the TMY-versus-historical
summary, including the mean, standard deviation, 95% confidence interval,
minimum, and maximum.

## Generate the Monte Carlo source data

Run C2 first:

```bash
uv run python tools/reproduce_article1_montecarlo.py \
  --rlp-directory "$BREOS_A1_RLP_DIR" \
  --weather-file "$BREOS_A1_HISTORICAL_WEATHER" \
  --case C2 \
  --n-procs 8 \
  --output results/article1/monte-carlo-v1
```

After you inspect C2, run the remaining cases without repeating C2:

```bash
uv run python tools/reproduce_article1_montecarlo.py \
  --rlp-directory "$BREOS_A1_RLP_DIR" \
  --weather-file "$BREOS_A1_HISTORICAL_WEATHER" \
  --case C1 \
  --case C3 \
  --case C4 \
  --case C5 \
  --n-procs 8 \
  --output results/article1/monte-carlo-v1
```

Omit `--runs` to use the configured 10,000 trajectories. Each case writes
`runs.csv`, `yearly.csv`, `summary.json`, and `provenance.json`.

## Preserve the final bundle

Keep the complete `results/article1` directory with the submitted manuscript.
Archive that directory in the manuscript data deposit. Commit only the BREOS
code, configurations, tests, and documentation.
