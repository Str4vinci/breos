# Run the Article 1 simulations

`tools/run_article1.py` runs the complete Article 1 workflow. It uses local
inputs from `dev/article1-inputs/` and writes results to `results/article1/`.
Both directories are ignored by Git.

## Prepare the environment

Run all commands from the BREOS repository root at the final release commit:

```bash
uv sync --extra dev --extra docs --frozen
uv run python tools/run_article1.py check
```

The `check` stage verifies every external input and records its SHA-256 digest.
It does not run a simulation. The runner stops if the active BREOS version is
not 0.6.0 or the tracked worktree is dirty.

## Run the deterministic workflow

Run the fixed candidates, all projected optimizations, and the context
analyses:

```bash
uv run python tools/run_article1.py
```

To inspect C1-C5 before starting the longer optimizations, split the work into
two commands:

```bash
uv run python tools/run_article1.py fixed
uv run python tools/run_article1.py analysis
```

The deterministic workflow produces:

- the C1-C5 fixed-design tables at EUR 500/kWh;
- the EUR 350, 500, and 711/kWh projected Pareto fronts;
- the hourly-resolution optimization;
- the H0-load C2 comparison;
- the field-v2 and laboratory degradation optimizations and C2 replays;
- the orientation table; and
- the 2005-2023 historical-weather comparison.

## Run Monte Carlo

Run the configured 10,000 trajectories for C2 first, then C1, C3, C4, and C5:

```bash
uv run python tools/run_article1.py monte-carlo
```

Use `--mc-runs` only for a trial run. A trial bundle does not pass final
verification:

```bash
uv run python tools/run_article1.py monte-carlo --mc-runs 25
```

The Monte Carlo configuration uses the EUR 500/kWh base case. The EUR 350 and
711/kWh values apply only to the deterministic Pareto sensitivity.

## Verify the result bundle

After every full run completes, verify the provenance and artifact hashes:

```bash
uv run python tools/run_article1.py verify
```

To run every stage and verify the result in one command, use:

```bash
uv run python tools/run_article1.py all
```

The complete result bundle is in `results/article1/`. Keep that directory with
the manuscript source data. Do not commit the licensed E-REDES profile or the
third-party validation data.

## Local input layout

The default input directory has this layout:

```text
dev/article1-inputs/
├── rlp/
│   ├── EREDES_2025_BTN_1000kwh_15min.csv
│   └── EREDES_2025_BTN_1000kwh_hourly.csv
├── validation/
│   ├── daily_results.csv
│   ├── monthly_results.csv
│   └── weekly_results.csv
└── weather/
    └── porto_historical_2005_2024_openmeteo.csv
```

If the input bundle is elsewhere, pass `--input-root`. To change the result
directory, pass `--output`.

The historical weather file contains 2005-2024. The Article configuration
samples 2005-2023 to match the archived workflow and the TMY source period.
Update Section 2.7 of the manuscript, which currently says that Monte Carlo
also samples 2024.
