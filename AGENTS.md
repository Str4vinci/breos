# Agent Guide

This repository is the open-source core of BREOS. Keep changes scoped and preserve the `breos.App` facade as the most stable public entrypoint.

## Project map

- `breos/app.py` - public facade that wires weather, PV, load, battery, economics, and emissions.
- `breos/load_profiles.py` - bundled demandlib H0 profile support plus user-supplied external RLPs.
- `breos/battery.py`, `breos/solar.py`, `breos/weather.py` - core simulation models.
- `breos/economics.py`, `breos/emissions.py`, `breos/optimization.py` - analysis and sizing helpers.
- `breos/data/` - packaged presets and redistributable sample data used after installation.
- `configs/` - editable example and template configuration files for users.
- `tests/` - pytest coverage for public behavior and lower-level modules.
- `docs/` - Sphinx/MyST source; generated output lives in `docs/_build/` and is ignored.

## Common commands

```bash
uv sync --extra dev --extra docs
uv run pytest -q
uv run ruff check breos tests
uv run ruff format --check breos tests
uv build
```

## Release data policy

Only redistribute profiles whose upstream terms are clear enough for a public package. BREOS currently bundles demandlib-derived H0 examples. E-REDES, REE, and direct BDEW downloads remain supported only as user-supplied local files via `rlp_directory`.

## Change guidance

- Prefer extending `breos.App` through backwards-compatible config keys.
- Avoid repo-relative runtime paths; packaged resources should be loaded through `breos.resources`.
- Keep generated files out of git unless they are intentional release assets.
- Add focused tests for public API behavior when touching defaults, packaged data, or serialization.
