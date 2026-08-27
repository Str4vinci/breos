# BREOS

BREOS is an open source simulator for PV and storage systems for buildings. Built with a modular architecture, it invites users to "bring-your-own" data and simulate solar systems.

## Project map

Keep changes scoped and preserve the `breos.App` facade as the most stable public entrypoint.

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

## Change guidance

- Prefer extending `breos.App` through backwards-compatible config keys.
- Avoid repo-relative runtime paths; packaged resources should be loaded through `breos.resources`.
- Keep generated files out of git unless they are intentional release assets.
- Add focused tests for public API behavior when touching defaults, packaged data, or serialization.

## Pull requests

- Use a specific, descriptive branch name with a conventional change-type prefix such as `feat/`, `fix/`, `refactor/`, or `docs/`. Do not use generic branch names such as `agents` or `codex`.
- Start every PR description with a plain, human-readable paragraph that explains the purpose of the change.
- Before drafting a PR description, review relevant previously closed PRs and follow the repository's established tone and structure.
