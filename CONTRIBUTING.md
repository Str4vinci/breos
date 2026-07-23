# Contributing to BREOS

Thanks for your interest in contributing to BREOS! This guide will help you get set up.

## Development Setup

1. Clone the repository:

```bash
git clone https://github.com/Str4vinci/breos.git
cd breos
```

2. Install [uv](https://docs.astral.sh/uv/) (recommended) or use pip:

```bash
# With uv (recommended)
uv sync --extra dev --extra docs

# With pip
pip install -e ".[dev]"
```

3. Run the tests to make sure everything works:

```bash
uv run pytest tests/ -v
```

For a fast end-to-end smoke test without writing code:

```bash
uv run breos validate-config configs/examples/quickstart.toml
uv run breos run --config configs/examples/quickstart.toml --dry-run
```

For release-style validation, run the same gates used by CI:

```bash
uv run ruff check breos/ tests/ tools/
uv run ruff format --check breos/ tests/ tools/
uv run pytest tests/ -v --cov=breos --cov-report=term-missing
uv run pytest tests/ -m slow -v
uv run python tools/verify_release_artifacts.py
uv run --extra docs sphinx-build -W -b html docs docs/_build/html
```

## Branching

- `main` — stable, release-ready code. Do not push directly.
- `develop` — integration branch for ongoing work.
- Feature branches — create off `develop` for your work.

```bash
git checkout develop
git checkout -b feature/your-feature-name
```

## Making Changes

1. Create a feature branch off `develop`
2. Write your code
3. Add or update tests as needed
4. Make sure all tests pass: `uv run pytest tests/ -v`
5. Commit with a clear message describing what and why
6. Push and open a Pull Request against `develop`

## Tests

All tests are in `tests/` and use pytest. Tests run offline using synthetic weather fixtures — no API calls required.

- `test_app.py` — public API facade
- `test_battery.py` — battery simulation and degradation
- `test_economics.py` — cost calculations and projections
- `test_emissions.py` — CO2 savings
- `test_solar.py` — PV production

To run a specific test file:

```bash
uv run pytest tests/test_app.py -v
```

## Pull Request Guidelines

- PRs should target `develop`, not `main`
- Include a brief description of what changed and why
- Make sure CI passes. It runs lint, format checks, tests and core-package
  coverage, release artifact verification, the Sphinx docs build, and
  lightweight macOS/Windows public-entrypoint checks on every PR to `develop`
  or `main`.
- Keep PRs focused — one feature or fix per PR

## Reporting Issues

Open an issue on [GitHub](https://github.com/Str4vinci/breos/issues) with:

- What you expected to happen
- What actually happened
- Steps to reproduce (if applicable)
- Python version and OS

## Questions

For questions or collaboration, reach out at lrodrigues@fe.up.pt.
