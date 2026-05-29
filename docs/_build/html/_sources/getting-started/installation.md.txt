# Installation

BREOS requires Python 3.13 or newer.

## From PyPI

```bash
pip install breos
```

## From source

```bash
git clone https://github.com/Str4vinci/breos.git
cd breos
pip install -e .
```

## Development install

If you plan to contribute, install with the dev extras for testing and
linting:

```bash
pip install -e ".[dev]"
```

To build the documentation locally, install the `docs` extras:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

The repository uses [`uv`](https://docs.astral.sh/uv/) for dependency
management. If you prefer uv, the equivalents are:

```bash
uv sync --extra dev          # development
uv sync --extra docs         # documentation
```

## Verifying the install

```python
import breos
print(breos.__version__)
```
