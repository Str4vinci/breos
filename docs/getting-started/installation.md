# Installation

BREOS requires Python 3.11 or newer.

## From PyPI

```bash
pip install breos
```

With [uv](https://docs.astral.sh/uv/), add it to a project or run the CLI
directly without installing:

```bash
uv add breos
uvx breos run --location porto --n-modules 10 --annual-consumption-kwh 4000
```

To install a specific release tag directly from GitHub instead:

```bash
pip install "breos @ git+https://github.com/Str4vinci/breos.git@v0.3.0"
```

## From source

```bash
git clone https://github.com/Str4vinci/breos.git
cd breos
pip install -e .
```

## Optional features

The base install includes the core simulation stack. Install extras for
workflows that need heavier optional packages:

```bash
pip install "breos[plots]"          # matplotlib plotting helpers
pip install "breos[optimization]"   # pymoo multi-objective sizing
pip install "breos[weather]"        # Open-Meteo historical weather fetching
pip install "breos[fast]"           # Numba kernels
pip install "breos[validation]"     # Excel / Arrow dependencies for local validation work
pip install "breos[location-tools]" # geocoding and timezone lookup helpers
```

For a source checkout, use the editable equivalents, for example
`pip install -e ".[plots]"`.

## Development install

If you plan to contribute, install with the dev extras for testing and
linting. The dev extra also installs BREOS's optional feature dependencies so
the full local test suite can exercise optional paths:

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
uv sync --extra dev                  # development
uv sync --extra dev --extra docs     # development plus documentation
```

## Verifying the install

```python
import breos
print(breos.__version__)
```
