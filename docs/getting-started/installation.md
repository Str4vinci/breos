# Installation

BREOS requires Python 3.11 or newer.

## From the stable tag

```bash
pip install "breos @ git+https://github.com/Str4vinci/breos.git@v0.2.3"
```

PyPI publishing is planned for a future release. Until then, install the latest
stable GitHub tag or use a source checkout.

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
pip install -e ".[plots]"          # matplotlib plotting helpers
pip install -e ".[optimization]"   # pymoo multi-objective sizing
pip install -e ".[weather]"        # Open-Meteo historical weather fetching
pip install -e ".[fast]"           # Numba kernels
pip install -e ".[validation]"     # Excel / Arrow validation workflows
pip install -e ".[location-tools]" # geocoding and timezone lookup helpers
```

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
