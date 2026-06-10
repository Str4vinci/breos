# Installation

BREOS requires Python 3.11 or newer.

## From the stable tag

```bash
pip install "breos @ git+https://github.com/Str4vinci/breos.git@v0.3.0"
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
pip install "breos[plots] @ git+https://github.com/Str4vinci/breos.git@v0.3.0"
pip install "breos[optimization] @ git+https://github.com/Str4vinci/breos.git@v0.3.0"
pip install "breos[weather] @ git+https://github.com/Str4vinci/breos.git@v0.3.0"
pip install "breos[fast] @ git+https://github.com/Str4vinci/breos.git@v0.3.0"
pip install "breos[validation] @ git+https://github.com/Str4vinci/breos.git@v0.3.0"
pip install "breos[location-tools] @ git+https://github.com/Str4vinci/breos.git@v0.3.0"
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
