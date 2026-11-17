# Installation

BREOS requires Python 3.11.5 or newer.

## From PyPI

```bash
pip install breos
```

With [uv](https://docs.astral.sh/uv/):

```bash
# add BREOS to a uv project
uv add breos
# or run the CLI without installing it
uvx breos --version
```

To install a specific release tag directly from GitHub instead, replace
`X.Y.Z` with the release you want:

```bash
pip install "breos @ git+https://github.com/Str4vinci/breos.git@vX.Y.Z"
```

## Check the install

```bash
breos --version
```

It prints the installed version, for example `breos 0.8.0`. Then run your
first simulation in the [Quickstart](quickstart.md). See
[Troubleshooting](troubleshooting.md) if a command fails or a simulation
cannot reach its weather source.

## From source

```bash
git clone https://github.com/Str4vinci/breos.git
cd breos
pip install -e .
```

## Optional features

The base install includes the core simulation stack. Extras add heavier
optional packages for some workflows:

| Extra | Adds |
|---|---|
| `plots` | matplotlib plotting helpers |
| `optimization` | pymoo multi-objective sizing |
| `weather` | Open-Meteo historical weather |
| `fast` | Numba dispatch accelerator |

Name one extra, or several separated by commas:

```bash
pip install "breos[plots]"
pip install "breos[plots,weather]"
```

For a source checkout, use the editable equivalents, for example
`pip install -e ".[plots]"`.

## Development install

If you plan to contribute, install with the dev extras for testing and
linting. The dev extra also installs the optional packages exercised by the
full local test suite:

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
# development
uv sync --extra dev
# development plus documentation
uv sync --extra dev --extra docs
```
