# Public Release Checklist

This checklist keeps the public release process reproducible and protects
`develop` from regressions while BREOS is pre-1.0.

## Branch Protection

`develop` and `main` are protected in the GitHub repository settings, with
the rules enforced for administrators too:

- Require pull requests before merging, and dismiss stale approvals when new
  commits are pushed.
- Require the four `test (3.11)` to `test (3.14)` checks of the `Tests`
  workflow. Each runs the test suite; `test (3.12)` also runs the lint,
  format and type checks, the installed-wheel release artifact check and the
  docs build. On a documentation-only PR the other three versions skip their
  steps and pass.
- Require conversation resolution before merging.
- Block force pushes and branch deletion.

`main` also requires branches to be up to date before merging. `develop` does
not (since 2026-10-01), so two PRs that pass alone can still break together;
the `Tests` run on every push to `develop` checks each merged tree.

Under Settings → Actions → General, workflows from fork PRs need approval
for all outside collaborators, not only first-time contributors. Read the
diff before approving a run.

Use `main` only for stable releases. Release tags are created on `main` after
the release commit has passed the same checks.

## Release Branch

1. Merge every release-scope PR into `develop` and wait for the `Tests` run
   on the resulting `develop` push. Trigger the workflow manually
   (`workflow_dispatch`) on `develop` once, so the macOS/Windows, slow and
   coverage jobs run on the release candidate too.
2. Create `release/X.Y.Z` from `develop`. On it:
   - Set `version` in `pyproject.toml` and run `uv lock`, which records the
     new version in `uv.lock`.
   - Move the `[Unreleased]` changelog entries under `## [X.Y.Z] - YYYY-MM-DD`.
   - Set `version` in `CITATION.cff`; set `date-released` on the release day.
   - Update the supported-versions table in `SECURITY.md`.
   - Regenerate the PV validation baseline and report after the version
     bump (see below).
3. Push the branch and open a PR from it into `main`. The push and the PR run
   the full Linux matrix, the floors and no-Numba jobs, the macOS/Windows
   smoke tests, the slow tests and the coverage report.

## Pre-Release Gates

Run these checks locally on the release branch:

```bash
uv run ruff check breos/ tests/ tools/
uv run ruff format --check breos/ tests/ tools/
uv run mypy breos
uv run pytest tests/ -v -n auto
uv run pytest tests/ -m slow -v
uv run python tools/verify_release_artifacts.py
BREOS_DOCS_OFFLINE=1 uv run --extra docs sphinx-build -W -b html docs docs/_build/html
uvx cffconvert --validate
```

The test suite includes `tools/generate_config_docs.py --check` (the
generated configuration reference matches the schema) and the App golden
outputs (`tools/generate_app_golden.py --check`, bit for bit).

The release artifact verifier must build both wheel and sdist, confirm packaged
runtime data is present, confirm generated docs are not shipped, and import
BREOS from the installed wheel instead of the source checkout. It also imports
all 14 vendored BLAST models and verifies the installed BLAST license, DOE
notice, and pinned upstream provenance.

Regenerate the example-gallery results *after* bumping the package version,
so every page is stamped with the release: run
`uv run python tools/regenerate_gallery_results.py` (all cases, about ten
minutes; it fetches the Open-Meteo history for the Monte Carlo case), then
`uv run python tools/regenerate_gallery_results.py --check`, and commit
`docs/examples/_results/`.

Regenerate `validation/baselines/breos_baseline.json` *after* bumping the
package version, not before. The baseline records `breos.__version__` as read
at generation time, so a baseline generated on the previous version stamps
itself with that version while encoding the new release's behavior, which
misleads anyone later diffing it against the release it names. The same
applies to `validation/REPORT.md`, which names the version it was generated
against:

```bash
uv run python validation/run_breos.py --write-baseline
uv run python validation/compare.py
uv run pytest tests/test_validation_drift.py
```

Any yield change in the regenerated baseline must be explained by a changelog
entry.

## Release Validation Matrix

The `Tests` workflow runs the complete matrix on Python 3.11, 3.12, 3.13, and
3.14 on every PR and push, together with a `floors` job on the lowest declared
dependency versions and a `no-numba` job without the optional Numba backend.
Draft PRs run nothing until they are marked ready for review. A PR that
changes only documentation (`docs/`, `design/`, `maintainers/`, the top-level
Markdown files and `CITATION.cff`) runs the suite, checks and docs build on
Python 3.12 alone.
macOS and Windows run a focused public-entrypoint smoke suite and the backend
bit-identity check on PRs into and pushes to `main` and `release/**`, nightly,
and on demand. The slow tests and a separate `coverage-report` job run
nightly, on demand, and on pushes to and PRs into `release/**`. The coverage
job publishes branch-aware core-package coverage on Python 3.12, excluding the
vendored BLAST-Lite implementation, so a release build produces a coverage
snapshot without every pull request paying for one. That job is a report, not
a gate — no threshold is configured, and a failure there means the
instrumented run broke rather than that coverage is insufficient. The
following release claims must remain tied to executable checks:

| Gate | Executable coverage |
| --- | --- |
| Native remains the default and matches an explicit native run | `tests/test_battery_profiles.py`, `tests/test_runners.py` |
| Adapter parameters and trajectories match pinned upstream BLAST | `tests/test_blast_multicondition_parity.py` |
| All 14 models execute and restore snapshots | `tests/test_blast_engine.py` |
| One continuous run equals snapshot continuation | `tests/test_blast_engine.py`, `tests/test_runners.py` |
| Leap-year and 15-minute behavior | `tests/test_load_profiles.py`, `tests/test_weather.py`, `tests/test_battery.py` |
| Replacement resets model state and battery inventory | `tests/test_battery.py` |
| Battery power limits and shared inverter interactions | `tests/test_battery.py`, `tests/test_inverter.py` |
| Bifacial metadata is inert by default and rear gain is opt-in, attributable, and provenance-carrying | `tests/test_solar.py`, `tests/test_app.py`, `tests/test_validation_drift.py` |
| Legacy PV defaults remain stable while IAM, temperature, and recommended/equal-nameplate rows are reproducible | `tests/test_pv_model_core.py`, `tests/test_solar.py`, `tests/test_validation_drift.py` |
| Snapshot JSON round trips and schema rejection | `tests/test_battery_profiles.py` |
| Range/horizon warnings deduplicate across continuation | `tests/test_blast_engine.py`, `tests/test_runners.py` |
| Installed wheel contains models, provenance, license, and notice | `tools/verify_release_artifacts.py` |
| App results match the committed golden outputs bit for bit | `tests/test_app_golden.py` |
| Tariff valuation, reference tariff, and smart charging | `tests/test_app_tariff.py`, `tests/test_reference_tariff.py`, `tests/test_smart_charging.py` |
| Python and Numba dispatch backends are bit-identical | `tests/test_numba_dispatch_parity.py` |
| Importing BREOS does not need Numba | `no-numba` job in `.github/workflows/tests.yml` |

The upstream parity fixture records the generating Python and NumPy versions;
it is a checked release artifact, not regenerated during CI. Regeneration must
use the pinned unmodified BLAST-Lite source and be reviewed as a scientific
data change.

### Regenerating the BLAST parity fixture

`tools/generate_blast_parity_fixture.py` is the maintained generator for
`tests/fixtures/blast/blast_parity_multicondition.json`. Its adjacent
`.manifest.json` sidecar records the exact source commit and version, Python
and NumPy versions, fixture schema and SHA-256, named profile definitions and
hashes, and the canonical generation command. Ordinary tests read these
committed artifacts and do not need a BLAST-Lite checkout.

Clone BLAST-Lite separately, check out the commit recorded in the manifest,
and install its runtime dependencies in an environment with the recorded
Python and NumPy versions. The generator requires the repository root
explicitly and checks that it is clean before importing `blast.models`:

```bash
git clone https://github.com/NatLabRockies/BLAST-Lite.git /tmp/BLAST-Lite
git -C /tmp/BLAST-Lite checkout d789e00bca60f628de640745c18eb724b07358bd
git -C /tmp/BLAST-Lite status --short
python tools/generate_blast_parity_fixture.py --blast-checkout /tmp/BLAST-Lite --check
```

Omit `--check` to rewrite the fixture and sidecar through atomic file
replacement. Review both as scientific artifacts. The tool refuses a
different commit unless `--allow-unexpected-commit` is passed; use that
override together with an explicit `--source-version` only in a pull request
that updates BREOS's upstream pin.

## Publishing To PyPI

The `Publish` workflow (`.github/workflows/publish.yml`) uses
[PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) — no
API tokens are stored in the repository. It builds the wheel and sdist with
`uv build`, runs the release artifact verifier on exactly those files, and
uploads them. A tag push publishes to PyPI after checking that the tagged
commit is on `main`; a manual run publishes to TestPyPI.

One-time PyPI setup (per index):

1. On pypi.org, open **Your account → Publishing** and add a publisher for
   project `breos`: owner `Str4vinci`, repository `breos`, workflow
   `publish.yml`, environment `pypi`. Use a *pending* publisher if the
   project does not exist on the index yet — the first successful publish
   creates and claims the project name.
2. In the GitHub repository settings, create a `pypi` deployment environment
   and restrict it to `v*` tags. Optionally require manual approval so every
   upload gets a human confirmation step.
3. Repeat both steps on test.pypi.org with environment `testpypi` to enable
   the dry-run path.

Release flow:

1. Merge the release PR into `main` after all gates pass.
2. Optionally trigger the `Publish` workflow manually (`workflow_dispatch`) on
   `main` to dry-run the upload against TestPyPI, then verify the release installs
   into a throwaway environment. Download only the BREOS wheel from TestPyPI
   and install that file, so every runtime dependency comes from PyPI:

   ```
   rm -rf /tmp/breos-testpypi-dist
   uvx pip download --no-deps --only-binary :all: \
     --index-url https://test.pypi.org/simple/ \
     --dest /tmp/breos-testpypi-dist breos==X.Y.Z
   uv venv --clear /tmp/breos-testpypi
   VIRTUAL_ENV=/tmp/breos-testpypi uv pip install \
     /tmp/breos-testpypi-dist/breos-X.Y.Z-py3-none-any.whl
   VIRTUAL_ENV=/tmp/breos-testpypi uv pip show breos
   VIRTUAL_ENV=/tmp/breos-testpypi uv pip check
   cd /tmp && /tmp/breos-testpypi/bin/python -c "import breos; print(breos.__version__)"
   /tmp/breos-testpypi/bin/breos run --location porto --n-modules 10 \
     --annual-consumption-kwh 4000 --dry-run
   ```

   The import runs from `/tmp` so it loads the installed wheel, not the source
   checkout, and the dry run is the quickstart's offline installation check.
   Do not list TestPyPI as an index for `uv pip install`: TestPyPI hosts a
   `numpy` project, and uv's default `first-index` strategy then takes `numpy`
   only from TestPyPI, where no release satisfies `numpy>=2.0`. Do not work
   around it with an `unsafe-*` index strategy either; it lets any TestPyPI
   upload stand in for a dependency.
3. Tag the release commit on `main` with an annotated tag
   (`git tag -a vX.Y.Z -m "BREOS X.Y.Z" && git push origin vX.Y.Z`). The
   workflow refuses tags whose commit is not on `main`, then publishes to
   PyPI. See [Release tags](#release-tags) for the historical exceptions.
4. Create the GitHub Release from the tag. Publishing it triggers the
   repository's Zenodo webhook, which archives the release; check the Zenodo
   record's metadata against `CITATION.cff`. Confirm the published release
   installs from a clean environment:

   ```
   uv venv --clear /tmp/breos-pypi
   VIRTUAL_ENV=/tmp/breos-pypi uv pip install breos==X.Y.Z
   VIRTUAL_ENV=/tmp/breos-pypi uv pip show breos
   VIRTUAL_ENV=/tmp/breos-pypi uv pip check
   ```
5. Merge `main` back into `develop` with a PR ("Merge main into develop after
   X.Y.Z release"), so the release commits and dates are on `develop`.

## Release Tags

Release tags are annotated, with the message `BREOS X.Y.Z`. Three historical
tags are lightweight instead:

| Tag | Commit | Date |
| --- | --- | --- |
| `v0.5.0` | `134c99e` (merge of #108) | 2026-08-05 |
| `v0.5.1` | `62ab4f4` (merge of #117) | 2026-08-11 |
| `v0.6.0` | `c1a641d` (merge of #140) | 2026-08-31 |

Each points at the release merge commit and was published from it, so the
commit is correct; only the tag object is missing. Leave them as they are. Do
not delete and re-push them as annotated tags: that rewrites a published ref,
and anyone who fetched the old tag keeps it. `git describe` skips lightweight
tags unless `--tags` is passed, so use `git describe --tags` when these
releases matter.

## Data And Docs

- Keep runtime config and redistributable load-profile data under `breos/data/`
  and load it through package resources.
- Keep generated docs out of git and release artifacts.
- Keep Sphinx source docs in `docs/` so the documentation site is rebuildable.
- Do not bundle external load profiles unless redistribution permission is
  recorded and covered by tests.

## API Stability

- Treat `breos.App`, documented config/result keys, and names in `breos.__all__`
  as the primary stable surface for the 0.x series.
- Add golden-output tests before changing core energy balance, economics,
  emissions, or time-resolution behavior.
- Keep module-level APIs importable unless a removal is documented in the
  changelog with a migration path.

## Public Wording

Use **Python library for PV and battery energy-system simulation and
optimization** as the default public wording in the README, package metadata,
docs front page, and release notes.

Rationale:

- `library` matches how users install and import BREOS.
- `simulation and optimization` describes the main user-facing purpose.
- `framework` can describe the internal/extensible architecture, but sounds
  broader than the current pre-1.0 public surface.
- `engine` can be used for the simulation core, but should not describe the
  whole project unless a stable lower-level engine API is intentionally exposed.
- `model` should refer to specific algorithms or component models, not the full
  project.
