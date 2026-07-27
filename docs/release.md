# Public Release Checklist

This checklist keeps the public release process reproducible and protects
`develop` from regressions while BREOS is pre-1.0.

## Branch Protection

Before publishing a public package release, protect `develop` in GitHub
repository settings.

Recommended `develop` rules:

- Require pull requests before merging.
- Require the `Tests` workflow to pass, including the installed-wheel release
  artifact smoke test.
- Require branches to be up to date before merging.
- Dismiss stale approvals when new commits are pushed.
- Block force pushes.
- Block branch deletion.
- Require conversation resolution before merging.

Use `main` only for stable releases and protect it at least as strictly as
`develop`. Release tags should be created from `main` after the release commit
has passed the same checks.

## Pre-Release Gates

Run these checks locally before cutting a release candidate:

```bash
uv run ruff check breos/ tests/ tools/
uv run ruff format --check breos/ tests/ tools/
uv run pytest tests/ -v
uv run python tools/verify_release_artifacts.py
uv run --extra docs sphinx-build -W -b html docs docs/_build/html
```

The release artifact verifier must build both wheel and sdist, confirm packaged
runtime data is present, confirm generated docs are not shipped, and import
BREOS from the installed wheel instead of the source checkout. It also imports
all 14 vendored BLAST models and verifies the installed BLAST license, DOE
notice, and pinned upstream provenance.

## 0.4.x Validation Matrix

The `Tests` workflow runs the complete matrix on Python 3.11, 3.12, 3.13, and
3.14. Python 3.12 also reports branch-aware core-package coverage, excluding
the vendored BLAST-Lite implementation, while macOS and Windows run a focused
public-entrypoint smoke suite. The following release claims must remain tied to
executable checks:

| Gate | Executable coverage |
| --- | --- |
| Native remains the default and matches an explicit native run | `tests/test_battery_profiles.py`, `tests/test_runners.py` |
| Adapter parameters and trajectories match pinned upstream BLAST | `tests/test_blast_multicondition_parity.py` |
| All 14 models execute and restore snapshots | `tests/test_blast_engine.py` |
| One continuous run equals snapshot continuation | `tests/test_blast_engine.py`, `tests/test_runners.py` |
| Leap-year and 15-minute behavior | `tests/test_load_profiles.py`, `tests/test_weather.py`, `tests/test_battery.py` |
| Replacement resets model state and battery inventory | `tests/test_battery.py` |
| Battery power limits and shared inverter interactions | `tests/test_battery.py`, `tests/test_inverter.py` |
| Snapshot JSON round trips and schema rejection | `tests/test_battery_profiles.py` |
| Range/horizon warnings deduplicate across continuation | `tests/test_blast_engine.py`, `tests/test_runners.py` |
| Installed wheel contains models, provenance, license, and notice | `tools/verify_release_artifacts.py` |

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
API tokens are stored in the repository. It re-runs the release artifact
verifier, builds the wheel and sdist with `uv build`, and uploads them.

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
2. Optionally trigger the `Publish` workflow manually (`workflow_dispatch`)
   to dry-run the upload against TestPyPI, then verify with
   `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ breos`.
3. Tag the release commit on `main` (`git tag vX.Y.Z && git push origin vX.Y.Z`).
   The workflow refuses tags whose commit is not on `main`, then publishes to
   PyPI.
4. Create the GitHub Release from the tag and confirm
   `pip install breos==X.Y.Z` works from a clean environment.

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
