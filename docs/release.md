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
uv run --extra docs sphinx-build -b html docs docs/_build/html
```

The release artifact verifier must build both wheel and sdist, confirm packaged
runtime data is present, confirm generated docs are not shipped, and import
BREOS from the installed wheel instead of the source checkout.

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
