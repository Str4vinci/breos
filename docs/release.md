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

## Data And Docs

- Keep runtime config and redistributable load-profile data under `breos/data/`
  and load it through package resources.
- Keep generated docs out of git and release artifacts.
- Keep Sphinx source docs in `docs/` so the documentation site is rebuildable.
- Do not bundle external load profiles unless redistribution permission is
  recorded and covered by tests.

## API Stability

- Treat `breos.App` and documented config/result keys as the primary stable
  surface for 0.1.x.
- Add golden-output tests before changing core energy balance, economics,
  emissions, or time-resolution behavior.
- Keep module-level APIs importable unless a removal is documented in the
  changelog with a migration path.

## Open Positioning Question

Before the public release, decide the project wording used in the README,
package metadata, docs front page, and release notes:

> Is BREOS a simulation engine, a framework, a model, or a library?

Working recommendation: use **Python library for PV and battery energy-system
simulation and optimization** as the default public wording.

Rationale:

- `library` matches how users install and import BREOS.
- `simulation and optimization` describes the main user-facing purpose.
- `framework` can describe the internal/extensible architecture, but sounds
  broader than the current 0.1.0 public surface.
- `engine` can be used for the simulation core, but should not describe the
  whole project unless a stable lower-level engine API is intentionally exposed.
- `model` should refer to specific algorithms or component models, not the full
  project.
