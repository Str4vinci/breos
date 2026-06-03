# 0001 — Docs organized by puzzle piece, App-led

BREOS docs are organized by domain "puzzle piece" (Weather, PV, Load
profiles, Energy balance, Battery, Cost analysis, Optimization, Plotting)
rather than by Python module. The landing page leads with the
{py:class}`~breos.App` facade — the unique thing BREOS offers — with
composable submodule usage introduced later as "Building custom pipelines."

The API reference mirrors the same eight puzzle pieces with hand-curated
`autosummary` lists, plus a recursive `api/appendix.md` for utilities,
constants, and other module APIs that remain importable but are not part of
the narrow top-level release surface. Narrative ("user-guide/") will be
separated from theory ("concepts/") in future PRs so degradation-model math
doesn't drown the practical battery-configuration page.

## Why

- The App facade is BREOS's differentiator versus pvlib; leading with it
  gives a 30-second first-time-user win.
- Puzzle pieces cross Python module boundaries (PV = `solar.py` +
  `pv_modules.py` + `inverter.py`). Documenting per-module would scatter
  related material.
- Curated `autosummary` blocks keep the primary user-facing names on the
  index while the appendix still surfaces additional module APIs through a
  recursive crawl.
