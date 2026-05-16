# 0001 — Docs organized by puzzle piece, App-led

BREOS docs are organized by domain "puzzle piece" (Weather, PV, Load
profiles, Energy balance, Battery, Cost analysis, Optimization, Plotting)
rather than by Python module. The landing page leads with the
{py:class}`~breos.App` facade — the unique thing BREOS offers — with
composable submodule usage introduced later as "Building custom pipelines."

The API reference mirrors the same eight puzzle pieces with hand-curated
`autosummary` lists, plus a recursive `api/appendix.md` for utilities,
constants, and other names that ended up in `__all__` but aren't part of
the primary surface. Narrative ("user-guide/") will be separated from
theory ("concepts/") in future PRs so degradation-model math doesn't drown
the practical battery-configuration page.

## Why

- The App facade is BREOS's differentiator versus pvlib; leading with it
  gives a 30-second first-time-user win.
- Puzzle pieces cross Python module boundaries (PV = `solar.py` +
  `pv_modules.py` + `inverter.py`). Documenting per-module would scatter
  related material.
- Curated `autosummary` blocks keep ~30 user-facing names on the index
  while still surfacing the full `__all__` via the appendix's recursive
  crawl.
