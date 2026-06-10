# BREOS Roadmap

This document tracks planned work that is not yet scheduled. Items here
are intentions, not commitments — see GitHub issues for active work.

## Architecture

### Wrap third-party modules behind adapters

Concentrate every direct import of `pvlib`, `scipy`, `rainflow`, and other
third-party scientific libraries in a small `breos.adapters` layer so that
upstream API changes only affect a single file rather than the whole
package. The current `Location` parameter exposed by
`solar.calculate_pv_production_dc` (and several other public functions) is
a `pvlib.Location`, which means BREOS does not own its own public API.

- Tracking issue: [#11](https://github.com/Str4vinci/breos/issues/11)
- Design: [docs/architecture/third-party-wrapping.md](docs/architecture/third-party-wrapping.md)
- Scope: pvlib first (Phase 1), then scipy / rainflow (Phase 2), then IO
  clients (Phase 3). Pandas, numpy, and matplotlib are kept direct.
- Estimated effort: ~3–4 weeks of focused work, split into many small
  PRs.

## Onboarding and discoverability

### Make the first successful run easier to trust

The 0.3.0 onboarding pass added the "10-minute first run" quickstart,
the required-inputs page, the recipes page, the generated packaged-options
reference, the `breos list` discovery commands, and `breos validate-config` /
`breos run --dry-run` config inspection with `--json` output. Remaining work:

Ongoing docs hygiene:

- Keep install snippets and docs status text aligned with the current release
  on PyPI.
- Keep the README and getting-started configuration tables in sync with the
  public `App` config surface, including battery SOC/EOL/RTE keys and
  `pv_loss_overrides`.
- Keep the representative quickstart output excerpt in the docs close to what
  current dependency versions actually produce.

Option discovery work:

- Add matching Python helpers where useful, building on `list_modules()`.

Config inspection work:

- Extend the dry-run summary with the fully resolved PVWatts loss components
  (today it only echoes `pv_loss_overrides`).

Agent and contributor setup:

- Keep `AGENTS.md` and `CONTRIBUTING.md` aligned on branch flow, test gates,
  and the public `breos.App` facade as the preferred extension point.
- Consider a single `make`/`just`/script entry for local validation if command
  drift becomes a recurring issue.

## Capability extensions

### String-aware inverter validation and modeling

BREOS currently models PV systems at the aggregate array level. That is useful
for fast production, battery, and economics studies, but it does not prove that
a proposed PV layout is electrically buildable. Future work should add
string-aware validation and, later, string-aware inverter modeling when callers
provide module, inverter, environment, MPPT, and string-topology data.

- Design note: [docs/architecture/string-inverter-sizing.md](docs/architecture/string-inverter-sizing.md)
- Phase 1: apply aggregate inverter AC clipping consistently in the main
  `App` energy flow.
- Phase 2: add a pure validation API for string voltage windows, startup
  voltage, MPPT current limits, parallel-string compatibility, and DC/AC ratio
  warnings.
- Phase 3: extend module and inverter catalogs with the datasheet fields needed
  for those checks.
- Phase 4: accept optional MPPT/string topology from callers and use it to
  improve multi-array energy modeling.
- Non-goal: code-compliance certification, conductor/fuse sizing, and physical
  wiring auto-routing.

## Distribution and release automation

### PyPI trusted publishing

Shipped in 0.3.0: `.github/workflows/publish.yml` publishes `v*` tags that
point at `main` to PyPI via GitHub Actions OIDC trusted publishing, re-runs
the release artifact verifier before upload, and offers a manually triggered
TestPyPI dry-run. The one-time index and environment configuration is
documented in [docs/release.md](docs/release.md). Remaining work:

- Protect `v*` tags in GitHub repository settings so only maintainers can
  create release tags.

## Longer-Term Research Modules

The following modules exist in the broader research codebase and may be
released into BREOS in future versions, or are available for academic
collaboration upon request (see README for context):

- Time-of-Use (TOU) tariff optimization with multi-period pricing and
  strategy comparison.
- Vehicle-to-Home (V2H) simulation with EV scheduling and bidirectional
  charging.
- Multi-chemistry battery support — Sodium-ion (SIB), Vanadium Redox Flow
  (VRFB), Solid-State (SSB).
- Thermal energy storage (TES) with phase-change material modeling.
- Heat pump integration with COP modeling and coupled electro-thermal
  energy balance.
- Community Self-Consumption (CSC) modeling for multi-building scenarios.

## Reference load profiles pending license verification

The following sample load profiles were removed from `rlp/` and `breos.load_profiles` before the open-source release because their redistribution terms were not confirmed. They can be re-added once written permission or a clear license is obtained from the upstream providers.

- **SynPRO Family profile** (Fraunhofer ISE) — was profile key `"2"` / `family_profile_SynPro.csv`. Contact: synpro@ise.fraunhofer.de.
- **LoadProfileGenerator family-with-3-kids profile** (Noah Pflugradt, FZJ IEK-3) — was profile key `"3"` / `LoadProfileGenerator_family_3kids.csv`. Tool itself is MIT-licensed; output redistribution policy needs author confirmation.
