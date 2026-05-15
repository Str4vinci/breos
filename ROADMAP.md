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

## Capability extensions

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
