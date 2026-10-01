# Architecture and implementation notes

These documents give contributors design context. They are not part of the
Read the Docs user site. A note describes either the current design or a
proposal. Accepted decisions and their reasons are in the
[architecture decision records](../adr/index.md). Delivery plans are removed
once they ship; git history keeps them.

| Note | Status |
|---|---|
| [Tariffs and smart charging](tariffs-and-smart-charging.md) | Current design (0.7.0): tariff domain, smart-charging modes, valuation and validation tooling |
| [Numba dispatch backend](numba-dispatch-backend.md) | Current design: bit-identity contract and backend boundaries |
| [BLAST degradation engine](blast-degradation-engine.md) | Current design, with deferred Monte Carlo and resistance work |
| [Battery degradation policy](battery-degradation-policy.md) | Active maintainer policy |
| [String inverter sizing](string-inverter-sizing.md) | Proposed capability |

Use `ROADMAP.md` for public release intent and GitHub issues for active
discussion. Update user-facing `docs/` only when behavior is implemented and
ready for users to rely on.
