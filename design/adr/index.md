# Architecture Decision Records

Short records of decisions whose context isn't visible in the code. ADRs
exist to answer "why did you do it this way?" for future readers — the
code shows the *what*. An accepted decision is not rewritten later. An
amendment, or a note in place, marks text that a later decision replaced
or that the implementation does differently.

| ADR | Status |
|---|---|
| [0001: Documentation architecture](0001-docs-architecture.md) | Accepted; updated for 0.7.0 |
| [0002: Tariffs and smart charging](0002-tariffs-and-smart-charging.md) | Accepted, amendments A1–A17; A18 and A19 proposed; implemented in 0.7.0, A15–A19 in 0.7.1 |
| [0003: Economic basis and currency-neutral results](0003-economic-basis.md) | Accepted, E1–E10; implemented in 0.7.0. E11–E13 Proposed, implemented for 0.7.1 |
| [0004: AC-coupled batteries](0004-ac-coupled-batteries.md) | Accepted 2026-10-04, amended 2026-10-13; implemented for 0.8.0 |
| [0007: String topology and electrical checks](0007-string-topology-and-electrical-checks.md) | Proposed 2026-10-11; checks and energy per MPPT implemented, several inverters without a battery |
| [0008: Shading as an input](0008-shading-input.md) | Accepted 2026-10-12; level 1 implemented |
| [0009: Module electrical layout](0009-module-electrical-layout.md) | Accepted 2026-10-13; `bypass_diodes` on the module implemented |
