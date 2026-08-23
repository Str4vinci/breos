# 0002 — Tariffs are resolved values; smart charging is an instruction layer

- **Status:** Accepted for 0.7.x implementation
- **Date:** 2026-08-20

## Context

BREOS 0.5.x values every imported and exported kilowatt-hour at one annual
price. The historical PhD tree contains useful time-of-use (TOU) schedules and
smart-charging experiments, but it also contains a second battery simulator,
tariff-specific degradation and economics paths, and supplier-price estimates.
Porting that module would give the same configuration different physical and
financial meanings depending on its entry point.

`breos.App` remains the stable public entry point. The canonical battery step
must remain the sole owner of energy conservation, conversion losses, power and
SOC limits, degradation, replacement, and ledger construction. Tariff
classification, price resolution, and controller decisions may prepare values
for that step; they may not reproduce it.

## Source freeze

The legacy source is frozen at commit
`183bcc124d2176c496d189cee0341bac617b6d54`. The reviewed files and Git blob
identifiers are:

| Legacy path | Blob |
|---|---|
| `dev/pvbat/tou.py` | `ac77e9af52028f3b8e0e3670a0ef67f439ccfb23` |
| `dev/pvbat/payback.py` | `318fd40ad05308b0f1add6fbc740812a4e3865a7` |
| `dev/pvbat/acc.py` | `4ada58a38ef32470207eb1ac56c1a59b259661ce` |
| `dev/pvbat/numba_kernels.py` | `ec7e410fa9fd389ece8a57a6217c2d55dea769e8` |
| `dev/tools/compute_a2_daily_sc_oracle.py` | `c9efa5398cd4ff53d51ebe7366f2de2d94685078` |
| `dev/tools/compute_a2_perfect_foresight_bound.py` | `d1bcf9a24329e55b9e9268ea0d6b4f42abe4398d` |
| `dev/docs/a2_daily_sc_oracle_handover.md` | `8383a8b40947c77cbd197ee75cf90d7ea3f27868` |
| `dev/docs/adr/0003-real-economics-basis.md` | `8da169bb6dc68c9dadffae5370ee032e3325dc13` |

The PhD worktree is dirty. Uncommitted files and notebook results are evidence,
not implementation sources. Ports must be derived from the frozen blobs above
or independently reimplemented against primary sources.

The frozen TOU blob's 2027 Portuguese docstring incorrectly attributes the
electricity-period decision to ERSE Directive 3/2026. ERSE's acts catalog
identifies that directive as the gas-tariff decision for gas year 2026–2027.
BREOS must cite ERSE's final CP137 report, closing explainer, and current
electricity-tariff page for the approved 2027 periods. Spain's 2.0TD schedule
must cite CNMC Circular 3/2020 rather than Royal Decree 446/2023. Supplier
prices are separate, dated inputs and are never implied by those schedules.

## Decision

### Tariff configuration

Omitting `tariff` preserves the 0.5.x flat-price path exactly, including its
EUR interpretation and existing `costs.electricity_cost`,
`costs.electricity_sold_cost`, and `costs.daily_power_cost` inputs.

TOU valuation uses one nested top-level `tariff` table:

```toml
[tariff]
schedule = "pt_mainland_2026_daily_tri"
currency = "EUR"
import_prices = { peak = 0.30, mid_peak = 0.20, off_peak = 0.12 }
export_prices = { all = 0.04 }
fixed_charge_per_day = 0.30
boundary_policy = "strict"
```

The exact keys have these meanings:

- `schedule` is a versioned bundled schedule identifier. It contains civil-time
  period rules and regulatory provenance, never supplier prices.
- `currency` is an uppercase supported ISO 4217 code shared by energy prices,
  fixed charges, CAPEX, OPEX, and replacement costs. 0.7.0 initially enables
  EUR; adding a currency requires a complete same-currency cost catalog or
  explicit user costs. BREOS does not convert currencies or fetch FX rates.
- `import_prices` and `export_prices` map period labels to non-negative values
  per kWh. `all` is an explicit fallback for every schedule period; missing
  used periods otherwise fail validation.
- `fixed_charge_per_day` is a non-negative charge in the declared currency.
- `boundary_policy` is `strict` in 0.7.0. A schedule containing boundaries that
  the simulation resolution cannot represent is rejected. Approximation may be
  added later only as an explicit policy recorded in provenance.

There is no country fallback and no generic German preset. Initial bundled
schedule identifiers are `pt_mainland_2026_daily_bi`,
`pt_mainland_2026_daily_tri`, `pt_mainland_2026_weekly_bi`,
`pt_mainland_2026_weekly_tri`, `pt_mainland_2027_daily_bi`,
`pt_mainland_2027_daily_tri`, `pt_mainland_2027_weekly_bi`,
`pt_mainland_2027_weekly_tri`, and `es_2_0td`. The 2027 Portuguese identifiers
describe the approved schedule, but their provenance must record the phased
effective dates; selecting one before it applies requires an explicit
user-supplied study date rather than silent calendar switching.

Schedule and price data remain separate internally. A later convenience preset
may refer to one schedule and one dated price set, but neither can mutate the
other and the resolved result records both identifiers.

### Resolved tariff value

The tariff domain exposes three immutable concepts:

1. a schedule definition with period rules and regulatory provenance;
2. prices with currency and price-source provenance; and
3. a resolved tariff aligned to one simulation index.

A resolved tariff contains period labels/codes, import prices, export prices,
fixed charge, timezone, currency, source identifiers, effective dates, and a
deterministic schedule hash. Resolution classifies each timezone-aware instant
in local civil time while retaining the original instant ordering. Repeated DST
hours therefore remain two distinct instants with the same local clock label;
nonexistent local instants are never manufactured.

The hash covers the schedule identifier/version, tariff timezone, UTC
nanosecond instants, and resolved period labels. Price values and price-source
provenance have their own hash so a pure revaluation can change prices without
pretending the schedule changed.

### Valuation and cashflows

Each project-year simulation is valued inside the existing App year loop.
Annual import cost and export revenue are sums of timestep energy multiplied by
the resolved price arrays; the no-system baseline uses the same arrays and
calendar. Annual energy totals remain alongside monetary components.

New monetary names are currency-neutral. Existing `*_eur` results remain
compatibility aliases only while the resolved currency is EUR. Mixed-currency
inputs fail before simulation. Initial CAPEX, imports, exports, fixed charges,
O&M, and replacements remain distinct annual cashflow components. Simple and
sustained discounted payback are separate outputs; NPV remains the financial
ranking metric.

### Smart-charging configuration

Omitting `smart_charging`, or setting its mode to `disabled`, produces no
instructions and must match the current greedy simulation exactly.

```toml
[smart_charging]
mode = "fixed_target"
target_usable_fraction = 0.50
charge_periods = ["off_peak"]
discharge_periods = ["mid_peak", "peak"]
grid_charge_efficiency = 0.95
grid_import_limit_w = 5000
```

`target_usable_fraction` is deliberately not named `target_soc`: zero maps to
the configured minimum SOC and one maps to the configured maximum SOC. This
avoids treating unusable nominal capacity as an available target. Charge and
discharge period names must exist in the resolved tariff. Fixed-target mode
requires a tariff and a positive-capacity battery.

`grid_import_limit_w` caps total site import, including simultaneous load. Grid
charging is also bounded by battery charge power and the hybrid inverter's AC
rating. The configured `grid_charge_efficiency` is the AC-to-stored-DC
efficiency and is independent of the DC-to-AC discharge efficiency. The first
supported strategy does not grid-charge while PV is being exported.

### Dispatch instructions and origin accounting

The controller produces aligned arrays for:

- whether discharge is allowed;
- the minimum usable-energy fraction retained before discharge; and
- the target usable-energy fraction for grid charging, with no target on
  inactive steps.

These are inputs to the canonical dispatch step, not requested energy flows.
The step computes feasible flows from the current effective capacity, power
limits, load, PV, and inverter constraints. A pure-Python implementation is the
reference. An optional Numba day kernel may implement the identical array
contract only after full ledger parity and warm end-to-end speedup are shown.

Stored energy is split into PV and grid origins. Charging adds to the matching
origin. Discharge and standing/capacity losses remove each origin
proportionally to its share at the beginning of that operation. Replacement
removes both origins and initializes replacement energy under the existing
battery-state convention. The ledger records both origin balances; only
PV-origin discharge contributes to PV self-consumption and avoided-grid
emissions.

### Boundary and terminal conventions

Normal `App.run()` simulations use `physical_carry`: stored energy, origin
shares, and degradation state flow from one project year into the next and the
initial/final states are reported. Validation or optimisation objectives must
declare a terminal convention. Smart-charging oracle comparisons default to
`cyclic_soc`; free terminal depletion is never an unreported benefit.

Adding grid-origin flows and component cashflows advances the ledger schema to
2.0. The default greedy path remains numerically compatible, but consumers can
use the schema version to detect the additive origin and valuation fields.

## Consequences

- Tariff schedules can be tested independently of supplier offers and can be
  revalued without rerunning price-blind dispatch.
- Price-aware dispatch forces simulation because changing prices may change
  instructions and physical flows.
- The App runner remains the only project simulation and economics entry point.
- Half-hour regulatory boundaries initially require 15-minute input; hourly
  studies fail rather than receive an undocumented approximation.
- Live prices, FX, thermal storage, heat pumps, electro-thermal dispatch, V2H,
  and community settlement are outside this decision.

## Implementation gates

Implementation follows the delivery sequence in
`design/architecture/0.7x-phd-porting-plan.md`. In particular:

1. tariff resolution and valuation land before dispatch changes;
2. no-op instruction parity covers native and BLAST degradation;
3. fixed-target charging lands only with per-step conservation and origin
   reconciliation tests; and
4. persistence controllers and perfect-information oracles remain experimental
   or tooling-only and replay every schedule through production physics.
