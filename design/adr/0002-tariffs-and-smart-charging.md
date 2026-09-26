# 0002 — Tariffs are resolved values; smart charging is an instruction layer

- **Status:** Accepted for 0.7.x implementation; amendments A1–A10 Proposed
- **Date:** 2026-08-20; amendments 2026-09-26

## Context

BREOS 0.5.x values every imported and exported kilowatt-hour at one annual
price. A historical research tree contains useful time-of-use (TOU) schedules
and smart-charging experiments, but it also contains a second battery simulator,
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

The historical research worktree was dirty during review. Uncommitted files
and notebook results are evidence, not implementation sources. Ports must be
derived from the frozen blobs above or independently reimplemented against
primary sources.

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
`design/architecture/0.7x-tariffs-and-smart-charging-plan.md`. In particular:

1. tariff resolution and valuation land before dispatch changes;
2. no-op instruction parity covers native and BLAST degradation;
3. fixed-target charging lands only with per-step conservation and origin
   reconciliation tests; and
4. persistence controllers and perfect-information oracles remain experimental
   or tooling-only and replay every schedule through production physics.

## Amendments proposed for 0.7 readiness

The 0.7 readiness audit (#187) found details the decision above leaves open
and statements the code has since outgrown. Each amendment below is
**Proposed**. Accepting one replaces the text it names; until then the
original text stands. Economic conventions and money naming are in
[ADR 0003](0003-economic-basis.md).

### A1. Civil time comes from the configuration, not the index (#180) — Proposed

The simulation index is not in civil time. PVGIS weather keeps the UTC offset
of 1 January of the sample year (`fetch_tmy_weather_data`), so a Berlin run's
index is UTC+01:00 all year. Naive CSV and Monte Carlo weather become UTC.
Every row is still the correct instant, so an explicit conversion recovers
civil time.

- The resolver takes the configured `ResolvedAppConfig.timezone` and
  `tz_convert`s the index to it before classifying periods. It never reads
  `index.tz` and never uses naive times. The resolved tariff records that zone.
- The resolved tariff carries a civil-day boundary array: the positions where
  the local date changes, plus the end of the index. Everything with per-day
  meaning uses it: daily charge windows, daily persistence and its day-one
  warm start, and any later daily or monthly charge. A civil day is 23, 24 or
  25 hours; `steps_per_day` is never used for these.
- Degradation day windows stay positional (`steps_per_day` blocks in
  `simulate_energy_balance` and the compiled kernel). Moving them would change
  every aged result and is not tariff work. Controllers must not assume a
  degradation window is a civil day.
- Monthly result rows group by civil month in the configured zone. Today
  `monthly_to_dicts` groups on the result frame's own clock, which is the
  configured zone only when the index is in it.
- The schedule hash takes instants as
  `index.tz_convert("UTC").as_unit("ns").asi8`. pandas 3 builds
  microsecond indexes, so without `.as_unit("ns")` the hashed integers are
  not nanoseconds.

Follow-up test (with #184): pin the result index timezone for each weather
source (PVGIS fetch, preset-named local file, naive CSV, timezone-aware CSV,
injected weather and Monte Carlo), so a change in a source's offset fails a
test before it moves tariff periods.

### A2. Project years replay the start-year calendar (#180) — Proposed

Every year loop replays one year of inputs. The App reuses its `start_year`
load and PV series each year, projected optimization repeats one weather year
with one load series, and Monte Carlo restamps each sampled weather year to
its target year. The tariff follows the same rule in 0.7.0: it is resolved
once on the simulated calendar and reused for every project year. Weekday
patterns, holidays and effective dates do not advance.

Advancing calendars needs per-year load, PV and tariff construction, which
belongs to the shared projection loop of #179, and the inputs do not support
it yet: weather is a typical year and the standard load profiles are built on
one calendar. The 2027 Portuguese schedules are selected explicitly, never by
date, so one calendar does not silently misprice them. Provenance records
`calendar_policy = "replay_start_year"` and the calendar year. A leap
`start_year` makes every project year 366 days.

### A3. Half-hour boundaries need 15-minute input (#180) — Proposed

0.7.0 resolutions stay `h` and `15min`. Config validation and
`utils.get_hours_per_step` accept only those, and 30 minutes would also need
load-profile, weather-resampling and kernel support. It is not needed for
exactness: 15-minute steps represent every half-hour boundary. Under
`boundary_policy = "strict"`, hourly input with half-hour boundaries is
rejected. 30-minute input is later work.

### A4. The flat-path baseline is the App golden fixture (#187) — Proposed

Replaces "preserves the 0.5.x flat-price path exactly". Unreleased fixes,
including replacement booking at the swap instant, have already moved flat
results away from 0.5.x. The baseline is `tests/fixtures/app_golden`, written
by `tools/generate_app_golden.py` and checked by `tests/test_app_golden.py`.
Omitting `tariff` must leave it unchanged. A change that moves it regenerates
it in its own commit, with the reason.

### A5. Valuation reaches every year loop (#179) — Proposed

Replaces "valued inside the existing App year loop". Projects are simulated in
three loops: `run_app_simulation`, Monte Carlo's `_simulate_trajectory`, and
`_evaluate_projected_design_metrics` for projected optimization. Price-weighted
import cost, export revenue and fixed charge are computed once, in the shared
projection loop #179 introduces. Until an entry point uses that loop, it
rejects a `tariff` or `smart_charging` table rather than valuing at flat
prices.

`objective_basis = "steady_state"` is rejected with a tariff or smart charging
until #179 retires `calculate_financials`, which values one year at scalar
prices.

### A6. Grid-charge conversion and shared limits (#178) — Proposed

Replaces "`grid_charge_efficiency` is the AC-to-stored-DC efficiency".
`grid_charge_efficiency` is the AC-to-DC conversion of the hybrid inverter's
charging path. Grid AC energy `a` becomes DC charge input `a × η_grid`, which
then passes through the existing charge efficiency like PV charge input:
stored energy rises by `a × η_grid × eff_charge`. The DC input is booked as
battery charge input, so resistance-fade derating, `Battery_Charge_Loss`,
cell self-heating and therefore aging all see it. The AC-to-DC loss has its
own column. The key is a scalar with no default: `breos/inverter.py` has no
AC-to-DC path or part-load curve to derive one from.

Grid charging runs after PV allocation in the same step. PV keeps priority on
every limit the two share, so a grid-charge instruction never reduces PV
self-consumption:

- charge input: `cap_charge_in_wh` bounds PV and grid charge input together;
- inverter: grid-charge AC input is at most the AC nameplate minus the step's
  PV AC output, so AC throughput in both directions never exceeds the rating;
  and
- site: grid-charge import is at most `grid_import_limit_w` minus the step's
  load import.

Netting inside the converter, where PV DC goes to the battery while the grid
serves the load, is not modelled; it would change origin attribution.

### A7. The target moves with temperature and health (#178) — Proposed

`target_usable_fraction` applies each step to that step's capacity window: the
target energy is `emin + f × (emax − emin)`. `emin` and `emax` scale with the
temperature capacity factor every step and with SOH every day
(`_apply_capacity_window`), so the target is a fraction of what is usable now,
not a fixed energy. A pack charged to 0.5 on a warm night reads a different
fraction as it cools, and fade lowers the target energy over the project.
Results report the configured fraction and the stored energy reached. With a
zero instruction the target is `emin` exactly, which keeps no-op parity
reachable.

### A8. Stored energy has three origins (#178) — Proposed

Stored energy is PV origin, grid origin and an unattributed remainder. The
initial energy of a fresh run (full at max SOC) and a replacement pack's
energy are unattributed. Grid origin is therefore explicit state, never
derived as `E − E_pv`. Discharge and standby, capacity-window and replacement
removal take from all three in proportion to their shares at the start of the
operation. Only PV-origin discharge counts as self-consumption, as today.

`charge_periods` and `discharge_periods` must be disjoint, so every step
either charges or discharges. That keeps exact the single origin fraction the
step takes before dispatch; the step asserts it.

### A9. Ledger schema 2.0 reconciles origins from output (#178) — Proposed

Schema 2.0 adds, per origin, battery discharge (DC, and AC to load) and
removal by standby, capacity window and replacement, plus grid-charge AC
input, its conversion loss and its cost. Each origin then reconciles step by
step from the ledger alone: the next balance is the balance plus charge minus
discharge minus removals. Of the duplicate pairs `Battery_AC_To_Load_PV` /
`PV_Origin_Battery_AC_To_Load`, `Sell_To_Grid` / `PV_AC_Export`,
`PV_Curtailment` / `PV_DC_Curtailed` and `Battery_Standby_Loss` /
`Standby_Loss`, only the second name remains. The version constant moves from
`breos/runners/app.py` to sit beside `_LEDGER_COLUMNS`, and
`SimulationSummary` and Monte Carlo output report it as App provenance does.

### A10. Avoided emissions use net exchange (#178) — Proposed

Avoided emissions are
`(Load − Import − B_u) × CI + PV_AC_Export × CI_export`, where `B_u` is
unattributed battery energy delivered to load, `CI` the scalar grid factor
and `CI_export` the export displacement factor. Grid-charge energy and its
round-trip losses enter through `Import`, so time-shifted grid energy earns
nothing and its losses count against the system, the only defensible result
with a scalar annual factor. Without grid charging, `Load − Import − B_u` is
direct PV plus PV-origin battery AC to load, the quantity credited today.
Computing it from those columns plus the grid-origin terms keeps default
results bit-identical rather than equal up to rounding. PV production for LCOE and CO2 is built from
`PV_AC_Export`, not its `Sell_To_Grid` alias.

### Implementation notes for the dispatch-seam PR

These are not decisions. They record where the current step resists A6–A9,
for the seam PR that follows #177 (one dispatch step for both backends).

- `_dispatch_dc_step`'s `charge()` assigns its ledger entries and runs at
  most once per step. PV and grid charge must accumulate into one charge
  input, which A6 also needs for self-heating: the thermal model reads only
  that input.
- Seam points: discharge availability plus a `discharge_allowed[i]` gate, and
  grid charge as a sub-step after PV allocation in both branches of the step.
- Ledger construction is spread across the step's keys, `_LEDGER_COLUMNS`,
  `_STATE_ROWS`, the compiled kernel's row constants, replacement row
  rewrites, column renames, the PV-only summary buffers and the three year
  loops. Consolidate it before adding schema 2.0 columns.
- `_ResultBuffers` and `_PvOnlySummaryBuffers` create columns with `setattr`
  in a loop, so a misspelled new column is not caught. Use explicit fields.
