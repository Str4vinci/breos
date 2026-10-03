# 0002 — Tariffs are resolved values; smart charging is an instruction layer

- **Status:** Accepted; amendments A1–A17 Accepted, A18 and A19 Proposed.
  Implemented in 0.7.0, A15–A19 in 0.7.1,
  with two additions that no amendment accepted: discharge-only mode and
  calendar-month seasons (see
  [Additions without an amendment](#additions-without-an-amendment)).
- **Date:** 2026-08-20; amendments 2026-09-26; A6 accepted 2026-09-26;
  A1–A5 and A7–A10 accepted 2026-09-27; A11 and A12 accepted 2026-09-30;
  A13 and A14 accepted 2026-10-01; A15–A17 accepted 2026-10-02; A18 and
  A19 proposed 2026-10-03

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

The review used a frozen, committed revision of the historical research
implementation. That code is design evidence, not an implementation source.
Ports are derived from the frozen revision or reimplemented independently
against primary sources. Uncommitted research files and notebook results are
evidence only.

The frozen TOU source's 2027 Portuguese citation needs the directive's date:
[Diretiva n.º 3/2026, de 19 de agosto](https://diariodarepublica.pt/dr/detalhe/diretiva/3-2026-1159784501)
approves the electricity periods. A separate
[Diretiva n.º 3/2026, de 26 de junho](https://diariodarepublica.pt/dr/detalhe/diretiva/3-2026-1138879591)
sets gas tariffs. Cite the August act, Article 2 for the schedules and Article
3 for implementation: BTN bi/tri-hourly meters change over from 1 July through
31 December 2027. A specific installation uses the new schedule from its own
changeover, rather than from one universal date. Spain's 2.0TD schedule
must cite CNMC Circular 3/2020 rather than Royal Decree 446/2023. Supplier
prices are separate, dated inputs and are never implied by those schedules.

## Decision

### Tariff configuration

Omitting `tariff` preserves the 0.5.x flat-price path exactly, including its
EUR interpretation and existing `costs.electricity_cost`,
`costs.electricity_sold_cost`, and `costs.daily_power_cost` inputs.
*(Replaced by A4, accepted 2026-09-27.)*

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

*(0.7.0 also accepts an inline `[tariff.custom_schedule]`, which can have
calendar-month seasons; see
[Additions without an amendment](#additions-without-an-amendment).)*

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
*(Replaced by A5, accepted 2026-09-27.)* Annual import cost and export revenue are sums of timestep energy multiplied by
the resolved price arrays; the no-system baseline uses the same arrays and
calendar. *(Amended by A13, accepted 2026-10-01: a `[reference_tariff]`
prices the baseline instead.)* Annual energy totals remain alongside monetary components.

New monetary names are currency-neutral. Existing `*_eur` results remain
compatibility aliases only while the resolved currency is EUR. *(Replaced by
[ADR 0003](0003-economic-basis.md) E8, accepted 2026-09-26: the `*_eur` names
are renamed in 0.7.0 with no aliases.)* Mixed-currency
inputs fail before simulation. Initial CAPEX, imports, exports, fixed charges,
O&M, and replacements remain distinct annual cashflow components. Simple and
sustained discounted payback are separate outputs; NPV remains the financial
ranking metric. *(Not implemented as written: 0.7.0 reports the sustained
discounted payback, as a whole year and interpolated, and no simple
payback.)*

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
requires a tariff and a positive-capacity battery. *(0.7.0 also has
the experimental `daily_persistence` (A12) and `discharge_only`; see
[Additions without an amendment](#additions-without-an-amendment).)*

`grid_import_limit_w` caps total site import, including simultaneous load. Grid
charging is also bounded by battery charge power and the hybrid inverter's AC
rating. The configured `grid_charge_efficiency` is the AC-to-stored-DC
efficiency and is independent of the DC-to-AC discharge efficiency. *(Replaced
by A6, accepted 2026-09-26.)* The first
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
emissions. *(Amended by A8 and A10, accepted 2026-09-27: initial and
replacement energy form a third, unattributed origin, and avoided emissions
use net exchange.)*

### Boundary and terminal conventions

Normal `App.run()` simulations use `physical_carry`: stored energy, origin
shares, and degradation state flow from one project year into the next and the
initial/final states are reported. Validation or optimisation objectives must
declare a terminal convention. Smart-charging oracle comparisons default to
`cyclic_soc`; free terminal depletion is never an unreported benefit.
*(Implemented differently in 0.7.0: each oracle records its own terminal
rule. The daily-target program buys back the energy the year ends without,
at the cheapest charge-step price. The LP bound leaves terminal energy free,
because a cyclic end state is not a bound on one year's bill. See
[validation tooling](../architecture/tariffs-and-smart-charging.md#validation-tooling).)*

Adding grid-origin flows and component cashflows advances the ledger schema to
2.0. The default greedy path remains numerically compatible, but consumers can
use the schema version to detect the additive origin and valuation fields.
*(Amended by A9, accepted 2026-09-27: schema 2.0 also drops four duplicate
columns. Ledger schema 3.0 then took money out of the ledger, under
[ADR 0003](0003-economic-basis.md) E4.)*

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

Implementation followed a delivery plan that was completed in 0.7.0 and then
removed; [Tariffs and smart charging](../architecture/tariffs-and-smart-charging.md)
describes the result. The gates were:

1. tariff resolution and valuation land before dispatch changes;
2. no-op instruction parity covers native and BLAST degradation;
3. fixed-target charging lands only with per-step conservation and origin
   reconciliation tests; and
4. persistence controllers and perfect-information oracles remain experimental
   or tooling-only and replay every schedule through production physics.

All four gates were met in 0.7.0.

## Implementation status (0.7.0)

The decision and amendments A1–A14 are implemented, with these exceptions
and additions:

- **A1, monthly rows.** Monthly rows group by civil month in the configured
  zone only for a `[period]` run. A full-year run groups them on the result
  frame's own clock.
- **Payback.** No simple payback is reported (see
  [Valuation and cashflows](#valuation-and-cashflows)).
- **Oracle terminal rules.** They differ from `cyclic_soc` (see
  [Boundary and terminal conventions](#boundary-and-terminal-conventions)).
- **A9, grid-charge cost.** Ledger schema 3.0 (ADR 0003 E4) took money out of
  the ledger. The grid-charge cost is a year-row and result value at year-1
  prices (`grid_charge_cost_year1_prices`), not a ledger column.

### Additions without an amendment

Two features were implemented without an amendment of their own. They are
recorded here as the 0.7.0 contract. No amendment accepted them, so they
carry no acceptance date. A later change to either
one needs an amendment, as for any other part of this decision.

**Discharge-only mode** (#338, PR #341). `smart_charging.mode =
"discharge_only"` lets the battery discharge on a step whose tariff period
is in `discharge_periods` and holds its charge on every other step.

- The grid never charges the battery. PV may charge it on every step.
- `discharge_periods` is required. Every grid-charging key is refused:
  `target_usable_fraction`, `charge_periods`, `grid_charge_efficiency` and
  `grid_import_limit_w`, and also the `daily_persistence` planner settings.
- It resolves to the same instruction arrays as `fixed_target`: a zero
  reserve and no grid target on any step. Discharging in every period is
  the same as greedy dispatch.
- Because the grid never charges the battery, A8's disjoint-period rule
  does not apply. The mode refuses `overlap_policy = "hold_target"` (A14).
- App, Monte Carlo and the projected optimizer accept it, and
  `provenance.smart_charging` records it.

**Inline schedules with calendar-month seasons** (PR #327; seasons #337,
PR #340). `[tariff.custom_schedule]` defines a schedule in the
configuration, with the same strict validation as a bundled one.

- Optional `seasons` maps each season name to its calendar months. Every
  month must be in exactly one season, so a quarter is a season of three
  months. A season name must not also be a period name.
- A step's season is the month of its civil date in the schedule's zone.
  With month seasons, a rule selects a season name or `all`, never
  `standard` or `dst`. Every day type and season must match exactly one
  rule. A holiday takes the rule of `holidays.day_type` in the season of its
  month.
- Import and export prices can be given per season and period. Each season
  must be priced, and each season prices exactly the periods it uses, or
  `all`. A seasonal reference tariff (A13) follows the same rules.
- Every project year replays the start-year calendar (A2). The bundled
  schedules are unchanged and have no month seasons.

## Amendments for 0.7 readiness

The 0.7 readiness audit (#187) found details the decision above leaves open
and statements the code has since outgrown. A6 was **Accepted** on 2026-09-26,
A11 and A12 on 2026-09-30, A13 and A14 on 2026-10-01, A15 on 2026-10-02 for
0.7.1, and every other amendment below on
2026-09-27. Each one replaces the text it names, and that text is marked in
place above; A11, A12 and A15 add rules and replace none. Accepting A6–A10 accepted the
design for the dispatch-seam and ledger work, not its implementation. Grid
charging, origin accounting, ledger schema 2.0 and net-exchange emissions were
then implemented for 0.7.0 in #279–#282 (#178). Economic
conventions and money naming are in
[ADR 0003](0003-economic-basis.md).

### A1. Civil time comes from the configuration, not the index (#180) — Accepted 2026-09-27

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
test before it moves tariff periods. *(Done in
`tests/test_result_index_timezone.py`.)*

### A2. Project years replay the start-year calendar (#180) — Accepted 2026-09-27

Every year loop replays one year of inputs. The App reuses its `start_year`
load and PV series each year, projected optimization repeats one weather year
with one load series, and Monte Carlo restamps each sampled weather year to
its target year. The tariff follows the same rule in 0.7.0: it is resolved
once on the simulated calendar and reused for every project year. Weekday
patterns, holidays and effective dates do not advance. *(Amended by A16,
accepted 2026-10-02: the calendar still replays, but static instructions may differ by
project year.)*

Advancing calendars needs per-year load, PV and tariff construction, which
belongs to the shared projection loop of #179, and the inputs do not support
it yet: weather is a typical year and the standard load profiles are built on
one calendar. The 2027 Portuguese schedules are selected explicitly, never by
date, so one calendar does not silently misprice them. Provenance records
`calendar_policy = "replay_start_year"` and the calendar year: every project
year replays the start-year calendar, and future weekdays, holidays and
tariff effective dates do not advance. A leap `start_year` makes every
project year 366 days.

### A3. Half-hour boundaries need 15-minute input (#180) — Accepted 2026-09-27

0.7.0 accepts hourly and 15-minute input only (`h` and `15min`). Config
validation and `utils.get_hours_per_step` accept only those, and 30 minutes
would also need load-profile, weather-resampling and kernel support. It is not
needed for exactness: 15-minute steps represent every half-hour boundary, so
exact half-hour tariff boundaries require 15-minute input. Under
`boundary_policy = "strict"`, hourly input with half-hour boundaries is
rejected. 30-minute input is later work.

### A4. The flat-path baseline is the App golden fixture (#187) — Accepted 2026-09-27

Replaces "preserves the 0.5.x flat-price path exactly". Unreleased fixes,
including replacement booking at the swap instant, have already moved flat
results away from 0.5.x. The baseline is `tests/fixtures/app_golden`, written
by `tools/generate_app_golden.py` and checked by `tests/test_app_golden.py`.
Omitting `tariff` must leave it unchanged. A change that moves it regenerates
it in its own commit, with the reason.

### A5. Valuation reaches every year loop (#179) — Accepted 2026-09-27

Replaces "valued inside the existing App year loop". Projects are simulated in
three loops: `run_app_simulation`, Monte Carlo's `_simulate_trajectory`, and
`_evaluate_projected_design_metrics` for projected optimization. Price-weighted
import cost, export revenue and fixed charge are computed once, in the shared
projection loop #179 introduces. Until an entry point uses that loop, it
rejects a `tariff` or `smart_charging` table rather than valuing at flat
prices. *(In 0.7.0 all three entry points use the shared loop in
`breos/projection.py`.)*

`objective_basis = "steady_state"` was to reject a tariff or smart charging
until #179 retired `calculate_financials`, which valued one year at scalar
prices. #179 removed both, so the case no longer arises: the optimizer scores
over the projected lifetime only.

### A6. Grid-charge conversion and shared limits (#178) — Accepted 2026-09-26

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
  PV AC output; and
- site: grid-charge import is at most `grid_import_limit_w` minus the step's
  load import.

The inverter limit is an explicit, conservative modelling assumption for 0.7:
summed AC throughput. In each step, PV AC output and grid-charge AC input
both count against the inverter's AC rating, so their sum never exceeds it.
A real hybrid inverter may net the two inside the converter, sending PV DC to
the battery while the grid serves the load, and so charge more than this rule
allows. The assumption can understate grid charging but never exceeds the
rating.

Netting is deferred because it needs three things 0.7 does not define: a
converter topology that says which paths share which power stage, a loss
calculation for the netted flows, and origin accounting for energy that is
redirected rather than converted. Subtracting one AC flow from the other
before applying the rating would supply none of these.

### A7. The target moves with temperature and health (#178) — Accepted 2026-09-27

`target_usable_fraction` applies each step to that step's capacity window: the
target energy is `emin + f × (emax − emin)`. `emin` and `emax` scale with the
temperature capacity factor every step and with SOH every day
(`_apply_capacity_window`), so the target is a fraction of what is usable now,
not a fixed energy. A pack charged to 0.5 on a warm night reads a different
fraction as it cools, and fade lowers the target energy over the project.
Results report the configured fraction and the stored energy reached. With a
zero instruction the target is `emin` exactly, which keeps no-op parity
reachable.

### A8. Stored energy has three origins (#178) — Accepted 2026-09-27

Stored energy is PV origin, grid origin and an unattributed remainder. The
initial energy of a fresh run (full at max SOC) and a replacement pack's
energy are unattributed. Grid origin is therefore explicit state, never
derived as `E − E_pv`. Discharge and standby, capacity-window and replacement
removal take from all three in proportion to their shares at the start of the
operation. Only PV-origin discharge counts as self-consumption, as today.

`charge_periods` and `discharge_periods` must be disjoint, so every step
either charges or discharges. That keeps exact the single origin fraction the
step takes before dispatch; the step asserts it.
*(Amended by A14, accepted 2026-10-01: opt-in overlap retains the target as
the discharge floor; one direction per step remains required.)*

### A9. Ledger schema 2.0 reconciles origins from output (#178) — Accepted 2026-09-27

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

### A10. Avoided emissions use net exchange (#178) — Accepted 2026-09-27

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

### A11. Controllers decide civil days; aging stays positional — Accepted 2026-09-30

Controller decisions are made at configured-timezone civil-day boundaries. If
a civil boundary falls inside a positional degradation window, dispatch is
split into subcalls and carries energy plus PV/grid origins between them;
degradation still closes once on its existing positional window. A controller
at a shared boundary observes the post-aging/post-replacement state. The
window close uses the `Battery_Energy_Beginning` ledger value from the
subcall containing its last step.
*(Extended by A19, proposed 2026-10-03: a controller may also decide on
steps it marks inside a civil day.)*

### A12. Daily persistence: warm start and forecast-terminal value — Accepted 2026-09-30

The experimental App mode `daily_persistence` is a daily controller under
A11. It keeps the fixed-target instruction layout and chooses one grid-charge
target per configured-zone civil day. A decision reads the known tariff, the
battery's measured state and the last complete observed local day, and never
the current or a future day's PV, load or temperature.

- **Forecast** (`repeat_previous_complete_local_day`). The last complete
  local day's PV DC, load and input temperature are repeated on every day of
  the planning window, slot by local wall-clock time and DST fold, never by
  position. A repeated fall-back slot the observed day lacks reuses the
  observed sample at that wall time; a spring-forward slot the observed day
  lacks is interpolated between its neighbouring wall times. Tariff labels,
  prices and day offsets are used as resolved, including the A2 seam.
- **Warm start** (`no_grid_until_one_complete_local_day`). Until one
  complete local day has been observed, a decision keeps the configured
  discharge gate and reserve and sets no grid target. A partial first day
  and a clipped `[period]` edge are not observations. The observation and
  any day decision in progress carry across the A2 year seam; a standalone
  `[period]` never joins its end to its start.
  *(Amended by A18, proposed 2026-10-03: under `hold_target` a step in both
  period lists has no floor, reserve 0, while there is no grid target.)*
- **Rolling solve.** Each day builds a fresh daily-target problem on the
  forecast from the measured stored energy, the current state of health and
  the resistance-adjusted efficiencies, which stay fixed inside the solve.
  Only the first day's target is executed; the next day is planned again
  from the simulated state.
- **Forecast-terminal value** (`preserve_start_energy`). Each solve's
  terminal target is the day's starting energy, capped at
  `nominal_energy_wh * soh * max_soc * capacity_factor(T)` for the final
  forecast temperature `T`, with no free terminal. Stored energy that ends
  the window below the target is priced at the cheapest import price of a
  step that may grid-charge (of any step when none may), through the
  grid-charge and battery charge efficiencies. The same rule applies at the
  project's final horizon. It is a planning penalty, not a dispatch
  instruction or a physical reset: App keeps `physical_carry` and reports
  the stored energy by origin at the start and end of the project.

The terminal price is a tractable continuation proxy, not a learned value
function; it is biased when recharge prices beyond the window differ from
those inside it. Monte Carlo and projected optimization, which share one
set of static instructions across trajectories or candidates, refuse the
mode. Price-aware dispatch forces simulation, so `App.revalue` re-simulates
when the per-step prices change. Results record the policy in
`provenance.smart_charging`, with a hash of the instructions every project
year executed, and no forecast or per-day target.
*(Extended by A17, accepted 2026-10-02: an optional wear cost in the
planner's objective.)*

### A13. The no-system baseline can have its own reference tariff (#339) — Accepted 2026-10-01

Amends "the no-system baseline uses the same arrays and calendar". An
optional top-level `[reference_tariff]` prices the household without the
system, independently of the system's `[tariff]` or flat costs:

- **Baseline.** Each year's no-system cost is the reference import price
  times the whole household load, summed over the steps, plus the reference
  fixed charge: `Cost_No_Sys_Annual = (Baseline_Import_Cost +
  Baseline_Fixed_Charge) × (1 + e)^(n − 1)`. The fixed charge is billed as
  the system's is: on the simulated duration, or on a `[period]` window's
  civil days. `reference_tariff.fixed_charge_per_day` is required; an explicit
  0 is valid when the household pays no fixed charge. The reference has no
  export prices.
- **Escalation.** `e` is `reference_tariff.import_price_escalation`, which
  escalates both components. Absent, it is the system's resolved import
  escalation; an explicit 0 is kept.
- **Absent.** Without a `[reference_tariff]` the baseline is the system's
  prices and fixed charge, as before, and results are bit-identical.
- **Scope.** The reference must be in the result's currency, is checked for
  timezone and resolution as the system tariff is, and is resolved once on
  the simulated index; every project year replays the start-year calendar
  (A2). A reference without a schedule is one flat price, `import_prices =
  { all = x }`. It never drives dispatch and never prices the system's grid
  flows.
- **Revaluation.** `App.revalue` re-prices a reference that is added,
  changed or removed from the retained household load, and never
  re-simulates for it, also under `daily_persistence`.
- **Entry points.** Monte Carlo resolves the reference once per study and
  prices each trajectory's own sampled load, so the paired differences stay
  consistent. The projected optimizer's NPV objective uses it. Choosing the
  cheapest reference a household is eligible for is a study decision, not a
  BREOS feature.

Results report the baseline components with or without a
reference, and `provenance.reference_tariff` only when one is configured.

### A14. Overlapping periods hold the target (#347) — Accepted 2026-10-01

Amends A8's requirement that `charge_periods` and `discharge_periods` be
disjoint. `smart_charging.overlap_policy` is `"reject"` by default, preserving
the existing validation and dispatch bit for bit. `"hold_target"` permits
shared periods under `mode = "fixed_target"`:

- On a step in both lists, `reserve_fraction = grid_target_fraction`. The
  target is also the discharge floor: above it the battery may discharge
  down to it; below it the grid may charge up to it. Non-overlap steps are
  unchanged. PV may charge above the target on every step.
- Both fractions use A7's same current usable-energy mapping, including
  temperature, health and replacement. Binding overlap discharge and grid
  charging land exactly on the bound, preventing rounding residues from
  causing tiny purchases or discharges on the next step. The existing
  disjoint dispatch arithmetic remains unchanged.
- The instruction invariant is that a step allowing discharge with a finite
  grid target has `reserve_fraction >= grid_target_fraction`. Production
  physics still forbids grid charging after discharge or while exporting
  PV, and asserts that no step both charges and discharges. The single
  pre-dispatch origin share remains exact for all three origins.
- `disabled` and `discharge_only` refuse `hold_target`, since they have no
  grid target. `daily_persistence` refuses it because its planning path
  replaces targets while retaining fixed reserves; it cannot plan and replay
  the same held-target instructions. Supporting it later requires the floor
  to follow each candidate and executed daily target, including warm start.
  *(Replaced by A18, proposed 2026-10-03: the floor follows each daily
  target, and `daily_persistence` accepts `hold_target`.)*
- App, Monte Carlo and projected optimization accept `hold_target` for
  `fixed_target`. Python and Numba execute the same kernel source. Results
  record `overlap_policy` in `provenance.smart_charging`; the
  instruction hash already covers the reserve and target arrays. The ledger
  schema does not change.

## Amendments for 0.7.1

These amendments follow the 0.7.0 release. Each one names the text it amends.

### A15. An annual network credit capped at the household's own network charges (#375) — Accepted 2026-10-02

Adds a rule; it replaces no text. Some network tariffs reduce a
household's network charges by an annual amount that cannot exceed what the
household paid for the network that year. Stromnetz Berlin's §14a EnWG
Modul 1 is one: 146.58 EUR a year, gross, capped at the connection's
network charges. A household with PV and storage imports less, so its cap
can bind where the same household without the system gets the full
amount. An optional `[tariff.annual_network_credit]` table, and the same
table under `[reference_tariff]`, express it:

```toml
[tariff.annual_network_credit]
amount_per_year = 146.58
network_fixed_per_year = 39.70
network_import_prices = { low = 0.0311, standard = 0.0888, high = 0.1659 }
```

- **Keys.** All three are required; an explicit 0 is valid.
  `amount_per_year` is the annual reduction. `network_fixed_per_year` is
  the network part of the fixed charge. `network_import_prices` is the
  network part of each import price per kWh, in the shape of
  `import_prices`: by period, or by month season and period, with `all` as
  the fallback. A reference without a schedule gives `{ all = x }`.
- **Gross parts, never added.** Every value is a part of a price the
  tariff already sets, with the same taxes. The network price of every
  period is at most its import price, and `network_fixed_per_year` is at
  most 365 days of the fixed charge; validation refuses anything else.
  Neither is added to the bill: the import cost and the fixed charge are
  priced as before, and the parts only set the cap.
- **Cap basis.** A household's eligible network charges for year `y` are
  its grid import times the network price, summed over the steps, plus
  `network_fixed_per_year × f`. The credit is `credit_y = min(amount_per_year
  × f, eligible_y)`. `f` is 1 for a simulated year, a leap year too, and
  `d / D` for a `[period]` window of `d` civil days in a year of `D` days:
  the amounts are annual, so they are pro rata by civil days, and a full
  year gets exactly the annual amount.
- **Households.** The system household's credit comes from
  `[tariff.annual_network_credit]` and its own grid import. The no-system
  household's credit comes from `[reference_tariff.annual_network_credit]`
  and the whole household load when a `[reference_tariff]` is set (none
  if that reference has no table), and otherwise from the `[tariff]` table
  and the whole load, as A13 prices it at the system's prices.
- **Escalation.** The amount and the cap escalate at the tariff's own
  import escalation: the system's import escalation for the system
  household, and A13's reference escalation for the no-system household.
- **Cash flows.** Each credit is booked in its household's yearly cash
  flow, `Cost_System_Annual` or `Cost_No_Sys_Annual`, before discounting,
  so NPV, payback and every comparison include it. LCOE excludes it, as it
  excludes the other tariff outcomes.
- **Dispatch.** The credit never drives dispatch. Smart charging and the
  experimental `daily_persistence` planner see the per-step prices only,
  also when a binding cap makes extra network charges free at the margin.
- **Entry points.** App, Monte Carlo (each trajectory from its own import
  and load), the projected optimizer objective and `App.revalue` apply it.
  `App.revalue` re-prices a credit added, changed or removed from the
  retained energy by period and the retained household load, and never
  re-simulates for it. The credit is pricing only, so it does not change a
  Monte Carlo year cache.
- **Results.** A household with a credit gets its eligible network charges
  and its credit in the year rows, the cost projection and each App
  `financial` row, and `provenance.tariff.annual_network_credit` or
  `provenance.reference_tariff.annual_network_credit` records the table.
  Without the table, results are unchanged bit for bit, and the result
  format stays "1".
- **Scope.** This is not a billing engine. Metering, control-box and other
  per-connection charges stay outside BREOS; a study adds them to the
  fixed charge or leaves them out.

### A16. Instructions per project year (#383) — Accepted 2026-10-02

Amends A2 for static instructions only; the calendar still replays.

The projection takes static instructions in one of three forms, through
`project_years`, `run_projection` and `run_app_simulation(instructions=)`:

- **One set** for every year: today's path, unchanged and bit-identical.
- **One set per project year**: a sequence with exactly one set for each
  year, all on the replayed calendar. A sequence of the same set is the one
  set, bit for bit.
- **A year planner**: a function called once as each project year begins,
  which returns that year's set. It receives a `YearStart`: the year index,
  the year's inputs with its PV degradation applied, the year's battery
  configuration, and the state its first step dispatches from. That state
  is the carried stored energy and origins, and the state of health,
  resistance growth and efficiencies after the degradation engine has
  restored them. It is the state a daily controller's first decision of
  the year would see (A11).

The seam is a planner hook at each year start, not only a precomputed list.
A list cannot express a plan that depends on the state earlier years left,
because that state exists only once the production run reaches it. A
planner can, in one pass. Planning outside the loop instead would need a
separate projection for each year, or a second copy of the year loop's
carry. The hook is called inside the simulation core after the opening
state is restored, so the planner reads the dispatch's own values and does
not repeat the core's rules for the native and BLAST engines. A list is
still accepted, because a schedule that is planned once is easier to
replay and compare as data.

Rules:

- A planner runs on per-step years with a battery, as a daily controller
  does; summary years accept one set or a sequence. Static instructions and
  a daily controller cannot be combined.
- Health changes inside a year as before. The planner sees only the year's
  opening state.
- The run records the set each year dispatched on
  (`ProjectionRun.year_instructions`). The replay tool records each year's
  instruction hash.
- App configuration does not change. No config key selects per-year
  instructions; they are for validation tools.

The daily-target oracle uses the planner for its `yearly` planning mode.
Year `y` is planned with perfect information on that year's PV, load and
temperature, at the state the production replay of years `1..y-1` reached.
The replay then runs year `y`. The stored energy that ends a year below its
opening energy is bought back, capped at the max-SOC energy at the year's
last temperature. Because of that cap, year one is planned as the
first-year mode plans it. The result is perfect-information daily targets
on a grid. It is not a bound on lifetime NPV: each year is chosen at its
opening health, and the choice ignores what the year's cycling costs later
years.

### A17. Daily persistence: an optional wear cost in the planner (#381) — Accepted 2026-10-02

Adds a rule to A12 and replaces none. A12's rolling solve minimises import
cost less export revenue plus the forecast-terminal value, with no term for
battery ageing. `smart_charging.wear_cost_per_kwh`, a `daily_persistence`
planner setting, adds one:

- **Term.** Each day's stage cost in the shared daily-target solver gains
  `wear_cost_per_kwh` times the DC energy the battery discharges that day
  (`Battery_Discharge_DC` of the planner's own dispatch, in kWh). It counts
  every discharge, whatever the stored energy's origin. Charge is not
  counted: energy still stored at the end of the window has not cycled, and
  the A12 terminal value prices it as energy only. Over a full cycle,
  charge and discharge throughput differ only by the losses, so their
  mean would add little but a charge on energy the window never uses.
- **Default.** 0, which leaves every decision and result bit-identical to
  0.7.0. The value must be finite and at least 0. `fixed_target`,
  `discharge_only` and `disabled` refuse the key, as they refuse the other
  planner settings.
- **Scope.** It is a planning weight in the tariff's currency, not a
  degradation model. It moves only the targets the planner picks. It is
  not a cash flow: the import cost, export revenue and NPV are unchanged,
  and realised ageing and replacement still come from the degradation
  model. BREOS does not derive a value; the user documentation shows one
  estimate, a replacement pack's price over its usable discharge throughput
  to end of life.
- **Callers.** `solve_daily_targets` takes it as `wear_cost_per_kwh`, so
  the offline oracles that use the same solver can set it. The plan reports
  it as `wear_cost`, apart from `stage_cost`, so a replay compares the
  plan's money with production's money alone.
- **Provenance.** `provenance.smart_charging.wear_cost_per_kwh` records the
  resolved value of every `daily_persistence` run, 0 included.
  `planner_version` stays `"1"`: the default planner decides as before.

### A18. Daily targets hold the floor on shared periods (#398) — Proposed 2026-10-03

Replaces A14's refusal of `hold_target` under `daily_persistence`; the
rest of A14 stands. A14 made the grid target the discharge floor on a step
in both period lists, but the daily-target planner replaced the target and
kept the reserve. A day target above the layout's reserve then broke the
instruction invariant, and one below it planned a floor that was not the
held target. Always dispatch with off-peak charging (discharge in every
period, charge off-peak, hold the target) could not be planned.

- **The floor follows the target.** `daily_target_instructions`, which
  places one target per day on the fixed-target layout, also sets the
  reserve to that day's target on every step that allows discharge and has
  a grid target. Other steps keep their reserve. Every caller places
  targets through it: each candidate the shared solver evaluates, the day
  `daily_persistence` executes, its warm start, and both planning modes of
  the daily-target oracle, the yearly planner (A16) included. Planning and
  production replay then run the same held-target instructions.
- **A day with no target.** On a day with no grid target (NaN), the
  reserve on those steps is 0, so the battery may discharge to the minimum
  SOC, as on a target-0 day. With no target there is nothing to hold. The
  layout's own reserve on such a step is only the target the layout was
  built with: the configured one for the oracle, a placeholder of 1 for
  `daily_persistence`, which would forbid discharge in the shared period
  for the whole day. The `daily_persistence` warm start
  (`no_grid_until_one_complete_local_day`) is such a day, so under
  `hold_target` it discharges in every discharge period, shared ones
  included. That is one day for a run that starts at local midnight and two
  when the first day is partial; the observation then carries across project
  years and replacements, so the warm start does not recur. The planner
  never chooses NaN; its candidates are finite.
- **What holding the target means.** The floor holds the energy up to the
  day's target, not all the stored energy. Above the target the battery may
  discharge to the load in a shared period, whatever charged it. A policy
  that never discharges off-peak is a discharge restriction, a period left
  out of `discharge_periods` under `reject`, not `hold_target`. With a
  target that changes from day to day, a shared period that crosses
  midnight holds the previous day's target before midnight and the new
  day's after it. When the new target is lower, the energy above it can
  include grid charge bought the evening before, and the battery may
  discharge it in the same shared period. The planner prices this in the
  costs it compares; it is a property of the policy, not of the dispatch
  step.
- **Configuration.** `daily_persistence` accepts
  `overlap_policy = "hold_target"`, in App configuration and in a directly
  constructed `SmartChargingSpec`. `disabled` and `discharge_only` still
  refuse it. `provenance.smart_charging` records `overlap_policy` as for
  `fixed_target`, and the instruction hash covers the executed reserves.
  `controller_version` and `planner_version` stay `"1"`: a configuration
  accepted before decides as before.
- **Unchanged.** Under `reject` no step both may discharge and has a grid
  target, so every reserve, plan and result is bit-identical. The dispatch
  kernel is unchanged; Python and Numba run the same held-target step as
  under `fixed_target`. The result format stays `"1"`.

### A19. Daily targets can be decided at the start of each charge window (#406) — Proposed 2026-10-03

Extends A11 and A12 and replaces neither. Under A12 `daily_persistence`
picks one target per civil day, decided at local midnight. Off-peak windows
usually cross midnight: under Bi-hourly, 22:00–08:00 is split between two
targets, and the 22:00–24:00 part charges toward a target chosen for a day
whose peak it does not serve. Under `hold_target` (A18) the floor also
changes at midnight, and a lower new target can release the grid energy the
battery bought the evening before. A household controller sets its target
when charging starts.

- **Configuration.** `smart_charging.decision_boundary`, a
  `daily_persistence` planner setting: `"civil_day"`, the default, keeps
  A12 bit for bit; `"charge_window_start"` decides one target per charge
  window. `fixed_target`, `discharge_only` and `disabled` refuse the key, as
  they refuse the other planner settings.
- **Windows.** A charge window is a maximal run of consecutive steps whose
  period is in `charge_periods`. Adjacent charge periods therefore form one
  window, a day may hold several windows (the Saturday of the Portuguese
  weekly cycle has three), and a window may cross midnight or run for days
  (off-peak from Saturday 22:00 to Monday 07:00 is one window). A window
  starts at a charge step that does not follow one; the first step of a
  replayed calendar follows the calendar's last step (A2). Windows are
  local wall-clock periods of the resolved tariff, so they keep their
  local start across DST. With every tariff period a charge period a
  window would never end, and the boundary is refused.
- **Decision and hold.** The target is decided at the window's first step
  and holds on every charge step until the next window starts. Under
  `hold_target` the floor is that one target for the whole window, across
  midnight. Steps before the first window start the run sees have no
  target, as in the warm start.
- **Controller seam (A11).** A controller may name, per step of the
  resolved calendar, where it decides inside a civil day. The session then
  ends a dispatch segment at each marked step and asks the controller there
  for the rest of the logical day, with the battery state the segment left.
  Civil-day decisions, observations, aging windows and the A2 seam carry
  are unchanged; a marked decision inside a day carried across the seam
  replaces the carried instructions from its step. A civil-day start that
  is not a window start keeps the target in force. A controller that marks
  no step decides exactly as before.
- **Causality and forecast (A12).** A window decision sees the known
  tariff, the battery state at the window start, and observations made
  before it only. Its forecast
  (`repeat_local_day_before_decision`) repeats, slot by wall time and fold,
  each slot's most recent observation: the current day so far, then the
  last complete local day. When that is the day before, the forecast is the
  local day that ends at the decision (23 or 25 hours across a DST change).
- **Warm start** (`no_grid_until_a_window_after_one_complete_local_day`).
  A window that starts before one complete local day has been observed gets
  no grid target and holds none until the next window starts. A run that
  begins at local midnight plans its first target at the first window start
  after its first day ends; one that begins mid-day, whose first day is not
  an observation, a day later. The day before a decision can be fully
  observed before that; it is not used, so the observation unit stays the
  civil day.
- **Rolling solve.** Each planning stage is one window, from its start to
  the next window's start. A solve covers the windows that start before the
  decision's wall-clock time `forecast_horizon_days` civil days later, the
  decision's own window first, cut where the known tariff ends. For a
  window every day at 22:00 that is `forecast_horizon_days` days, as under
  civil days. The session supplies two more days of known tariff for the
  last window's run to the next window start. Only the first window's
  target is executed; the next window is planned again. The A12 terminal
  value and the A17 wear cost apply unchanged.
- **Oracle and solver.** `DailyTargetProblem.from_tariff` and the
  daily-target oracle (`--decision-boundary`) take the same boundary, so
  each planning day is a window, from one window start to the next, and the
  steps before the first window form a day of their own. The oracle's
  yearly mode plans each project year on its own, so a window across the
  year seam takes a new target at the year start.
- **Provenance.** `provenance.smart_charging.decision_boundary` records the
  boundary of every `daily_persistence` run. Under `charge_window_start`
  `controller_version` and `planner_version` are `"2"` and the forecast and
  warm-start policies carry the names above; under `civil_day` everything
  stays as in A12. The result format stays `"1"`.
- **Unchanged.** The dispatch kernel. Every configuration accepted before
  decides, dispatches and reports as before; App goldens are bit-identical.
