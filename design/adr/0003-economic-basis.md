# 0003 — Economic basis, escalators, and currency-neutral results

- **Status:** Accepted (E1–E10); implemented in 0.7.0. E11–E13
  Proposed, implemented for 0.7.1
- **Date:** 2026-09-26; E1, E6 and E8 accepted 2026-09-26; E2–E5, E7, E9 and
  the E6 inflation default accepted 2026-09-27; E10 accepted 2026-10-01; E9
  amended 2026-10-01; E11 and E12 proposed 2026-10-03; E13 proposed
  2026-10-03

## Context

The 0.7 delivery plan called for a BREOS economic-basis ADR before tariff
valuation. That plan was completed and removed;
[Tariffs and smart charging](../architecture/tariffs-and-smart-charging.md)
describes the result. The legacy research ADR this record replaces is design
evidence only. #183
records where the current code resists currency, TOU valuation and component
cashflows. This record settles the conventions; it changes no code.

What the code did when this record was written (0.6.x), in
`cost_analysis_projection` (`breos/economics.py`) unless stated:

- One rate, `inflation_rate`, escalates the import price, O&M, the daily
  charge and replacement cost. The CLI help calls it "annual electricity price
  inflation". Only export has its own rate, `sell_price_inflation`.
- Annual energy, O&M and daily-charge flows are priced at year-1 prices
  escalated by `(1 + i)^(n − 1)` and discounted at the end of year `n` by
  `(1 + d)^n`. The entered price is therefore the first project year's price,
  and a year of discounting remains even when `i = d`.
- Replacements are priced at t = 0, inflated to the swap instant and
  discounted from it.
- The daily charge assumes 365 days (`first_year_days = 365`), ignoring leap
  and partial years.
- Nothing says whether the projection's `discount_rate` is nominal or real.
  The standalone `calculate_lcoe` holds O&M at first-year prices, and #251
  documents it as a real-terms (constant-price) LCOE.
- Defaults disagree. The App registry uses discount 0.03 and inflation 0.02;
  `CostParams` and optimization (`DEFAULT_DISCOUNT_RATE`,
  `DEFAULT_INFLATION_ELEC`) use 0.0 and 0.02; the `cost_analysis_projection`
  signature uses discount 0.02 and inflation 0.03; `calculate_lcoe` and
  `calculate_lcoe_from_projection` default the discount rate to 0.0.
  `breos/data/configs/financials.json` says 0.05 and 0.02 but is never read
  at runtime.
- Replacement cost is money inside the physics layer: the App sets
  `BatteryConfig.replacement_cost` from the storage cost per kWh, and only
  optimization can override it (`_replacement_event_cost`).
- Money names carry the currency: 19 distinct `*_eur` identifiers in
  `breos/`, including `total_investment_eur`, `npv_savings_eur`,
  `lcoe_eur_kwh`, `battery_replacement_cost_eur`, Monte Carlo
  `total_replacement_cost_eur` and optimization `*_NPV_Eur` columns. The
  projection columns (`Cost_Import`, `Revenue_Export`, ...) are already
  neutral.
- `battery_replacement_cost_eur` sums t = 0 prices, neither inflated nor
  discounted, beside discounted neighbours.
- There is no result schema version. `LEDGER_SCHEMA_VERSION` covers the
  ledger and loss waterfall only.

## Decision

E1, E6 and E8 were **Accepted** on 2026-09-26. E2–E5, E7, E9 and the E6
inflation default were **Accepted** on 2026-09-27. Accepting them accepted
the design, not its implementation. All nine were then implemented for
0.7.0 under #183: E6 in #271, E5 and E7 in #273, E8 and E9 in #283, E1, E2
and E3 in #288, and E4 in #291. E10, accepted on 2026-10-01, was implemented
in #352. The changelog carries the migration table below as shipped.
E11, proposed on 2026-10-03, is implemented for 0.7.1 under #400, E12,
proposed the same day, under #404, and E13, also proposed that day, under
#376.

### E1. Nominal basis for the projection APIs — Accepted 2026-09-26

The projection APIs escalate cashflows, so their rates are nominal annual
rates and `discount_rate` is a nominal discount rate. They are
`cost_analysis_projection`, `calculate_lcoe_from_projection`, which reads the
escalated projection, the steady-state `calculate_financials`, which mirrors
it, and the App, Monte Carlo and optimization paths built on them. This is
the arithmetic BREOS already performs and how tariff and financing inputs are
usually quoted. (#270 later retired `calculate_financials` with the
steady-state objective basis.)

The standalone `calculate_lcoe` keeps its documented real-terms contract
(#251): it holds O&M at first-year prices, so its `discount_rate` is a real
rate. It is not a projection API and does not change. (0.7.0 later removed
`calculate_lcoe`, which nothing in the package called.)

Provenance records the rates the projection used and the implied real
discount rate, `(1 + d) / (1 + inflation_rate) − 1`. A user can enter real
rates and zero inflation, and the arithmetic runs unchanged. BREOS records
the rates it used, not the basis the user intended, so the record does not
distinguish a nominal study from a real one; that would need an explicit
basis setting, which is not part of 0.7.0. Real-terms outputs are not added
in 0.7.0.

### E2. Separate escalators, defaulting to today's behaviour — Accepted 2026-09-27

`inflation_rate` becomes general inflation. Import energy, the fixed charge
and O&M escalate at it unless their own rate is set. Export keeps its
existing rate. Replacement prices always inflate at `inflation_rate`, and an
explicitly configured learning rate reduces them; learning defaults to 0.0,
not to `inflation_rate`. New keys (spelling settled in the config registry):

| Component | Key | Default |
|---|---|---|
| Import energy and fixed charge | `import_price_escalation` | `inflation_rate` |
| Export energy | `sell_price_inflation` (existing) | 0.0 |
| O&M | `om_escalation` | `inflation_rate` |
| Replacement price reduction (learning) | `replacement_cost_learning` | 0.0 |

A replacement at time `t` years costs `C0 × (1 + inflation_rate)^t ×
(1 − replacement_cost_learning)^t`. With every new key omitted the arithmetic
is today's, so the App golden baseline does not move. The CLI help for `inflation_rate`
changes to match.

### E3. Timing conventions kept and documented — Accepted 2026-09-27

Energy, fixed-charge and O&M flows stay at year-1 prices, booked at year end.
Replacements stay at t = 0 prices, inflated to and discounted from the swap
instant. Initial CAPEX is at t = 0. Changing the energy timing would move
every NPV for no gain in correctness; the user documentation states it,
including the year of discounting that remains when escalation equals the
discount rate.

### E4. Economics prices replacements — Accepted 2026-09-27

The physics layer reports replacement events (instant and replaced kWh);
economics prices them. `BatteryConfig.replacement_cost` leaves the physics
path. Learning rates and price revaluation then need no re-simulation, and
App and optimization price replacements the same way.

### E5. Fixed charge by simulated duration — Accepted 2026-09-27

The daily charge is `fixed_charge_per_day × simulated hours / 24`. A whole
non-leap year is still exactly 365 days, so those results do not change; leap
years and partial runs (#242) are charged for their actual length. The count
does not depend on the index timezone. Per-day charges that differ by day
type use the civil-day array of ADR 0002 A1.

### E6. One default set — Accepted 2026-09-26; inflation default 2026-09-27

The default discount rate is 0.03 everywhere: the value App users already
get. It is defined once, beside `CostParams`, and read by the App registry, `CostParams`, optimization
(`DEFAULT_DISCOUNT_RATE`), Monte Carlo, `cost_analysis_projection`,
`calculate_lcoe_from_projection` and `calculate_lcoe`. For `calculate_lcoe`
the 0.03 is a real rate, under its E1 contract (until 0.7.0 removed it).
`financials.json` is deleted.

Callers who omit the discount rate get different results. `CostParams`,
`cost_params_from_config`, optimization (`calculate_financials` and the
projected objectives) and both LCOE functions defaulted to 0.0, so their
NPVs, LCOEs, objectives and selected designs change; a direct
`cost_analysis_projection` call defaulted to 0.02. App results do not change.
The release notes name these entry points and the old default.

The default inflation rate is 0.02 everywhere, by the same rule. It is 0.02
in the App registry, `CostParams` and optimization today; only the
`cost_analysis_projection` signature uses 0.03, so a direct call that omits
it changes.

An explicitly supplied 0.0 stays valid and is used as given, for either rate.
Defaults apply only when the key or argument is absent, never when its value
is falsy, and a test pins `discount_rate = 0.0` and `inflation_rate = 0.0`
for each entry point.

### E7. Year rows carry money at year-1 prices — Accepted 2026-09-27

The year loop adds import cost, export revenue, fixed charge and no-system
import cost, at year-1 prices, to each year row. Economics applies
escalators, timing and discounting. The flat case computes kWh × price in the
current operation order and stays bit-identical; TOU fills the same columns
from `sum(energy × price)` over the steps. Storing each year's energy by
tariff period, for revaluation without re-simulation, is a 0.7.x follow-up.
*(Done in 0.7.0: a tariff run records its energy by period, and by season
for a schedule with month seasons, and `App.revalue` re-prices from it.)*
*(0.7.1, #393: an App run whose dispatch reads no tariff, greedy or PV-only,
also keeps each year's priced flows step by step, so `App.revalue` prices it
on a first or a different schedule with the year loop's own sums. Energy by
period cannot serve there, and sums over the years cannot either, because
each year is escalated and discounted on its own.)*
The App `financial` rows gain the component cashflows the projection already
computes: `Cost_Import`, `Revenue_Export`, `Cost_Operation`, `Cost_Daily`,
`Cost_Replacement` and `Replacement_Time_Years`.

### E8. Currency-neutral names, no aliases — Accepted 2026-09-26

Money keys drop the currency token. The resolved currency is recorded once in
result metadata and provenance, and plot and export labels read it.
Replacement totals state their basis: `_t0_prices` means t = 0 prices,
neither inflated nor discounted. The App result also gains the discounted
total, `battery_replacement_cost_npv`, beside the renamed one.

The keys are renamed in 0.7.0 with no aliases, following the repository's
removal policy: BREOS removes superseded APIs rather than deprecating them
(#164), and ledger schema 2.0 drops its duplicate columns the same way. This
replaces ADR 0002's "compatibility aliases only while the resolved currency
is EUR". Aliases would give EUR and non-EUR runs different key sets, double
every money field in the App result, the golden fixture and the Monte Carlo
and optimization frames, and still need a removal release later. The cost is
that scripts reading the old names break once. A removed output key is simply
absent; a removed input key (`budget_eur`) raises an error that names its
replacement rather than silently falling back to the default budget.

The payback rename left open by #251 lands in the same change, so users
migrate once. "Exact" overstates a linear interpolation between year-end
points, so the fractional payback becomes "interpolated". It has no aliases
either.

Results that use the new names carry `result_schema_version`; E9 sets where
the field is carried and when it changes. A result without the field
predates the rename.

The migration table below was built by searching `breos/` for `eur`, `Eur`,
`EUR`, `€` and `_exact` at `origin/develop` 79bffcf. It lists every public
result key, column, label and parameter that carries the currency or the
payback "exact". The `breos sweep` CSV repeats the App result's scalar keys,
so rows 1–4 also apply to it.

| # | Surface | Old | New |
|---|---|---|---|
| 1 | `App.result()` | `total_investment_eur` | `total_investment` |
| 2 | `App.result()` | `npv_savings_eur` | `npv_savings` |
| 3 | `App.result()` | `lcoe_eur_kwh` | `lcoe_per_kwh` |
| 4 | `App.result()` | `battery_replacement_cost_eur` | `battery_replacement_cost_t0_prices` |
| 5 | `cost_analysis_projection` `attrs` | `lcoe_eur_kwh` | `lcoe_per_kwh` |
| 6 | Monte Carlo `runs` column, `summary` key | `npv_savings_eur` | `npv_savings` |
| 7 | Monte Carlo `runs` column, `summary` key | `lcoe_eur_kwh` | `lcoe_per_kwh` |
| 8 | Monte Carlo `runs` column | `total_replacement_cost_eur` | `total_replacement_cost_t0_prices` |
| 9 | Monte Carlo `runs` column, `summary` key | `payback_year_exact` | `payback_year_interpolated` |
| 10 | Optimization Pareto column | `NPV_Eur` | `NPV` |
| 11 | Optimization Pareto column | `Objective_NPV_Eur` | `Objective_NPV` |
| 12 | Optimization Pareto column, `objective_names` | `SteadyState_NPV_Eur` | `SteadyState_NPV` |
| 13 | Optimization Pareto column, `objective_names` | `Projected_NPV_Eur` | `Projected_NPV` |
| 14 | Optimization Pareto column | `Projected_Initial_Cost_Eur` | `Projected_Initial_Cost` |
| 15 | Optimization Pareto column | `Projected_Replacement_Cost_Eur` | `Projected_Replacement_Cost_T0_Prices` |
| 16 | Optimization Pareto column | `Projected_LCOE_Eur_kWh` | `Projected_LCOE_per_kWh` |
| 17 | Optimization Pareto column | `Projected_Breakeven_Year_Exact` | `Projected_Breakeven_Year_Interpolated` |
| 18 | Optimization `details["battery_replacement_treatment"]` | `replacement_cost_eur_each` | `replacement_cost_each_t0_prices` |
| 19 | Optimization config, `constraints` | `budget_eur` | `budget` |
| 20 | `breos.economics` function | `find_payback_year_exact` | `find_payback_year_interpolated` |
| 21 | `breos list cost-presets --json` | `electricity_cost_eur_kwh` | `electricity_cost_per_kwh` |
| 22 | `breos list cost-presets --json` | `export_price_eur_kwh` | `export_price_per_kwh` |
| 23 | `breos list cost-presets --json` | `storage_cost_eur_kwh` | `storage_cost_per_kwh` |
| 24 | `breos.io` summary label | `LCOE [EUR/kWh]` | `LCOE [<currency>/kWh]` |
| 25 | `breos.io` summary label | `Total Investment [EUR]` | `Total Investment [<currency>]` |
| 26 | `breos.io` summary label | `NPV Savings [EUR]` | `NPV Savings [<currency>]` |
| 27 | `plot_pareto_front_analysis` input column | `Net_Cost_Eur` | `Net_Cost` |
| 28 | `plot_tariff_comparison` input column | `Net Cost (€)` | `Net Cost` |
| 29 | `plot_tariff_comparison` input column | `No System Cost (€)` | `No System Cost` |

As shipped, the changelog's table omits rows 12 and 18, which went earlier in
0.7.0 with the steady-state objective basis (#270), rows 28 and 29, whose
function was removed (#287), row 27, whose function was removed with the
tool-only plots (#186), and rows 24–26, whose labels went with the
`breos.io` summary helper that produced them. Later in 0.7.0 the optimizer
columns `Projected_Breakeven_Year` and `Projected_Breakeven_Year_Interpolated`
(row 17) became `Projected_Payback_Year` and
`Projected_Payback_Year_Interpolated`.

Rows 24–26 substitute the resolved currency code, so an EUR run writes the
same text as today. (0.7.0 later removed the helper that wrote them.)
Names that are already neutral keep them: the projection
columns (`Cost_Import`, `Revenue_Export`, the `*_NPV` columns),
`attrs["total_investment"]` and `attrs["final_npv_savings"]`. The private
argument `_estimate_battery_replacement_treatment(replacement_cost_eur=...)`
and the locals `payback_exact` (optimization) and `be_year_exact` (plotting)
are renamed too but are not public. Hard-coded `€` and `EUR` in plot axis
labels, the `breos list cost-presets` text output and docstrings read the
currency or say "currency". The changelog carries this table.

### E9. Result schema version — Accepted 2026-09-27

`App.result()`, Monte Carlo summaries and optimization provenance gain a
top-level `result_schema_version`, independent of the ledger schema. It
starts at `"1.0"` with the E8 names. A rename or removal bumps the major
version; an added field bumps the minor.
*(Amended 2026-10-01: `result_schema_version` is a format number, not a
major.minor version. It changes, to the next integer, only when a field is
renamed or removed; an added field leaves it unchanged, the release's
changelog lists it, and `breos_version` identifies the release. No version
had been released, so 0.7.0 ships format `"1"`, the first released format;
results of earlier BREOS versions carry none. The ledger schema is
unaffected.)*

### E10. Optional terminal-health credit — Accepted 2026-10-01

An optional `[terminal_value]` table has one key, `basis`: `"none"`
(default, also when the table or key is omitted) or
`"battery_health_fraction"`. It is an accounting sensitivity, not resale
value. Only the battery pack installed at the end is credited; PV modules,
the inverter and stored energy are excluded. Capacity health omits
resistance-related limits.

Let `h` be the final installed pack's capacity SOH fraction after the last
step's degradation, terminal cycle finalization and any replacement, and
`h*` the physical `battery_eol_percentage` used by the simulation. The
credited fraction is `f = clip((h - h*) / (1 - h*), 0, 1)`. No separate
threshold is introduced. Inputs must be finite and `h* < 1`. At or below
threshold the credit is zero; a fresh pack receives full credit.

At exactly `t = T`, the end of the project horizon in years, the full
replacement-pack price is computed by the existing replacement-outlay
routine (E3/E4): `C0 × (1 + inflation_rate)^T ×
(1 - replacement_cost_learning)^T`. `C0` is the resolved full replacement
price at t = 0, including cost overrides. The nominal credit is this price
times `f`, and its present value is discounted by `(1 + discount_rate)^T`,
the same convention as other year-T flows. This uses exponent T, not T − 1
(the annual energy-flow exponent), and no mid-year fallback.

Every replacement outlay remains. Replacement policy is unchanged: when the
terminal-replacement guard skips the final swap, the old pack remains and
receives zero credit if it is at or below threshold. A zero-capacity or
absent battery reports explicit zero credit when enabled. A partial
`[period]` run has no lifetime economics and reports null credit.

Results gain `terminal_health_credit`, `terminal_health_credit_npv` and
`npv_savings_terminal_adjusted` (unadjusted NPV plus credit present value).
App rounds money only at serialization. When disabled these scalars are
null and terminal-value provenance is absent. When enabled for a lifetime
run, provenance records basis, formula version, final SOH, threshold,
resolved price basis, rates, horizon, timing and replacement policy.
Monte Carlo computes each trajectory before aggregation, reports all three
statistics in its existing style, and records each trajectory's valuation
inputs. Disabled Monte Carlo values are NaN and have no statistics.
`App.revalue` recomputes from retained final health when prices, rates or the
table change, without re-simulating for this sensitivity.

Projected optimization accepts but ignores `[terminal_value]`: evaluated
designs do not report the credit and ranking continues on unadjusted NPV.
Unadjusted NPV, cashflows, paybacks, LCOE, emissions, dispatch and aging are
unchanged even when the sensitivity is enabled.

### E11. Minimum service time of a replacement battery (#400) — Proposed 2026-10-03

Adds a rule; it replaces no text. E3 and E4 book a replacement at its swap
instant at full price, however little of the project is left for the new
pack. `battery_allow_terminal_replacement = false` (#304) skips only the
swap at the close of the horizon's final degradation period, the last day.
A pack that reaches end of life on day 100 of year 20 is still bought and
serves about nine months. In the time-of-use study for the upcoming
publication this happens in 426 of 24 516 Porto R1 cases and in 4482 of
17 700 fixed-target grid-charging runs, and a swap at 19.3 years costs
about 3700 EUR. The study's rule is that a pack is replaced only if at least
one project year remains for it.

- **Key.** `battery_replacement_min_remaining_years` (App and Monte Carlo),
  `[battery] replacement_min_remaining_years` (optimizer) and
  `BatteryConfig.replacement_min_remaining_years`: a finite number of at
  least 0, default 0. A swap is skipped when the project time left after
  it is less than this. A minimum above the horizon skips every swap.
- **Time basis.** The time left is the horizon less the swap's
  `Replacement_Time_Years`: each project year is one simulated span, and a
  swap at the end of step `k` (1-based) of an `n`-step year `y` is booked at
  `y − 1 + k / n`. The horizon is the project's, not the span's: a
  projection year passes the whole years after it to its battery
  (`BatteryConfig.replacement_years_after_span`), so the test is
  `years_after × n + (n − k) < minimum × n`, in steps. A swap with exactly
  the minimum left is kept, which is what "at least one project year
  remained" means at the close of year T − 1. A minimum that is a whole
  number of steps per year has an exact boundary; any other is rounded once,
  in the product.
- **Skipped pack.** The skipped swap is not recorded and not priced. The
  old pack stays installed and keeps ageing below its end-of-life
  threshold, and its state is reported, as with the terminal guard. Every
  later end-of-life check is skipped too, because less time is left.
- **Old key.** `battery_allow_terminal_replacement` keeps its meaning. The
  two are independent tests, and a swap must pass both. The final period
  has no time left, so any positive minimum also skips it; the old key
  matters only at minimum 0. Neither key is deprecated: the old one is the
  exact last-day guard and needs no horizon.
- **Where it is enforced.** At the same end-of-life check as the terminal
  guard, between degradation windows in Python. The compiled dispatch
  kernel is unchanged, so Python and Numba skip the same swaps.
- **Provenance.** The resolved value is in `provenance.resolved_config`
  (App and Monte Carlo), in the optimizer's
  `battery_replacement_treatment` with a `minimum_service` description, and
  in the `[terminal_value]` `replacement_policy` beside
  `allow_terminal_replacement`.

- **Reporting.** The final state of health does not say when a swap was
  skipped or why, so every crossing of the threshold is recorded as an
  event: a degradation period that closes with the installed pack at or
  below `eol_percentage`. Each event holds the project `year`, `time_years`
  (measured as `Replacement_Time_Years` books a swap, so a replaced
  crossing's time is the booked one), the closing step's `date`, the
  `action` (`"replaced"` or `"kept"`), the `reason` (`"end_of_life"` for a
  swap; `"min_remaining_years"`, `"terminal_period"` or, with replacement off
  (`battery_enable_replacement = false`, #377, or a direct `BatteryConfig`),
  `"replacement_disabled"` for a skipped one) and the `soh_pct` the check compared with the threshold. A
  pack that is kept crosses once, so it is recorded once, and a skipped
  swap is always the last event. App results list the events in
  `battery_end_of_life_events` and repeat the first in the scalar
  `battery_first_end_of_life_*` fields, which a sweep CSV keeps. Monte
  Carlo reports the first crossing per trajectory and the projected
  optimizer per design. `SimulationSummary.end_of_life_events` carries
  the span's `breos.battery.EndOfLifeEvent` records, and the detailed
  degradation frame's `attrs` the same events as JSON-safe dicts (pandas
  writes attrs as JSON), so the public return tuple keeps its shape. The
  `date` moves the closing step's date forward by the project year, since
  every year replays the first year's calendar; Monte Carlo restamps its
  sampled weather years to `target_year`. `action` and `reason` are open vocabularies: a new end-of-life
  action adds a value and renames none.

With the default 0 no swap is skipped and results are bit-identical. The
result format stays `"1"`; the key and the event fields add fields and
rename none.

### E12. Retire a battery whose replacement is skipped (#404) — Proposed 2026-10-03

Adds a rule; it replaces no text. Under E11 a pack whose swap is skipped
stays in service and keeps dispatching below its threshold to the end of
the project. That assumes a worn battery is at least as good as none, and it
is not always: a prescribed dispatch, such as grid charging whose conversion
and round-trip losses exceed the tariff saving, loses money every day it
runs. The time-of-use study for the upcoming publication compares three
end-of-life policies for the final project year: replace, keep and retire.

- **Key.** `battery_skipped_replacement_action` (App and Monte Carlo),
  `[battery] skipped_replacement_action` (optimizer) and
  `BatteryConfig.skipped_replacement_action`: `"keep"` (default) or
  `"retire"`.
- **Composition with E11.** The key says what happens at a crossing whose
  replacement is skipped, by `battery_replacement_min_remaining_years`, by
  `battery_allow_terminal_replacement = false`, or by
  `battery_enable_replacement = false` (#377; `[battery] enable_replacement`
  and `BatteryConfig.enable_replacement`), which skips every swap. Whether to buy stays with those rules;
  the new key only chooses the alternative to buying. The three policies
  are then: replace (minimum 0), keep (minimum `m`, `"keep"`) and retire
  (minimum `m`, `"retire"`). A single `battery_end_of_life_action =
  "replace" | "keep" | "retire"` was considered and rejected. Its
  `"replace"` would mean "replace unless the minimum skips it, then keep",
  which is `"keep"` under another name; and a policy that replaces early
  swaps but retires in the final year, the study's case, would need the
  minimum as well, so the three values would not be independent. Never
  replacing is `battery_enable_replacement = false` with either action; a
  minimum above the horizon gives the same run with the reason
  `"min_remaining_years"`. With replacement on and the default minimum and
  terminal guard no swap is skipped, so the key has no effect unless one of
  them is set.
- **Retirement instant.** The crossing: the close of the degradation period
  whose health reached the threshold, the instant a replacement would have
  been booked at (E3, E4). The closing step was dispatched as usual.
- **Operation after it.** The battery neither charges nor discharges, from
  PV or from the grid, whatever the dispatch instructions say. Each later
  step is computed by the PV-only path with the same inverter, so its flows
  equal the PV-only system's bit for bit, and Python and Numba agree. A
  daily controller and a year planner still run, and their instructions are
  recorded, but they move no energy. A retired pack stays off in every
  later projection year (`CarryState.battery_retired`). The retired span's
  degradation state holds `"battery_retired": True`, so a span continued
  from it stays retired; `battery_retired=True` says the same explicitly,
  and is refused with a state from a pack that was not retired. A fresh
  pack that starts at or below its threshold crosses at its first period
  close, so a skipped swap retires it there.
- **Stored energy.** It leaves the system with the retired pack, as a
  replaced pack's does: it is booked in `Battery_Replacement_Energy_Removed`
  with its origins, nothing is added, and the ledger closes. Freezing it in
  a disconnected pack was the alternative; it would report energy that can
  never be used as stored, and as final stored energy by origin, and leave
  it to the capacity window as the health changes. The pack's open
  rainflow half cycles are counted at the crossing, as for a replaced pack.
- **Money.** The investment and every cash flow before the crossing are
  unchanged, and no replacement is bought or priced. The fixed charge and
  O&M continue: BREOS has no separate battery O&M line, so retirement
  saves none. The retired pack is at or below its threshold, so the E10
  terminal health credit is zero, as for a kept pack; a retired pack is
  never replaced later. `[terminal_value]`'s `replacement_policy` records
  the action.
- **Health.** The pack stays installed at zero charge and the aging model
  still runs on it, with no cycles. At zero charge both the native and the
  BLAST engines give little or no further fade: none for the native
  engine, and under 1e-6 percentage points a day for BLAST in the tested
  cases. It is reported as for a kept pack.
- **Reporting.** The crossing is an E11 event with action `"retired"` and
  the reason of the skip.

The default `"keep"` is E11's behaviour, and results are bit-identical. The
result format stays `"1"`.

Follow-ups, not part of E12: a tariff switch from a given project year,
independent of retirement (#402), and an economic rule that compares
replace, keep and retire at each crossing (#403).

### E13. Currency selection (#376) — Proposed 2026-10-03

Adds a rule; it replaces E8's "the resolved currency" source, which in
0.7.0 was the `[tariff]` currency or EUR, and `SUPPORTED_CURRENCIES =
{"EUR"}`. The arithmetic never depended on the currency, but a study in
another currency could only be run by labelling its inputs EUR, and the
EUR cost presets and defaults then mixed silently with them.

- **Key.** A top-level `currency` (App, Monte Carlo, `breos sweep`, CLI
  `--currency`, and the optimizer's top level): an ISO 4217 code, upper-cased.
  `SUPPORTED_CURRENCIES` becomes the active ISO 4217 codes without fund,
  precious-metal, testing and no-currency codes. The check is of the label,
  not of price realism.
- **Resolution.** The run's currency is `currency`, else the `[tariff]`
  currency, else EUR (`DEFAULT_CURRENCY`, the currency of the bundled cost
  catalogue and the `CostParams` defaults). `breos.tariffs.run_currency`
  replaces `result_currency`, which could not see the key.
- **Consistency.** `[tariff].currency` and `[reference_tariff].currency`
  must equal it. Each cost preset records its own `currency` in
  `costs.json`; a preset in another currency than the run's is refused,
  however many of its values `[costs]` overrides. No exchange rates exist.
- **No relabelling.** In a run whose currency is not EUR, every cost the run
  prices with whose `CostParams` default is not zero must be given in
  `[costs]` (for the optimizer, the flat prices may come from
  `[financials]`), or by a preset in the run's currency. The error names
  every missing key. Which costs a run prices with follows
  `calculate_costs`: a PV-only run (or a search whose
  `constraints.max_battery_kwh` is 0) needs no battery costs, and a run with
  a `[tariff]` needs no flat prices. Zero defaults (`land_cost`,
  `other_costs`, `maintenance_cost`, `operation_cost`, a zero
  `wear_cost_per_kwh`) are zero in any currency and stay. The optimizer's
  `constraints.budget` defaults to 10000 EUR, so such a run sets it. The
  tariff, the reference tariff, the annual network credit, an explicit
  `battery.replacement_cost` and `wear_cost_per_kwh` are explicit inputs in
  the run's currency.
- **Carried.** `provenance.currency` (App, Monte Carlo, optimizer),
  `attrs["currency"]` on the cost projection, Monte Carlo runs and the
  optimizer's Pareto frame, as in E8. A CSV keeps no attrs, so the sweep
  CSV, the Monte Carlo runs and yearly CSVs and `write_cost_projection` add
  a `currency` column, and plots read it as they read `attrs["currency"]`.
  `write_cost_projection` takes the label from `attrs["currency"]`, else
  from the frame's own `currency` column, and writes none when the frame
  records neither. `breos validate-config` reports the resolved currency,
  and `breos list cost-presets` each preset's own. `currency` is an
  economics key, so `App.revalue` accepts it, and it is one of the keys the
  input stage never reads, so a sweep over it shares prepared inputs.
  A revaluation that changes the run's currency must restate every money
  input in the new one, since `App.revalue` merges tables key by key and
  checks only the labels of the tariff and the reference tariff: each
  non-zero cost the old `[costs]` set, and each non-zero price list, fixed
  charge and `annual_network_credit` amount of a `[tariff]` or
  `[reference_tariff]` the run keeps, is refused unless the change restates
  it, sets it to None or removes its table. Zero amounts are zero in any
  currency and stay. The cost preset must be removed or one in the new
  currency. A non-zero `wear_cost_per_kwh` is not a revaluation key, so such
  a run is refused; a new App states it in the new currency.
- **Property.** Scaling every money input by k scales every money output by
  k and leaves every physical output and the payback years unchanged. In
  the optimizer, with the budget scaled too, the NPV ranking of the
  evaluated designs and their budget feasibility are unchanged. The search
  itself is not invariant: pymoo sums the positive parts of the budget
  violation (money) and the area and ZEB violations (m² and a fraction)
  unscaled, so k re-orders infeasible candidates against each other, and
  the NSGA-II path, and so the front, can differ between currencies.
  Normalising the constraints would change EUR searches and is left out.
  The tests check the property with k a power of two, for which the floats
  scale exactly, in App (flat, tariff with reference, network credit,
  terminal value, fixed target and daily persistence with a wear weight),
  Monte Carlo and the optimizer, where no candidate violates the area.

A run that sets no `currency` resolves as before, and results are
bit-identical. The result format stays `"1"`: the resolved config gains
`currency`, the CSVs gain a column, and nothing is renamed or removed.

## Consequences

- With no new keys, flat App results match the golden baseline except for
  leap-year and partial runs (E5). Optimization, `CostParams` and LCOE
  callers that omit the discount rate change with the 0.03 default, and
  direct `cost_analysis_projection` calls that omit either rate change with
  the 0.03 and 0.02 defaults (E6).
- TOU valuation, escalator scenarios and replacement learning share one
  valuation step and need no re-simulation for price-blind dispatch.
- Downstream code moves to the neutral names and the interpolated payback
  names once, in 0.7.0, using the E8 table.
- `cost_analysis_projection` is split into valuation, discounting and
  metrics, emissions, and file output, and LCOE and lifetime CO2 are
  computed once rather than again in each runner (#292, under #183).
