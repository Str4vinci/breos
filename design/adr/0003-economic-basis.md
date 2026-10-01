# 0003 — Economic basis, escalators, and currency-neutral results

- **Status:** Accepted (E1–E10)
- **Date:** 2026-09-26; E1, E6 and E8 accepted 2026-09-26; E2–E5, E7, E9 and
  the E6 inflation default accepted 2026-09-27; E10 accepted 2026-10-01; E9
  amended 2026-10-01

## Context

The 0.7 plan calls for a BREOS economic-basis ADR before tariff valuation
(`design/architecture/0.7x-tariffs-and-smart-charging-plan.md`, source-to-target
map). The legacy research ADR it replaces is design evidence only. #183
records where the current code resists currency, TOU valuation and component
cashflows. This record settles the conventions; it changes no code.

What the code does today, in `cost_analysis_projection` (`breos/economics.py`)
unless stated:

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
and E3 in #288, and E4 in #291. The changelog carries the migration table
below as shipped.

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
`breos.io` summary helper that produced them.

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
