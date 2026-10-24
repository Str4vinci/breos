# Collective self-consumption

`breos.Community` simulates separately metered households sharing PV at one
location. A household's own PV and battery serve it first, exactly as in an
`App` run. Sharing is settled afterwards on each meter's 15-minute import
and export. It changes supplier bills, not equipment dispatch.

A community can include households without PV and shared PV-only assets.
Assets have zero household load and no battery. Their actual imports, such
as inverter night draw, and supply-contract fixed charges are still billed.
Each household keeps its own supplier prices or tariff calendar.

## Run a community

Like `App`, the constructor takes a configuration mapping. Load a TOML file
in Python first; the single-site `breos run` command does not run communities.

```python
import tomllib
from breos import Community

with open("configs/examples/community-portugal.toml", "rb") as stream:
    community = Community(tomllib.load(stream))
community.simulate()
result = community.result()
ledger = community.settlement()

home = result.participants["apt-1"]
print(home.npv_savings, home.npv_alone, home.community_benefit)
print(result.shared_coverage_of_import)
print(result.shared_coverage_of_load)
print(result.provenance["settlement_schema_version"])
```

Weather is loaded as for App. For an offline study set `weather_file` in
`[community]`; see [Offline weather](../how-to/offline-weather.md). This example
does not bundle weather. Its prices and equipment costs are illustrative
inputs, not a supplier offer or regulated tariff.

`result()` returns a typed `CommunityResult` with unrounded account tables.
`settlement()` returns a new DataFrame for the first project year, in Wh per
quarter-hour. Both raise `RuntimeError` before `simulate()`. Simulating again
replaces the previous result. First-year interval frames and every year's
totals are kept; full-horizon physical arrays are released after pricing.
The [ledger and results reference](../api/community-settlement.md) defines
the columns, accounts, closure and retention.

## Configuration tables

This complete Porto example has three households, one shared roof and a
battery behind `apt-1`'s meter. `apt-2` has no PV. It uses current PV defaults,
including `solar_position = "auto"`, without restating them.

```{literalinclude} ../../configs/examples/community-portugal.toml
:language: toml
```

Download {download}`community-portugal.toml <../../configs/examples/community-portugal.toml>`.

### The community table

`location`, `resolution = "15min"`, `allocation` and all three shared prices
are required. The shared prices are in the community currency per kWh; an
explicit zero is valid. Other clock and economic keys use App's defaults.

| Key | Meaning |
|---|---|
| `allocation` | `"fixed"` or `"proportional"`; see the allocation rules below. |
| `pool_participant_export` | Pool households' metered export, default `true`. Set `false` to keep it sold at each household's own export price. |
| `shared_energy_price` | Paid by receivers for shared energy actually used, and received by injectors. |
| `shared_network_charge` | One community-wide charge on shared use, paid outside the community. |
| `surplus_price` | Sale price for community surplus, attributed to injectors. |
| `schedule` or `custom_schedule` | Calendar for shared price lists, using the [tariff schedule](configuration.md#time-of-use-tariffs) format. |
| `boundary_policy`, `study_date` | Shared price schedule interpretation, as for tariffs. |

Each shared price takes a scalar or a price list by tariff period (and month
season where configured). For example, with
`schedule = "pt_mainland_2026_daily_bi"`, use
`shared_energy_price = { peak = 0.12, off_peak = 0.08 }`.

Set these App keys once in `[community]`; repeating one inside a site is
refused even if its value agrees:

`location`, `weather_file`, `weather_source`, `start_date`, `resolution`,
`projection_years`, `currency`, `execution_backend`, `discount_rate`,
`inflation_rate`, `import_price_escalation`, `om_escalation`,
`replacement_cost_learning`, `sell_price_inflation`.

The community uses one location, timezone and weather/calendar. Every
project year replays the commissioning calendar, including tariff periods,
as App does; the battery's age and replacement clock continue across years.
Hourly settlement and partial-year `[period]` runs are refused.

### Participants and assets

Each `[[participants]]` entry has a stable `id`, load and supplier prices:
flat `[participants.costs]` prices or `[participants.tariff]`. It may also
have `[participants.reference_tariff]`, own PV, a home battery, inverter
details, costs and supported smart charging using App's site keys. A household
without PV omits PV keys and supplies `annual_consumption_kwh` plus supported
load-profile and billing keys; its metered import is its prepared load and
its export is zero. A household with a battery but no PV is deferred to 0.9.
App's [key reference](config-reference.md) describes the reused site keys;
it does not describe community tables.

Each `[[assets]]` entry has an `id`, PV/inverter keys and `ownership`:
participant shares summing to 1, or `"external"`. It must explicitly state
both import prices and fixed supply charges: `costs.electricity_cost` and
`costs.daily_power_cost`, or `tariff.import_prices` and
`tariff.fixed_charge_per_day`. State zero when a charge is absent. Ownership
splits investment, O&M, actual import bills, fixed charges and injection
income once. External owners' expenses stay outside participant accounts;
their receipts leave the community.

Ids are unique across assets and participants and match
`^[a-z0-9][a-z0-9_-]{0,63}$`. Results, ownership and provenance preserve
hyphens and underscores. Sites and community sums use sorted id order, so
reordering the configuration does not change the results.

### Allocation

- **Fixed:** every participant supplies a nonnegative `coefficient` and the
  coefficients sum to 1 (absolute tolerance `1e-12`). Each interval they are
  renormalised over households that are not exporting, including households
  with zero import. An allocated share above a household's import becomes
  surplus; it is not passed to another household.
- **Proportional:** omit all `coefficient` keys. Coefficients are proportional
  to the receivers' metered imports that interval. If their total import is
  zero, allocation is zero and all pooled injection becomes surplus.

An exporting household receives no share that interval, even when its
export is not pooled. Used and surplus energy are attributed to injectors
in proportion to each one's pooled injection. Assets never receive shares.

## References and money

Each participant has two comparisons:

- **Alone:** the same equipment and physical flows, its own supplier tariff
  and export prices, no sharing and no ownership of any shared asset.
- **No system:** its entire load billed at its reference tariff, or its own
  tariff/flat prices if there is no reference tariff.

`npv_savings` is against the no-system baseline. `community_benefit` is
`npv_savings - npv_alone`, with separate supplier savings, export changes,
shared payments/receipts, network fees, network-credit changes and ownership
parts. Whole assets have net present cost, not an invented no-system NPV.

Shared payments, shared receipts and shared-network fees escalate at
`import_price_escalation`; attributed surplus and unpooled export use
`sell_price_inflation`. Equipment costs, O&M, replacements and discounting
follow App. An optional pack-only terminal health credit stays a separate
sensitivity and does not change the reported community benefit.

An annual network credit is capped against each participant's remaining
supplier network charges, including their fixed part. Shared-network charges
do not increase that cap. The sharing-induced credit change is a separate
annual line; alone and no-system credits remain unchanged. An explicit
asset-tariff credit follows the same supplier cap and ownership split.
Configuration emits one `UserWarning` per such asset, naming its id:
producer-only connections rarely qualify for a network credit, and BREOS
does not judge eligibility. Participants do not receive this warning.

`shared_coverage_of_import` is presented first: shared use / metered
participant import, the coverage of a household's grid purchases, bounded
by 100%. `shared_coverage_of_load` divides shared use by gross household
load and can exceed 100% when metered import includes battery grid charging.
Shared use offsets metered import, which includes grid charging and inverter
draw; it is capped at that import, not at gross household load. Both ratios
remain available and are zero for a zero denominator. There is no unqualified
coverage field.

## Boundaries and regulatory context

This release shares PV only. It has no community storage, hierarchical or
dynamic allocation, time-varying fixed coefficients, peer-to-peer trading,
community `revalue`, Monte Carlo or optimizer. `[vehicle]` in a community
site and households with a battery but no PV are deferred to 0.9 (issues
to be filed). Assets refuse battery, load, smart-charging and reference-tariff
keys. Shared keys inside sites are refused rather than silently overridden.

Regulatory context is dated **2026-10-06**, following ADR 0006; it has not
been refreshed for this implementation. Portugal's Decreto-Lei n.º 15/2022,
as amended by Decreto-Lei n.º 99/2024 and n.º 130/2026, and ERSE Regulamento
n.º 815/2023 provide the recorded ACC context. The ERSE rules settle
15-minute metered balances, exclude injectors from receiving a share, and
leave fixed-allocation surplus with the receiving installation. The ADR
checked the two amending decree-laws in secondary sources only; exact
2026 network-charge/CIEG eligibility was not verified. BREOS does not
calculate legal eligibility, proximity, voltage-level charges or exemptions.
Supply the agreements and charges for the dated study; the example's
`0.02` shared-network charge is an assumption, not a Portuguese tariff.
