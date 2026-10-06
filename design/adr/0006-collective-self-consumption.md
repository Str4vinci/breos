# 0006 — Collective self-consumption: a community entry point beside App

- **Status:** Proposed. No code yet; implementation is planned for 0.8.0,
  once the maintainer accepts or amends the decisions below (#421).
- **Date:** 2026-10-06

## Context

BREOS models one site with one load. In collective self-consumption (ACC,
*autoconsumo coletivo*) the production of one or more PV systems is shared
among several participants, each with their own meter, supplier and tariff.
The 0.7.0 design note left it out
([Outside 0.7.0](../architecture/tariffs-and-smart-charging.md#outside-070)):
sharing coefficients are a simple transformation of the community's
production and each participant's consumption, but a community also needs
stable participant identities, a settlement ledger that reconciles every
15-minute interval, an order between shared PV, a community battery and the
batteries behind each meter, and participant tariffs and ownership, which
the coefficients do not give.

#421 asks for a record of the decisions first. Its contract:

- a community entry point that leaves `breos.App` unchanged for single-site
  runs;
- participants with stable identities, each with a load and optionally their
  own PV and battery behind the meter;
- allocation by fixed and proportional coefficients first; dynamic allocation
  later, as an experimental mode with a plain NumPy reference;
- a settlement ledger that reconciles every 15-minute interval: shared
  energy allocated, used by each participant, and exported;
- a defined order between shared PV, a community battery and batteries behind
  each meter;
- participant tariffs, the price of shared energy, network charges on shared
  energy, and the ownership and cost split of the shared assets;
- results per participant and for the community, against a defined reference
  without the community;
- energy and money closed per interval and per year; Python and Numba agree
  where the kernel is used.

This record answers each point or lists it as an open decision. It changes
no code. Its options carry a recommendation; the maintainer's answers
will replace the open decisions, as they did for ADR 0004.

## Current behaviour this record relies on

Line numbers are at `567df2a` (`origin/develop`, 2026-10-06).

**App is one site.** `App` (`breos/app.py:133-309`) resolves one config
(`resolve_app_config`, `breos/app_config.py:2806`) and runs
`run_app_simulation` (`breos/runners/app.py:352-532`): one weather series,
one PV DC series, one load, one optional battery, one tariff and one
no-system baseline. Its results and provenance describe that one site.

**The projection.** `project_years` (`breos/projection.py:852`) runs the
project years of one site and carries the battery state between them; it
accepts an `initial_carry` (`:862`, `:942`) and, with `record_priced_flows`,
keeps each year's per-step import and export (`ProjectionRun.priced_flows`,
`:804-808`). `run_projection` (`:1181`) and `value_projection` (`:1251`) are
the App path; App, Monte Carlo and the optimizer share them (#179).

**Meters.** A step's `Import_From_Grid` and `PV_AC_Export` are the site's
exchange with the grid. Within one step a site never imports and exports at
once: the DC step exports only when PV covers the load
(`breos/_dispatch.py:579-593`), grid charging runs only when nothing is
exported (`:674`), and the PV-only path splits each step into one or the
other (`breos/battery.py:1235-1237`). A step's two columns are therefore
the two quarter-hourly balances a meter would record at 15-minute steps.

**PV-only and battery sites.** A site without a usable battery takes the
vectorised PV-only path (`breos/battery.py:1197-1263`); a site with one runs
the day loop, in Python or with the Numba kernel
([Numba dispatch backend](../architecture/numba-dispatch-backend.md)).

**Input reuse.** `prepare_simulation_inputs_cached`
(`breos/app_inputs.py:659`) and `reuse_prepared_inputs` (`:611`) reuse
weather, PV and load preparation across App runs that share an input key
(`INPUT_INDEPENDENT_KEYS`, `:557-602`).

**Money.** A tariff prices a site's import, export, grid charge and the
no-system baseline step by step (`_tariff_weights`,
`breos/projection.py:361-377`), at year-1 prices (ADR 0003 E7); economics
escalates and discounts them (E2, E3). `calculate_costs`
(`breos/economics.py:188-274`) prices one site's CAPEX.

## Regulatory setting

Checked on 2026-10-06. Items resting on a secondary source, or not
verified, say so.

**Portugal.**

- **Decreto-Lei n.º 15/2022, de 14 de janeiro**, regulates self-consumption,
  ACC and renewable energy communities. Article 3(gg) defines the managing
  entity of a collective self-consumption (EGAC), Article 86 sets the EGAC's
  duties, among them the sharing of production and its coefficients, and
  Article 87 the sharing of energy; without a sharing mode communicated by
  the EGAC, production is shared in proportion to measured consumption
  (Article 87(1)). Article 83 sets proximity; in the original text, at low
  voltage the same transformer station or at most 2 km apart, and at medium,
  high and very high voltage the same substation and at most 4, 10 and 20 km.
  Article 212 sets the network access tariffs on energy
  shared through the public grid (RESP): the access tariffs of the consumer's
  voltage level, less the network-use tariffs of the levels above the
  production's connection level; an internal network pays none.
- **Amendments.** Decreto-Lei n.º 99/2024 and Decreto-Lei n.º 130/2026, de 29
  de junho (in force 28 August 2026, transposing Directive (EU) 2024/1711),
  amend Article 83 and the ACC rules. Since 99/2024 the substation and the
  distance are alternatives above low voltage, and the distances double in
  low-density territories. Under 130/2026 distances are measured from a main
  production unit designated by the EGAC, and an EGAC or third
  party may hold storage or generation serving the ACC. These two amendments
  were checked in secondary sources only; their full texts were not read.
- **Sharing rules.** The ERSE self-consumption regulation (Regulamento n.º
  815/2023, which revoked 373/2021) sets them, and was read for this record:
  - sharing is settled for each 15-minute period, by the distribution network
    operator, from metered balances (Article 28(7), Article 34);
  - the energy for sharing is what production installations, storage
    installations and consumer installations with integrated production or
    storage inject into the grid (Article 28(3)); an installation that injects
    in a period receives no share in it (Article 28(4));
  - modes: fixed coefficients, which may vary in time (Article 29);
    coefficients proportional to the consumption measured in each period
    (Article 30); hierarchical groups (Article 31); and dynamic coefficients
    communicated afterwards (Article 32). Missing or invalid fixed
    coefficients, and a missing dynamic communication, fall back to
    proportional (Articles 29(2), 32(3)–(4)). The detailed hierarchical and
    dynamic rules follow a pilot project (Article 28(11)); not verified
    whether they are generally available in 2026;
  - under fixed coefficients, the share of an installation that injects in
    the period goes to the others in proportion to their coefficients
    (Article 29(3));
  - energy allocated to a consumer above its measured consumption is its
    surplus, the positive quarter-hourly balance between the allocated energy
    and the consumption (Article 34(1)(c)(ii)); it is not passed to another
    participant. Only the hierarchical mode shares what is left in one group
    with other groups (Article 31(4)–(5));
  - each participant's supplier delivers the positive balance between its
    measured consumption and its allocated energy (Article 34(1)(f));
  - shared energy and the surplus are attributed to the injecting
    installations in proportion to their injection in each period (Articles
    29(4), 30(2));
  - the total surplus is sold through an aggregator, on a market or
    bilateral contract, or peer to peer (Article 8(7)–(8)); unsold surplus
    counts as reduced grid losses, unpaid (Article 8(9)).
- **Exemption of general-interest costs (CIEG).** Despacho n.º 4411-A/2026,
  de 2 de abril, exempts individual and collective self-consumption and
  renewable energy communities that use the RESP from the CIEG for seven
  years, for projects able to operate in 2026–2029. Checked in a government
  summary; the despacho itself was not read. The exact access-tariff terms
  per voltage level for 2026 were not verified.

**Germany.**

- **§42b EnWG, gemeinschaftliche Gebäudeversorgung** (Solarpaket I, BGBl.
  2024 I Nr. 151, in force 16 May 2024). PV electricity of a building is
  shared among its final consumers without passing through a network, by an
  agreed allocation key (*Aufteilungsschlüssel*); consumption is measured
  every quarter-hour, and the shareable amount in each 15-minute interval is
  at most what the plant generates or what all participants consume
  (Abs. 5). Each participant keeps its own supplier for the rest. That no
  network charge is due on the shared part follows from there being no
  network transit; the section itself does not say so. §42a (*Mieterstrom*)
  differs: its supplier must cover the whole supply.
- **§42c EnWG, energy sharing through the public grid** (BGBl. 2025 I Nr.
  347, in force 23 December 2025). Distribution network operators must
  support it within their balancing area from 1 June 2026, and with a
  directly adjacent area in the same control area from 1 June 2028. It needs
  renewable generation or storage holding only renewable electricity, a
  sharing contract with an allocation key, and quarter-hourly metering at
  every supply point and plant. It gives no reduction of network charges.
- **EU.** Directive (EU) 2024/1711 adds a right to energy sharing (Article
  15a of Directive (EU) 2019/944); §42c and Decreto-Lei n.º 130/2026
  transpose it.

**Consequence for the model.** In every regime above, sharing is a
calculation on 15-minute metered balances. No participant's equipment sees
the allocation while it runs: each site's flows are what its own meter
records, and the allocation is applied afterwards. The inputs of the
transformation are therefore each participant's metered import and export
after the equipment behind its meter, not its load, which refines the 0.7.0
note's wording.

## Decision

### K1. Scope

- One community per run, at one location: every site shares the weather,
  the calendar, the timezone and the resolution. The proximity rules keep a
  community local; a study that needs separate weather per site is outside
  0.8.0.
- **Sites.** A participant is a site with a load and, optionally, PV and a
  battery behind its meter. A community asset is a site with PV and,
  optionally, a battery, and no load of its own. Every site is simulated by
  the existing single-site physics.
- **Settlement** is a calculation on the sites' 15-minute import and export,
  as the regimes above apply it. It moves no energy.
- **Out of scope for 0.8.0:** hierarchical sharing, time-varying fixed
  coefficients, a community battery charged from other installations'
  shared energy, peer-to-peer trading, Monte Carlo and the optimizer for
  communities. Each is a later amendment.

### K2. Entry point

Options:

- (a) A new facade, `breos.Community`, in a new package `breos/community/`,
  with its own configuration and results, built on the single-site
  projection (recommended).
- (b) A `[community]` table inside App's configuration.
- (c) A post-processing function over finished App results.

(a) leaves `App`, its config registry, its result and its goldens untouched:
nothing in `breos/app.py`, `breos/app_config.py` or `breos/runners/app.py`
changes, and `breos/__init__.py` only adds the export. (b) would change
what an App result is (one site), its provenance and every key's meaning.
(c) cannot model a community battery, whose dispatch needs the
participants' demand while it runs.

```python
community = breos.Community(config)
community.simulate()
result = community.result()            # per participant and for the community
ledger = community.settlement()        # the first year's 15-minute settlement
```

`Community` mirrors App's methods: `simulate`, `result`, and `settlement` in
place of `timeseries`. `revalue` is not part of 0.8.0.

### K3. Configuration and participant identities

```toml
[community]
location = "porto"
start_date = "2026-01-01"
resolution = "15min"
projection_years = 20
currency = "EUR"
allocation = "fixed"                  # "fixed" or "proportional"
pool_participant_export = true
shared_energy_price = 0.10            # per kWh, paid to the injecting sites
shared_network_charge = 0.02          # per kWh of shared energy used
surplus_price = 0.04                  # per kWh of community surplus
# discount_rate, inflation_rate and the escalators, as App's keys

[[assets]]
id = "school-roof"
n_modules = 60
battery_kwh = 20
ownership = { apt-1 = 0.5, apt-2 = 0.3, apt-3 = 0.2 }
[assets.costs]
# App's cost keys for this site

[[participants]]
id = "apt-1"
coefficient = 0.5                     # "fixed" only
annual_consumption_kwh = 3500
load_profile = "eredes_btn_c"
tariff = { ... }                      # this participant's supplier tariff
```

- **Site keys.** Each `[[participants]]` and `[[assets]]` entry takes App's
  site keys (PV, battery, inverter, load, `tariff`, `reference_tariff`,
  `smart_charging`, `costs`, degradation) and is resolved by
  `resolve_app_config`, so each key means what it means in App. The keys
  that must agree across sites (location, weather, `start_date`,
  `resolution`, `projection_years`, `currency`, the discount rate and the
  escalators, `execution_backend`) are set once in `[community]` and refused
  in a site.
- **Identities.** `id` is required, unique across participants and assets,
  and matches `^[a-z0-9][a-z0-9_-]{0,63}$`. Results, the settlement ledger,
  ownership and provenance are keyed by it. Every community-wide sum runs in
  sorted id order, so reordering the configuration gives bit-identical
  results, and adding or removing a participant renames nothing.
- **No defaults for agreements.** `shared_energy_price`,
  `shared_network_charge` and `surplus_price` are required; an explicit 0 is
  valid. They are agreements and regulated charges BREOS has no source for,
  as ADR 0004 decided for the battery inverter price. Each takes a number or
  a price list by tariff period on a community schedule, in the shape of
  `import_prices`.
- **Resolution.** Options: (a) require `"15min"` (recommended); (b) settle
  hourly steps as an approximation. Every regime above settles 15-minute
  intervals, and a site's own flows inside an hour are not its quarter-hour
  balances: a participant can import in one quarter and export in the next,
  which an hourly balance nets. (a) follows ADR 0002 A3: an hourly community
  run is refused, not approximated.

### K4. Allocation

For each interval `t` (sums in id order):

```text
c_i  = participant i's metered import   (site Import_From_Grid, Wh)
x_i  = participant i's metered export   (site PV_AC_Export, Wh)
S    = sum of the assets' injection + sum of x_i if pooled   # energy for sharing
R    = { i : x_i = 0 }                  # receivers; a site injecting in t receives nothing

fixed:         k_i = coefficient_i / sum(coefficient_j for j in R)    for i in R
proportional:  k_i = c_i / sum(c_j for j in R)                       for i in R
#  k_i = 0 for every i when the denominator is 0: all of S is surplus
a_i = k_i * S                      # allocated
u_i = min(a_i, c_i)                # used
s_i = a_i - u_i                    # participant surplus
supplier_i = c_i - u_i             # bought from i's own supplier
surplus    = S - sum(u_i)          # community surplus, sold at surplus_price
```

- **Fixed** coefficients are constant over the project, at least 0, and
  sum to 1 over all participants. The share of a participant that injects
  in an interval goes to the receivers in proportion to their coefficients
  (Article 29(3)). A participant that neither imports nor exports, for
  example one whose own battery covers its load, stays a receiver: its share
  is allocated to it and becomes its surplus, since only an injecting
  site's share moves. A site never imports and exports in one step, so
  `x_i = 0` covers every importing participant. Options for coefficients that sum to less than 1: refuse
  (recommended) or treat the remainder as surplus. Refusing avoids a rule
  the regulation was not checked for.
- **Proportional** coefficients follow each interval's metered import. With
  `S ≤ Σc` every participant uses all it is allocated; above it each
  participant's consumption is covered and the rest is surplus.
- **No reallocation.** A participant's surplus `s_i` is not passed to
  another participant; it joins the community surplus (Article
  34(1)(c)(ii)). Options: no reallocation (recommended), or a second pass
  that redistributes `s_i` to participants with remaining demand. The second
  pass would settle more energy than the Portuguese operator does, and §42b
  does not prescribe it. A study that wants it uses proportional
  allocation, which never leaves a participant surplus while demand
  remains, or the later dynamic mode.
- **Dynamic allocation** comes later, as an experimental mode, `allocation =
  "dynamic"`. It takes an allocation function or an ex-post allocation file
  per interval, and is checked against a plain NumPy reference
  implementation of the same rule. It is not part of 0.8.0.
- **Implementation.** Allocation is vectorised NumPy over the year's
  interval arrays, in `breos/community/settlement.py`. It does not enter the
  dispatch kernel.

### K5. Order of shared PV, the community battery and the batteries behind each meter

Each interval:

```text
1. every participant site dispatches its own PV and battery for its own
   load, as an App site does                    -> c_i, x_i
2. PV-only assets inject all their PV AC        -> injection
3. the community battery's site dispatches against the virtual load
   L = max(0, sum(c_i) - sum(x_i if pooled) - PV-only asset injection)
                                                -> its injection
4. settlement allocates S and books the surplus (K4)
```

Options for the order between the participants' own batteries and the
shared energy:

- (a) Behind the meter first (recommended). Each participant's battery
  serves its own load before any sharing, exactly as in a single-site run.
- (b) Shared energy first: a participant's battery discharges only into the
  load the allocation leaves.

(a) is what the regimes above do: the allocation is computed afterwards on
metered balances, so a battery behind a meter cannot see it while it runs.
It also keeps every participant's physics identical to its App run (Q2), so
the reference without the community (K8) needs no second simulation. (b)
models a controller with live allocation data, which no regime above
provides; it is a possible later mode.

**The community battery.** One asset may have a battery, at the asset's own
connection. Its site runs the canonical step with `L` as its load: the
participants' demand left after the participants' pooled export and the
PV-only assets. The load is a control signal, not a physical load at the
asset. The site's injection is then everything its meter exports:
`PV_AC_To_Load + Battery_AC_To_Load + PV_AC_Export` of that run. The battery
charges only from its own site's PV that `L` leaves, discharges toward `L`,
and never charges from the grid: `[smart_charging]` is refused on an asset
in 0.8.0. Options: one community battery (recommended for 0.8.0), several
batteries sharing `L` by a rule, or a battery that is allocated shared
energy as a receiving storage installation (Article 28(4) allows this in
Portugal). The second and third need a sharing rule between stores and a
virtual charging path; they are later amendments.

**Fixed coefficients and the battery.** Under fixed allocation part of the
battery's injection can become surplus even when `L > 0`, because `L` is the
community total and the allocation is per participant. The battery's
dispatch is a controller assumption; the settlement applies the rule
exactly, and the surplus shows the cost of a coefficient that does not match
the demand.

**What the asset's run means.** In the battery asset's dispatch run,
`Houseload` is `L` and `Import_From_Grid` is the part of `L` the asset did
not serve: the participants' remaining demand, which they buy from their
suppliers, not an import at the asset. The asset's meter imports nothing,
since it never grid-charges, so its settlement rows record 0 import. No
asset is priced through a tariff (`_tariff_weights`): it would get an import
cost and a no-system baseline for a virtual load. An asset's money comes
only from the settlement and its own costs (K7).

**Participants' own export.** Options: (a) pooled into `S` when
`pool_participant_export = true`, the default (recommended); (b) never
pooled, always sold under the participant's own tariff. Portugal counts a
consumer installation's injection as energy for sharing (Article 28(3)); a
§42b building shares one plant's production. The key covers both.

### K6. Settlement ledger

A new frame, separate from the dispatch ledger, one row per interval and
site, with a schema version of its own (`SETTLEMENT_SCHEMA_VERSION =
"1.0"`). Energy columns are Wh per interval:

| Column | Participant rows | Asset rows |
|---|---|---|
| `Metered_Import` | `c_i` | 0 |
| `Metered_Export` | `x_i` | injection |
| `Pooled_Injection` | `x_i` if pooled, else 0 | injection |
| `Coefficient` | `k_i` (0 if not a receiver) | 0 |
| `Shared_Allocated` | `a_i` | 0 |
| `Shared_Used` | `u_i` | 0 |
| `Participant_Surplus` | `s_i` | 0 |
| `Supplier_Import` | `c_i − u_i` | 0 |
| `Own_Export` | `x_i` if not pooled, else 0 | 0 |
| `Attributed_Shared_Used` | `u` attributed to this injector | the same |
| `Attributed_Surplus` | surplus attributed to this injector | the same |

Attribution to injectors is in proportion to each one's pooled injection in
the interval (Articles 29(4), 30(2)). Community rows per interval:
`Energy_For_Sharing` `S`, `Shared_Used_Total`, `Community_Surplus`.
`Community.settlement()` returns the first project year's frame; each later
year keeps its totals, as App keeps its later years.

### K7. Money and ownership

Per participant and year, at year-1 prices, escalated and discounted as
App's flows are (ADR 0003 E2, E3):

- **Supplier.** `Supplier_Import` at the participant's own import prices,
  its own fixed charge, and its `Own_Export` at its own export prices. Its
  own tariff is resolved on the community index as App resolves one.
- **Shared energy.** `Shared_Used × shared_energy_price` paid to the
  injectors; `Shared_Used × shared_network_charge` paid out of the
  community. Both escalate at the community's `import_price_escalation`.
- **Injection.** Each injector receives `Attributed_Shared_Used ×
  shared_energy_price + Attributed_Surplus × surplus_price`; the surplus
  price escalates at `sell_price_inflation`.
- **Ownership.** An asset's CAPEX, O&M, replacements and its revenues are
  split by its `ownership` table, whose shares are at least 0 and sum to 1
  over participant ids, or by `ownership = "external"` for an asset a third
  party owns, whose money then leaves the community. Options for the
  shared-energy price: (a) one price per kWh, by period if needed, paid to
  the injectors (recommended); (b) no price, the owners using the energy
  free in proportion to ownership; (c) a discount on each participant's
  import price. (a) expresses (b) with a price of 0 and owners who are the
  users, and it keeps money closed by construction; (c) ties the community's
  price to each supplier's tariff, which the regulation does not.
- **Participants' own equipment** is priced as App prices it, in that
  participant's account.
- **Network charges.** One `shared_network_charge` for the community. In
  Portugal a study computes it from the ERSE access tariffs under Article
  212 of Decreto-Lei n.º 15/2022, with or without the CIEG exemption; for a
  §42b building it is 0; under §42c the full network charges apply. Options:
  one charge for the community (recommended), or one per participant for
  communities that span voltage levels. Per-participant charges are a small
  extension once a study needs them.

**Money closure.** In every interval, what receivers pay for shared energy
equals what injectors receive for it; network charges, supplier bills,
surplus revenue and external owners' money are the only money that crosses
the community's boundary.

### K8. References and results

- **Reference without the community.** Each participant alone: the same site
  with the same equipment behind its meter, its own tariff, buying `c_i`
  from its supplier and selling `x_i` under its own export prices, with no
  share of any asset. By K5(a) its energy is the community run's, so only the
  money differs. The community's reference is the sum of its participants'
  references; the assets are not built.
- **No-system baseline.** Each participant's whole load at its own
  `reference_tariff`, or at its tariff, as App's baseline (ADR 0002 A13).
- **Per participant** the result has: energy (load, own PV, metered import
  and export, shared allocated and used, participant surplus, supplier
  import), money per year, NPV against the no-system baseline, and the
  community benefit (NPV with the community less NPV alone), with its parts.
- **For the community:** the assets' energy and money, the energy for
  sharing, used and surplus, the share of participants' consumption covered
  by shared energy, the ownership split, the totals, and the money-closure
  residual.
- **Provenance** records the community table, each site's resolved config
  and App provenance, the ids and their order, the allocation mode and the
  settlement schema version.

### K9. Single-site behaviour, backends and entry points

- **App unchanged.** No App module, key, result field or golden changes;
  every App run is bit-identical. The tests add a guard that the App config
  registry and App goldens are untouched by the community package.
- **Kernel.** Communities add no dispatch code. Each site runs the existing
  projection with the community's `execution_backend`; the settlement is
  NumPy. Python and Numba agree bit for bit on every site, and so on the
  settlement, which a parity scenario checks.
- **A participant's run equals its App run.** A participant site resolved
  as an App config, at the community's shared keys, gives the same
  per-step flows as `App` on that config, bit for bit (Q2).
- **Input reuse.** Sites that share PV inputs reuse their preparation
  through `reuse_prepared_inputs`.
- **Memory.** Settlement needs every site's import and export for one year
  at a time. PR 3 settles year by year and keeps one year of steps per site;
  whether it drives the sites year by year through `project_years` with an
  `initial_carry`, or keeps each projection's `priced_flows`, is chosen there
  by measured memory.
- **Monte Carlo and the optimizer** are not part of 0.8.0 for communities.

## Closure

Per interval, to rounding:

1. Energy for sharing: `S = Σ asset injection + Σ Pooled_Injection` of the
   participants.
2. Allocation: `Σ a_i = S` when some participant receives, else 0; each
   `a_i = u_i + s_i`.
3. Surplus: `Community_Surplus = S − Σ u_i = Σ s_i + (S − Σ a_i)`.
4. Supply: `c_i = u_i + Supplier_Import_i`; `u_i ≤ c_i`.
5. Attribution: `Σ Attributed_Shared_Used = Σ u_i` and
   `Σ Attributed_Surplus = Community_Surplus`.
6. Each site's dispatch ledger keeps every identity of
   `tests/energy_conservation.py`; the community battery's site has `L` as
   its `Houseload`.
7. Money: receivers' shared-energy payments equal injectors' shared-energy
   revenue; per participant, money with the community less money alone
   equals the sum of its settlement terms.

Per year, the same identities hold for the year's sums.

## Reference cases

Three participants A, B and C; one 15-minute interval; Wh; no asset battery.
All cases are computed by hand from K4.

**S1 — proportional.** `S` = 6000, `c` = (3000, 1000, 4000). `k` = (0.375,
0.125, 0.5), `a` = `u` = (2250, 750, 3000), `Supplier_Import` = (750, 250,
1000), surplus 0.

**S2 — fixed coefficients that do not match the demand.** Coefficients (0.5,
0.3, 0.2), `S` = 6000, `c` = (1000, 3000, 4000). `a` = (3000, 1800, 1200),
`u` = (1000, 1800, 1200), A's surplus 2000, `Supplier_Import` = (0, 1200,
2800), community surplus 2000. Proportional allocation of the same interval
gives `u` = (750, 2250, 3000) and no surplus: the same energy, 2000 Wh more of
it used.

**S3 — an injecting participant.** Fixed coefficients (0.5, 0.3, 0.2); C
exports 500 from its own PV (`c_C` = 0, pooled) and the asset injects 5500, so
`S` = 6000. Receivers A and B, coefficients renormalised to (0.625, 0.375):
`a` = (3750, 2250), with `c` = (1000, 3000) `u` = (1000, 2250), community
surplus 2750, `Supplier_Import` = (0, 750). Attribution: the asset 11/12 and
C 1/12 of the interval, so used 3250 is attributed 2979.166667 and 270.833333,
and the surplus 2520.833333 and 229.166667.

**S4 — money.** S2 at `shared_energy_price` 0.10, `shared_network_charge`
0.02 and `surplus_price` 0.04 per kWh, and an import price of 0.20 for every
participant. B pays `1.8 × (0.10 + 0.02) + 1.2 × 0.20 = 0.456` against
`3.0 × 0.20 = 0.60` alone. Receivers pay `4.0 × 0.10 = 0.40` for shared
energy, all of it to the asset, the only injector; the asset also sells the
2000 Wh surplus for `2.0 × 0.04 = 0.08`; `4.0 × 0.02 = 0.08` of network
charges leaves the community.

**S5 — the community battery's load.** Pooled participant export 500, PV-only
asset injection 2000, `Σc` = 6000: the battery asset's `L` is 3500. Its
injection is its site's `PV_AC_To_Load + Battery_AC_To_Load + PV_AC_Export`,
whatever the battery does; the case checks identity 1 with the battery asset
included, and that `L` never counts a participant's export twice.

**Invariants**, at 15-minute steps, Python and Numba:

- Q1. App runs are bit-identical to the release before.
- Q2. Every participant's site flows equal its standalone App run bit for
  bit.
- Q3. A community with one participant, proportional allocation,
  `pool_participant_export = false` and no asset gives that participant's
  App energy and money, with every shared column 0.
- Q4. Reordering the participants in the configuration gives bit-identical
  results.
- Q5. Every closure identity holds per interval and per year.
- Q6. `u_i ≤ min(a_i, c_i)` and every settlement value is at least 0.

## Alternatives considered

- **A `[community]` table in App, or post-processing App results.** Rejected
  (K2).
- **One aggregated site for the whole community.** Rejected: it nets
  participants' import and export across meters, which no regime above
  does, and loses each participant's tariff and supplier bill.
- **Reallocating participant surplus.** Rejected for the fixed and
  proportional modes (K4); possible in the later dynamic mode.
- **Hourly settlement.** Rejected (K3).
- **Shared energy before the batteries behind each meter.** Deferred (K5).
- **Hierarchical sharing.** Deferred: its detailed Portuguese rules follow a
  pilot, and it needs groups in the configuration.

## Implementation plan and effort

Five PRs, about 15 working days for one developer, plus review. A sixth,
experimental PR adds dynamic allocation later. Once the work ships, this
section is replaced by an implementation status, as in ADR 0002.

The work does not depend on #419 or #420: assets and participants run
today's single-site physics. A participant or asset with an AC-coupled
battery or a vehicle becomes possible when those ship, through the site
keys, with no change here.

**Rollout gates.** No merged state exposes a community that cannot price
itself: PR 1–3 add internal modules only, PR 4 adds money, and PR 5 is the
first that exports `breos.Community`.

| PR | Scope | Effort | Main risk |
|---|---|---|---|
| 1 | `breos/community/settlement.py`: fixed and proportional allocation, receivers, participant and community surplus, attribution to injectors, the settlement frame and its version; closure helper; cases S1–S3; property tests (Q5, Q6). | 2.5 days | Floating sums that depend on order (Q4). |
| 2 | Community configuration: `[community]`, `[[assets]]`, `[[participants]]`; ids; site keys through `resolve_app_config` with the shared keys refused; coefficients, ownership and prices validated; 15-minute requirement. | 2.5 days | Keys whose App meaning does not fit a site in a community. |
| 3 | Community runner: participant sites, PV-only assets, the battery asset's virtual load; settlement year by year; per-site carry; Q2, Q3 and the parity scenario. | 4 days | Memory and time with many participants. |
| 4 | Money: supplier bills, shared price, network charges, surplus revenue, ownership, external owners; the reference without the community and the no-system baseline; NPV per participant and for the community; money closure; case S4. | 3.5 days | Escalation and discounting kept identical to App's per participant. |
| 5 | `breos.Community` facade, `result()` and `settlement()`, provenance, an example configuration, docs (a community guide, the settlement ledger), CHANGELOG; ADR implementation status. | 2.5 days | None significant. |
| 6 (experimental, later) | Dynamic allocation with a plain NumPy reference. | 2–3 days | Defining the rule before an implementation. |

Risks across the work:

- **Regulatory drift.** Portugal amended Decreto-Lei n.º 15/2022 in 2024 and
  twice in 2026, and §42c is new. The docs date the rules they cite, and
  the settlement keeps the mode names generic.
- **Scale.** A community of 50 participants runs 50 sites; those with a
  battery run the day loop. The input cache and the Numba backend help;
  PR 3 measures it.
- **Data.** Shared prices, network charges and surplus prices are user
  inputs.

## Open decisions for the maintainer

1. **Entry point (K2):** `breos.Community` in `breos/community/`, App
   untouched (proposed); or a table in App?
2. **Resolution (K3):** 15-minute steps required, hourly refused (proposed);
   or hourly as an approximation?
3. **Allocation (K4):** fixed coefficients summing to 1, renormalised over
   the receivers in each interval, and proportional to metered import; no
   reallocation of participant surplus (proposed)?
4. **Order (K5):** batteries behind each meter first, then sharing on
   metered balances (proposed); or shared energy first?
5. **Community battery (K5):** at most one, dispatched against the virtual
   load `L`, charging only from its own site's PV (proposed)?
6. **Participants' export (K5):** pooled by default, with a key to turn it
   off (proposed); or never pooled?
7. **Prices (K3, K7):** shared energy sold at one price per kWh (by period
   if needed) paid to the injecting sites (proposed), rather than free use
   by owners or a discount on each supplier's price; `shared_energy_price`,
   `shared_network_charge` and `surplus_price` required with no default, 0
   allowed; one network charge for the community (proposed)?
8. **Ownership (K7):** shares over participants or an external owner, for
   CAPEX, O&M, replacements and revenues (proposed)?
9. **Reference (K8):** each participant alone with the same equipment and
   tariff, plus the no-system baseline (proposed)?
10. **Scope (K1, K9):** no Monte Carlo, optimizer, hierarchical sharing or
    `revalue` for communities in 0.8.0 (proposed)?
11. **Target release:** 0.8.0, in the order of the plan above.

## Consequences

- A community of households and shared PV, with an optional community
  battery, can be simulated and settled per 15-minute interval with each
  participant's own tariff, and every participant's benefit is reported
  against the same household alone.
- `breos.App` and its results are unchanged; the community is a second
  facade built on the same single-site physics.
- The settlement ledger is a new frame with its own schema version; the
  dispatch ledger and the result format do not change.
- Hierarchical and dynamic sharing, several community batteries, virtual
  charging of storage, community Monte Carlo and optimization stay outside
  0.8.0.
