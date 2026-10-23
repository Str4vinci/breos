# 0006 — Collective self-consumption: a community entry point beside App

- **Status:** Implementation in progress for 0.8.0 (#421), following the
  maintainer's decisions of 2026-10-06 and 2026-10-22. PRs 1–3 implement
  internal settlement, configuration and the runner; PR 4 adds internal
  money accounts. The public facade and documentation remain PR 5 work.
  See [PR 4 implementation interpretations](#pr-4-implementation-interpretations).
- **Date:** 2026-10-06; decisions taken 2026-10-06 and 2026-10-22;
  amended 2026-10-22

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

This record answers each point. The maintainer took
its first eleven decisions on 2026-10-06 and four further decisions on
2026-10-22; the sections below follow them, and
[Decisions](#decisions) lists them. Two of them changed the proposal: 0.8.0
shares PV only, with no community battery (community storage is recorded
for a later amendment in K10), and the work is implemented after #419 and
#420.

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
  battery behind its meter. A community asset is a PV system with no load
  and no battery. Every site is simulated by the existing single-site
  physics.
- **What is shared.** In 0.8.0 the community shares PV only: the assets'
  output and, by default, the participants' own export. Batteries behind a
  participant's meter work as in App. Community storage is deferred (K10;
  decision 5).
- **Settlement** is a calculation on the sites' 15-minute import and export,
  as the regimes above apply it. It moves no energy.
- **Out of scope for 0.8.0** (decision 10): community storage (K10),
  hierarchical sharing, time-varying fixed coefficients, peer-to-peer
  trading, `revalue`, Monte Carlo and the optimizer for communities. Each
  is a later amendment.

### K2. Entry point

Options:

- (a) A new facade, `breos.Community`, in a new package `breos/community/`,
  with its own configuration and results, built on the single-site
  projection. Decided 2026-10-06 (decision 1).
- (b) A `[community]` table inside App's configuration.
- (c) A post-processing function over finished App results.

(a) leaves `App`, its config registry, its result and its goldens untouched:
nothing in `breos/app.py`, `breos/app_config.py` or `breos/runners/app.py`
changes, and `breos/__init__.py` only adds the export. (b) would change
what an App result is (one site), its provenance and every key's meaning.
(c) could settle PV-only assets, but it has no configuration that keeps
participant ids stable and the shared keys equal across sites, and no place
for community storage, whose dispatch needs the participants' demand while
it runs (K10).

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
  `smart_charging`, `costs`, degradation). Sites with PV are resolved by
  `resolve_app_config`; Community builds assets' zero loads and households
  without PV itself (A2, A7), keeping App's meanings for the keys it uses.
  The keys that must agree across sites (location, weather, `start_date`,
  `resolution`, `projection_years`, `currency`, the discount rate and the
  escalators, `execution_backend`) are set once in `[community]` and refused
  in a site.
- **Assets** take App's PV, inverter and cost keys, plus an explicit import
  price (required, no default, 0 allowed) and explicit fixed charges if the
  asset has a supply contract (likewise no default, 0 allowed). A battery,
  a load and `[smart_charging]` are refused on an asset in 0.8.0. App requires a
  positive `annual_consumption_kwh`, so the community builds an asset's
  zero load itself rather than resolving one through App.
- **Households without PV.** Community builds their 15-minute meter
  import from their load, with zero export, and uses App's tariff code.
  Shared energy is a settlement balance on metered kWh, not energy routed
  to the household. Their alone reference bills that same load on the same
  tariff without sharing; App's validation is unchanged (decision 14).
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
- **Resolution.** Decided 2026-10-06 (decision 2): (a). Options: (a)
  require `"15min"`; (b) settle hourly steps as an approximation. Every
  regime above settles 15-minute
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
  site's share moves. App's meter balances are mutually exclusive after
  the prerequisite inverter-draw fix (A6), so `x_i = 0` covers every importing
  participant. Coefficients that sum to anything but 1 are refused, rather
  than treating the remainder as surplus: that would be a rule the
  regulation was not checked for.
- **Proportional** coefficients follow each interval's metered import. With
  `S ≤ Σc` every participant uses all it is allocated; above it each
  participant's consumption is covered and the rest is surplus.
- **No reallocation.** A participant's surplus `s_i` is not passed to
  another participant; it joins the community surplus (Article
  34(1)(c)(ii)). Decided 2026-10-06 (decision 3), with the fixed and
  proportional modes above. The rejected option, a second pass that
  redistributes `s_i` to participants with remaining demand, would settle
  more energy than the Portuguese operator does, and §42b does not
  prescribe it. A study that wants it uses proportional allocation, which
  never leaves a participant surplus while demand remains, or the later
  dynamic mode.
- **Dynamic allocation** comes later, as an experimental mode, `allocation =
  "dynamic"`. It takes an allocation function or an ex-post allocation file
  per interval, and is checked against a plain NumPy reference
  implementation of the same rule. It is not part of 0.8.0.
- **Implementation.** Allocation is vectorised NumPy over the year's
  interval arrays, in `breos/community/settlement.py`. It does not enter the
  dispatch kernel.

### K5. Order of shared PV and the batteries behind each meter

Each interval:

```text
1. every participant site dispatches its own PV and battery for its own
   load, as an App site does                    -> c_i, x_i
2. every asset injects all its PV AC            -> its PV_AC_Export
3. settlement allocates S and books the surplus (K4)
```

Decided 2026-10-06 (decision 4): behind the meter first. The options were:

- (a) Behind the meter first. Each participant's battery serves its own load
  before any sharing, exactly as in a single-site run.
- (b) Shared energy first: a participant's battery discharges only into the
  load the allocation leaves.

(a) is what the regimes above do: the allocation is computed afterwards on
metered balances, so a battery behind a meter cannot see it while it runs.
It also keeps participants' physics identical to their App runs, or their
load-only construction without PV (Q2), so
the reference without the community (K8) needs no second simulation. (b)
models a controller with live allocation data, which no regime above
provides; it is a possible later mode.

**Assets.** An asset has no load and no battery, so it takes the vectorised
PV-only path. Its injection is the run's `PV_AC_Export`, after the
prerequisite App fix nets inverter draw against AC surplus (A6). Any
remaining grid import is billed at the asset's explicit import price,
with its explicit supply-contract fixed charges (A7). These costs follow
ownership (K7); the asset has no no-system baseline.

**Participants' own export.** Decided 2026-10-06 (decision 6): (a). Options:
(a) pooled into `S` when `pool_participant_export = true`, the default; (b)
never pooled, always sold under the participant's own tariff. Portugal
counts a consumer installation's injection as energy for sharing (Article
28(3)); a §42b building shares one plant's production. The key covers both.

### K6. Settlement ledger

A new frame, separate from the dispatch ledger, one row per interval and
site, with a schema version of its own (`SETTLEMENT_SCHEMA_VERSION =
"1.0"`). Energy columns are Wh per interval:

| Column | Participant rows | Asset rows |
|---|---|---|
| `Metered_Import` | `c_i` | actual asset import |
| `Metered_Export` | `x_i` | injection |
| `Pooled_Injection` | `x_i` if pooled, else 0 | injection |
| `Coefficient` | `k_i` (0 if not a receiver) | 0 |
| `Shared_Allocated` | `a_i` | 0 |
| `Shared_Used` | `u_i` | 0 |
| `Participant_Surplus` | `s_i` | 0 |
| `Supplier_Import` | `c_i − u_i` | actual asset import |
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
  Assets' actual imports and supply-contract fixed charges are billed at
  their explicit prices, with no invented no-system baseline (decision 13).
- **Shared energy.** Decided 2026-10-06 (decision 7): one price per kWh,
  by period if needed. `Shared_Used × shared_energy_price` is paid to the
  injectors; `Shared_Used × shared_network_charge` is paid out of the
  community. Both escalate at the community's `import_price_escalation`.
  The other options were no price, the owners using the energy free in
  proportion to ownership, and a discount on each participant's import
  price. A price expresses the first with a price of 0 and owners who are
  the users, and keeps money closed by construction; a discount ties the
  community's price to each supplier's tariff, which the regulation does
  not.
- **Injection.** Each injector receives `Attributed_Shared_Used ×
  shared_energy_price + Attributed_Surplus × surplus_price`; the surplus
  price escalates at `sell_price_inflation`.
- **Ownership.** Decided 2026-10-06 (decision 8). An asset's CAPEX, O&M,
  import bills, fixed charges and revenues are split by its `ownership`
  table, whose shares are at least 0 and sum to 1 over participant ids,
  or by `ownership = "external"` for an
  asset a third party owns, whose money then leaves the community.
- **Participants' own equipment** is priced as App prices it, in that
  participant's account.
- **Network charges.** One `shared_network_charge` for the community. In
  Portugal a study computes it from the ERSE access tariffs under Article
  212 of Decreto-Lei n.º 15/2022, with or without the CIEG exemption; for a
  §42b building it is 0; under §42c the full network charges apply. One
  charge for the community is decided; one per participant, for communities
  that span voltage levels, is a small extension once a study needs it.
- **Annual network credit.** Cap it against the household's remaining
  supplier network charges, including the fixed part, as App does.
  Shared-network charges do not enter the cap. Report any change caused by
  sharing as a separate annual line (decision 15).

**Money closure.** In every interval, what receivers pay for shared energy
equals what injectors receive for it; network charges, supplier bills,
surplus revenue and external owners' money are the only money that crosses
the community's boundary.

### K8. References and results

- **Reference without the community** (decision 9). Each participant alone:
  the same site with the same equipment behind its meter, its own tariff,
  buying `c_i` from its supplier and selling `x_i` under its own export
  prices, with no share of any asset. By K5(a) its energy is the community
  run's, so only the money differs. The community's reference is the sum of
  its participants' references; the assets are not built.
  For a household without PV, this is its whole load billed on its own
  tariff without sharing, using App's no-system billing code (decision 14).
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
  and App provenance (load and tariff provenance for households without PV),
  the ids and their order, the allocation mode and the settlement schema version.

### K9. Single-site behaviour, backends and entry points

- **App unchanged.** No App module, key, result field or golden changes;
  every App run is bit-identical. The tests add a guard that the App config
  registry and App goldens are untouched by the community package.
- **Kernel.** Communities add no dispatch code. Sites with PV run the existing
  projection with the community's `execution_backend`; households without
  PV use their load as meter import. The settlement is
  NumPy. Python and Numba agree bit for bit on every site, and so on the
  settlement, which a parity scenario checks.
- **A participant's run equals its App run.** A participant site resolved
  as an App config, at the community's shared keys, gives the same
  per-step flows as `App` on that config, bit for bit (Q2).
  Without PV, Q2 checks meter import against the prepared load and export
  against zero; Q3 checks the same load billed without sharing, using App's
  tariff code, rather than constructing an App with invalid zero PV.
- **Input reuse.** Sites that share PV inputs reuse their preparation
  through `reuse_prepared_inputs`.
- **Memory.** Settlement needs every site's import and export for one year
  at a time. PR 3 settles year by year and keeps one year of steps per site;
  whether it drives the sites year by year through `project_years` with an
  `initial_carry`, or keeps each projection's `priced_flows`, is chosen there
  by measured memory.
- **Monte Carlo and the optimizer** are not part of 0.8.0 for communities.

### K10. Deferred: community storage

Decided 2026-10-06 (decision 5): 0.8.0 has no community storage. Three
candidate models are recorded for a later amendment, without a choice
between them:

- **(A) A battery as a community producer.** A battery at an asset charges
  from the community's surplus and serves the community's remaining demand;
  its output is energy for sharing, allocated like PV. This is the model the
  proposal drafted: the asset's site runs the canonical step against a
  virtual load `L = max(0, Σ c_i − Σ pooled x_i − PV-only asset injection)`,
  and its injection is `PV_AC_To_Load + Battery_AC_To_Load + PV_AC_Export`
  of that run. That run's `Import_From_Grid` is the part of `L` it did not
  serve, which the participants buy, not an import at the asset. The asset
  dispatch path against `L` is what this model needs. Under fixed
  coefficients part of its output can become surplus even when `L > 0`,
  since `L` is a community total.
- **(B) Virtual capacity shares.** Each household holds a share of a
  battery as an account: its export is deposited and later withdrawn
  against its import, within its share. Examples: MVV Energie's "Strombank"
  research project in Mannheim (a 100 kWh battery from December 2014, with
  14 households and four businesses, each seeing its own *Kontostand*;
  checked in trade-press reports only), and Synergy and Western Power's
  PowerBank trial in Meadow Springs, Western Australia (a 105 kW, 420 kWh
  battery, 52 households, up to 8 kWh of virtual storage each for one
  Australian dollar a day; checked in the Western Australian government's statement of
  5 November 2018). Neither is a Portuguese or German ACC rule.
- **(C) Storage as a receiving installation.** A storage installation is
  allocated shared energy while it is not injecting, and its later injection
  is energy for sharing. ERSE Regulamento n.º 815/2023 gives storage
  installations both roles: their injection is energy for sharing (Article
  28(3)), and they may be allocated shared energy in periods without net
  injection (Article 28(4)). It needs a virtual charging path in the
  settlement.

## Closure

Per interval, to rounding:

1. Energy for sharing: `S = Σ asset injection + Σ Pooled_Injection` of the
   participants.
2. Allocation: `Σ a_i = S` when the receiver denominator is positive,
   else 0; each `a_i = u_i + s_i`.
3. Surplus: `Community_Surplus = S − Σ u_i = Σ s_i + (S − Σ a_i)`.
4. Supply: `c_i = u_i + Supplier_Import_i`; `u_i ≤ c_i`.
5. Attribution: `Σ Attributed_Shared_Used = Σ u_i` and
   `Σ Attributed_Surplus = Community_Surplus`.
6. Each site's dispatch ledger keeps every identity of
   `tests/energy_conservation.py`.
7. Money: receivers' shared-energy payments equal injectors' shared-energy
   revenue; per participant, money with the community less money alone
   equals the sum of its settlement terms.

Per year, the same identities hold for the year's sums.

## Reference cases

Three participants A, B and C; one 15-minute interval; Wh. All cases are
computed by hand from K4.

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

**S5 — nobody to receive.** Proportional allocation. A and B export 300
and 200 (pooled), C neither imports nor exports, and the asset injects 2500,
so `S` = 3000. The receivers are C alone (`x_C` = 0), whose import is 0, so
the denominator is 0, every `k_i` is 0, and all 3000 is community surplus,
attributed 2500 to the asset, 300 to A and 200 to B.

**Invariants**, at 15-minute steps, Python and Numba:

- Q1. App runs are bit-identical to the release before.
- Q2. A participant with PV has site flows equal to its standalone App run
  bit for bit. Without PV, meter import equals its prepared 15-minute load
  and export is zero.
- Q3. A community with one participant, proportional allocation,
  `pool_participant_export = false` and no asset gives that participant's
  App energy and money, with every shared column 0. Without PV, the
  comparison is the same load billed on the same tariff without sharing,
  using App's no-system billing code; no invalid App config is constructed.
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
- **A community battery in 0.8.0.** Deferred by decision 5 (K10).

## PR 4 implementation interpretations

Recorded 2026-10-23 for maintainer review. These choices describe the internal
implementation; they do not introduce an App key, public facade or regulatory
assumption beyond the explicit study prices.

1. **Retention (maintainer's PR 4 instruction).** Interpret K9's memory note
   as keeping the full first-year settlement and every year's per-site and
   community energy totals by default. `run_community` can explicitly keep
   every year's settlement frame. `price_community` can independently retain
   every year's settlement, money ledger and interval residuals; by default
   it retains only year 1, plus all annual accounts and closure maxima.
   Later settlement frames are rebuilt one year at a time from the retained
   unrounded meter arrays, priced and released. The money result does not
   retain the physics run. A3's full-horizon projection, battery carry and
   event clocks remain unchanged. This replaces PR 3's retention of every
   settlement frame (about 410 MiB for 50 participants over two years).
2. **Price basis and signs.** Interval ledgers and priced yearly site rows
   use year-1 currency/kWh prices; positive costs are outflows and positive
   receipts are inflows. Cashflows and community yearly totals are nominal
   for that project year. Fixed supplier charges, annual credits, equipment
   and replacements are annual entries, not apportioned to intervals.
   Fixed charges use App's simulated duration in days. No additional taxes,
   demand charges or community administration costs are inferred.
3. **Exact App arithmetic.** Supplier bills use App's tariff/reference
   pricing functions and valuation/discounting helpers. Annual flat prices
   preserve App's sum/divide/multiply order; tariff prices preserve its
   weighted-power reduction. Meter Wh are converted back to average W for
   those pricing helpers only. This is not a second dispatch conversion.
   Load-only supplier bills use App's no-system order of summing import and
   fixed charges before escalation. They have zero equipment CAPEX, O&M,
   replacement and terminal value, even if a cost preset contains such keys.
4. **Escalation and time.** Shared payments, shared receipts and the single
   shared-network charge escalate at the common import rate; `None` inherits
   inflation as in App. Own unpooled export and attributed surplus escalate
   at the sell rate. O&M uses its own rate. CAPEX is at t=0, annual cashflows
   at year end, and pack replacements at App's actual swap instants, with
   inflation and cost learning. AC coupling retains both inverter costs.
5. **References and terminal credit.** The alone reference reuses the same
   physical projection, owns no asset shares and sells all its own export.
   Its no-system baseline uses its reference tariff, or its supplier tariff
   or flat prices. All escalation rates remain community-wide as validated
   by PR 2, including refusal of per-site reference escalation. Optional
   terminal health credit remains App's separate pack-only sensitivity;
   unadjusted NPV determines community benefit. Identical physical terminal
   credit in the alone and community accounts cancels from that benefit.
6. **Ownership and external owners.** Whole-asset accounts expose costs and
   net present cost, with no no-system baseline or savings NPV. Participant
   accounts take the configured fraction of each asset's initial investment,
   O&M, actual imports, fixed charges, credits and injection income once.
   Community totals sum participant accounts with those shares included;
   adding the separately reported whole-asset accounts would double count.
   External assets' expenses and CAPEX stay outside participant accounts;
   their gross injection receipts leave the community boundary. Their net
   present cost is reported separately, not added to the community's NPV.
7. **Energy sales and benefit parts.** Pooled export has zero own supplier
   export revenue; only attributed shared-use and surplus receipts replace
   it. Unpooled export retains its own price. Per-participant annual and NPV
   benefit components are supplier import savings, own export revenue change,
   shared payments, shared-network fees, direct shared receipts, surplus
   receipts, the credit change, and asset ownership. The ownership NPV part
   includes initial investment as well as annual net costs.
8. **Annual credits.** Recompute the cap using remaining supplier imports
   and the supplier's fixed network part only; shared-network charges never
   increase it. Keep the alone and no-system credits unchanged. The signed
   sharing line is after-sharing credit minus alone credit, escalated at the
   import rate; a negative number reduces the participant's benefit. PR 2
   also accepts an explicit credit in an asset tariff: the implementation
   honors App's supplier cap on that asset's actual imports and splits it by
   ownership. **Question:** should producer-only contracts instead refuse
   annual credits? No eligibility rule is inferred from a household example.
9. **Closure and precision.** Shared payments close against receipts to all
   injectors. The interval boundary includes supplier energy bills, shared
   network charges, surplus sales and external-owner receipts; the annual
   boundary additionally includes supplier fixed charges/credits, equipment
   O&M and replacement outlays. Initial investment is in NPV, outside annual
   operating closure. Prices and accounts remain unrounded; sums use sorted
   ids. Residuals expose floating-point roundoff rather than reallocating or
   rounding cash between participants.
10. **Coverage denominator.** K8's “consumption covered” is reported both as
    shared use divided by gross household load and divided by metered import,
    with zero for a zero denominator. Own PV self-consumption stays separate.
    **Question:** which coverage ratio should PR 5 present by default?

The earlier conservative refusal of `[vehicle]` remains: its ACC semantics
need maintainer review after #420. No further money policy is needed to run
the internal accounts; the two questions above concern future validation
and presentation.

## Implementation plan and effort

Five PRs, about 14.5 working days for one developer, plus review. A sixth,
experimental PR adds dynamic allocation later. Once the work ships, this
section is replaced by an implementation status, as in ADR 0002.

**Order in 0.8.0** (decision 11). The work is implemented after the
AC-coupled battery (#419) and vehicle-to-home (#420) work. It uses that work
through the single-site physics. The order means the reference results, the
example configuration and the docs are computed on the final PV defaults
(#417, #418, now implemented), and a participant's site keys include AC
coupling and a vehicle when the community ships.

**Additional prerequisite** (decision 12). A separate App change on its
own branch must net an idle PV inverter's night draw against AC bus surplus
before export in both PV-only and AC-battery paths. ACC depends on that
fix before PR 3's meter-balance and Q2–Q3 checks; it adds no netting of its
own. Several-inverter App results move by a few Wh a year in that separate
change. ACC's unchanged-App guard is measured after the fix, and Q3 holds
exactly without an inverter-draw exception.

**Rollout gates.** No merged state exposes a community that cannot price
itself: PR 1–3 add internal modules only, PR 4 adds money, and PR 5 is the
first that exports `breos.Community`.

| PR | Scope | Effort | Main risk |
|---|---|---|---|
| 1 | `breos/community/settlement.py`: fixed and proportional allocation, receivers, participant and community surplus, attribution to injectors, the settlement frame and its version; closure helper; cases S1–S3 and S5; property tests (Q5, Q6). | 2.5 days | Floating sums that depend on order (Q4). |
| 2 | Community configuration: `[community]`, `[[assets]]`, `[[participants]]`; ids; App site keys with the shared keys refused; Community's no-PV household path; assets' battery, load and smart-charging keys refused; explicit asset import prices and supply-contract fixed charges; coefficients, ownership and prices validated; 15-minute requirement. | 2.5 days | Keys whose App meaning does not fit a site in a community. |
| 3 | Community runner: participant sites, no-PV households from their load, and PV-only assets with their zero load; settlement year by year; per-site carry; Q2, Q3 and the parity scenario, after the separate App inverter-draw fix. | 3.5 days | Memory and time with many participants. |
| 4 | Money: participant and asset supplier bills, shared price, network charges, supplier-only annual credit caps and their separate annual changes, surplus revenue, ownership, external owners; the reference without the community and the no-system baseline; NPV per participant and for the community; money closure; case S4. | 3.5 days | Escalation and discounting kept identical to App's per participant. |
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

## Amendments of 2026-10-22

The record was written against `567df2a`. The implementation it will build
on is now `4a3cc04`: ADR 0004 is implemented, including M1–M8, as are the
electrical work of ADR 0007–0009, the 0.8.0 refactors and #416–#418. The
"Current behaviour" section remains a historical snapshot, with its line
references at `567df2a`. Every location below is at `4a3cc04`. These notes
qualify the body where the code has moved or its behaviour changed; the
first eleven decisions are unchanged. The maintainer resolved the four
questions on 2026-10-22 in decisions 12–15 below. A6's App fix is a future
dependency, not behaviour present at `4a3cc04`. The regulatory sources have
not been rechecked.

- **A1. The single-site entry points remain.** K2 and PR 2 use
  `resolve_app_config` (`breos/app_config.py:3302-3421`); App resolves and
  runs one site (`breos/app.py:162-180`), through `run_app_simulation`
  (`breos/runners/app.py:620-815`). Validation now returns `ValidatedSpecs`
  and uses the registry's checkers (`breos/app_config.py:1961-2014`), rather
  than requiring a second community copy of App's rules. Remove community
  metadata before resolving a site's App payload. Apply App's recursive
  key normalisation only to that payload: it changes hyphens in mapping
  keys (`breos/app_config.py:1579-1609`), so applying it to `ownership`
  would turn `apt-1` into `apt_1` and break K3's identities.
- **A2. Community supports households without PV.** Decided 2026-10-22
  (decision 14): Community builds their meter import from the 15-minute
  load, with zero export, and uses App's tariff code. Shared energy changes
  the billed balance, not physical routing. Q2 checks those load-only
  flows; Q3 and the alone reference bill the same load on the same tariff
  without sharing, using App's no-system billing code
  (`breos/projection.py:379-411`, `:499-521`). App's positive-PV validation
  stays unchanged (`breos/app_config.py:2662-2677`, `:2711-2714`).
- **A3. Keep the whole projection clock and carry.** The shared interfaces
  are `ProjectionYear`, `ProjectionRun` and `project_years`
  (`breos/projection.py:744-836`, `:1026-1262`); `run_projection` and
  `value_projection` remain the App wrappers (`:1265-1310`, `:1358-1408`).
  Carry includes energy origins, native degradation state, controller state
  and retirement (`:125-225`). K9's alternative of repeated
  one-year calls with `initial_carry` alone is insufficient: each call
  finalises rainflow, applies terminal replacement policy and calculates
  remaining service years against its own horizon (`:1127-1147`), and
  controller and event year numbers would restart. PR 3 must either retain
  full-horizon `priced_flows` (available even without a tariff), or add a
  streaming seam to the shared projection that preserves those semantics.
  `record_priced_flows=True` needs frame years, not aligned summaries
  (`:1157-1193`); ordinary App runs keep them only when dispatch is price
  blind (`breos/runners/app.py:720-738`). Request them explicitly for ACC,
  including price-aware household controllers; settlement does not steer
  their dispatch.
- **A4. Dispatch powers become settlement energies once.** Ledger schema
  `"3.1"` is already present, unreleased (`breos/_dispatch.py:275-345`).
  K4–K6's Wh mean `Import_From_Grid × 0.25` and `PV_AC_Export × 0.25`,
  not the frame columns as stored: flows and retained `priced_flows` are
  average W (`breos/app.py:268-290`, `breos/projection.py:829-833`). Keep
  settlement schema `"1.0"` separate. Closure 6 uses the current conservation
  helper, including `PV_AC_To_Battery − PV_Charge_Conversion_Loss` in the
  charge-input split, and night draw in the load-supply identity
  (`tests/energy_conservation.py:82-119`, `:147-202`); origins still
  reconcile separately (`:218-254`). No new dispatch schema is needed.
- **A5. Behind the meter still runs first, with AC night import.** The AC
  step consumes the complete PV-only chain, serves load, charges and then
  exports; grid charging runs only without export or discharge
  (`breos/_dispatch.py:961-1018`, `:1046-1085`). Its import includes the PV
  inverter's night draw even while the battery serves the load (ADR 0004
  M3); do not subtract that draw or `Grid_AC_To_Battery` from K4's import.
  With one PV inverter these paths still give mutually exclusive import
  and export. The DC hybrid's night draw is instead conditional on no PV
  and no battery flow (`:1341-1350`). K5's order, K8's reuse of participant
  physics for the alone reference, and Q2 stand for App-valid sites of
  either topology. Clipping and AC-cable loss have already reduced export;
  neither is shareable energy or an extra settlement deduction.
- **A6. Fix simultaneous inverter-draw import and export in App.** At
  `4a3cc04`, both paths add night draw without netting against export
  (`breos/battery.py:1415-1419`, `breos/_dispatch.py:1046-1052`). Decided
  2026-10-22 (decision 12): a separate App change on its own branch nets an
  idle PV inverter's draw against AC bus surplus before export, in PV-only
  and AC-battery runs. A step then never shows both import and export
  caused by inverter draw. ACC depends on that fix, settles App's meter
  balances directly (K4–K6), and preserves Q3 exactly, with no exception.
- **A7. Assets export the full PV-only result and pay for actual imports.**
  Use App's DC and electrical preparation (`breos/app_inputs.py:474-535`,
  `:612-726`) and all `ProjectionYear` fields with zero load and no battery
  (`breos/runners/app.py:675-686`). Conversion includes the curve, voltage
  row, derating, clipping, cables, output scale and per-inverter string DC
  (`breos/battery.py:1308-1358`, `breos/inverter.py:467-533`). After A6's
  fix, use App's export and actual import, including remaining night draw
  (`breos/battery.py:1415-1447`). Decided 2026-10-22 (decision 13): record
  asset imports in both K6 import columns and bill an explicit import
  price, required with no default, 0 allowed; supply-contract fixed charges
  are likewise explicit. Split the cost by ownership (K7), with no
  invented no-system baseline for the asset.
- **A8. Input reuse is by the whole preparation, not PV alone.**
  `reuse_prepared_inputs` retains only the latest preparation and gives each
  run a deep copy (`breos/app_inputs.py:850-872`, `:968-994`). Group runs by
  `input_configuration_key` (`:905-927`); `INPUT_INDEPENDENT_KEYS` is now
  derived from `AppConfigField.affects_inputs` (`:838-842`). Different load
  profiles, annual demands or battery-temperature inputs prevent reuse even
  when PV matches (`:538-602`). Weather CSV and metadata-sidecar contents
  also enter the key (`:930-965`). K9 and PR 3 may use this existing cache,
  but must not promise a shared PV/weather preparation across arbitrary
  households or drop load keys to force a hit. Sharing those stages further
  requires a separate implementation that preserves Q2 and provenance.
- **A9. Money remains separate from physics.** K7 and PR 4 build on
  `build_costs_dict` (`breos/app_config.py:3239-3258`), `price_year_rows`,
  `value_year_rows` and `discount_cashflows` (`breos/economics.py:485-530`,
  `:749-901`). AC coupling prices a simple PV inverter plus the required
  separately sized battery inverter (`:257-340`); bundled replacement
  prices are supported but terminal health credit stays pack-only
  (`breos/projection.py:1383-1400`). Preserve those costs and replacement
  instants in both participant accounts. Reprice supplier import and own
  export after settlement; do not add shared receipts to the standalone
  export revenue and sell the pooled energy twice. Shared payments and
  receipts need separate components: they escalate at the import rate,
  whereas surplus and unpooled export escalate at the sell rate. The
  existing valuer has no ACC components; extend composition in the community
  package while retaining App's year-end discounting, O&M and replacement
  rules. Asset ownership splits costs and revenues once, without adding
  an App no-system baseline for the asset.
- **A10. Cap annual network credit at supplier network charges only.**
  Decided 2026-10-22 (decision 15): use the household's remaining supplier
  network charges, including the fixed part, following App's cap
  (`breos/projection.py:436-449`). Shared-network charges do not enter it.
  Keep the alone reference's credit and report any sharing-induced change
  as a separate annual line in K7–K8 and Closure 7; economics subtracts
  each household's credit (`breos/economics.py:749-777`, `:835-843`).
- **A11. Use unrounded result inputs and retain the new diagnostics.**
  `build_year_row` defines the shared annual energy schema
  (`breos/projection.py:265-376`); `PV_Production_kWh` is delivered direct
  PV plus PV-origin battery AC and export, not raw inverter production.
  Public result format remains `"1"` (`breos/result_schema.py:16`), with
  `yearly`, `monthly`, `financial`, `pv_loss_waterfall`, `electrical` and
  provenance (`breos/app_results.py:392-478`). Those public values are
  rounded; K4–K8 and sorted-id sums must use the underlying arrays and
  year rows. Keep App's own-PV self-consumption separate from shared use.
  AC charging/discharging losses and PV-to-battery AC have their own
  waterfall blocks (`breos/runners/app.py:451-475`, `:569-600`); provenance
  records the resolved electrical config, weather/load sources, execution,
  tariff/reference, economics and resolved solar timing
  (`breos/app_results.py:274-348`). Community results add settlement and
  ownership accounts beside these site records, without changing App fields.
- **A12. Reference arithmetic and closure.** S1–S5 depend only on K4,
  not on BREOS PV or dispatch defaults. Rechecked with exact rational
  arithmetic: every published number is unchanged, including S3's rounded
  attribution and S4's 0.456 bill. Closure 2 now says the receiver
  denominator must be positive: S5 has a receiver but no allocation. Q1
  guards the single-site implementation ACC builds on, not the older PV
  defaults. Q2 and Closure 6 include the electrical and AC paths described
  above (`breos/battery.py:1308-1448`,
  `tests/energy_conservation.py:82-202`). Extend PR 3's reference checks to
  night import, clipping, cable loss and several inverters, including
  decisions 12–13; S1–S5 alone do not cover those paths. K10 stays deferred:
  its candidate A's virtual-load
  formula is not a ready dispatch adapter, since night import and grid
  charging can add to `Import_From_Grid` beyond unmet virtual load
  (`breos/_dispatch.py:1046-1085`).
- **A13. Current preparation names, defaults and rollout.** PR 2–3 use
  `build_pv_production_breakdown(...).dc_after_losses`
  (`breos/app_inputs.py:474-535`), not the removed `build_dc_system_base`;
  App's arrays share one PV engine. Weather loading goes through
  `load_weather_for_simulation` (`:334-439`) and `read_weather_csv`
  (`breos/weather.py:419-434`), preserving source timing and the common
  calendar. `solar_position = "auto"` is now the default
  (`breos/pv/model_options.py:90`, `:288-300`); the runner records its
  resolution and supplies per-step inverter fields each year
  (`breos/runners/app.py:658-686`). PR 5's examples and Q1 baseline must use
  these defaults. PR 2 inherits App's AC rating/efficiency/price checks
  (`breos/app_config.py:2944-2984`, `:3332-3338`) and its refusal of a DC
  battery or daily-target planner beside several string-level inverters
  (`:3544-3559`), rather than adding a different ACC model. The five-PR
  order and rollout gates stand; PR 3 must budget A3/A8 and PR 4 A7/A10.
  #419 and #417–#418 are already implemented at this revision. #420 is
  still a prerequisite under decision 11; this check specifies no vehicle
  flows or carry that are not present yet, and those seams need rechecking
  after its implementation.

## Decisions

Taken by the maintainer on 2026-10-06. The body sections follow them.

1. **Entry point:** `breos.Community` in `breos/community/`; App untouched
   (K2).
2. **Resolution:** 15-minute steps only; an hourly community run is refused
   (K3).
3. **Allocation:** fixed coefficients that sum to 1, renormalised over the
   receivers in each interval, and coefficients proportional to metered
   import; no reallocation of participant surplus; dynamic allocation later,
   as an experimental mode (K4).
4. **Order:** the batteries behind each meter first, then settlement on
   metered balances (K5).
5. **No community battery in 0.8.0.** 0.8.0 shares PV only: the assets'
   output and the participants' pooled export. Batteries behind a
   participant's meter still work. Three community storage models are
   recorded for a later amendment (K10). Changed from the proposal, which
   had one community battery.
6. **Participants' export:** pooled by default; `pool_participant_export =
   false` turns it off (K5).
7. **Prices:** one shared-energy price per kWh, by period if needed, paid to
   the injecting sites; `shared_energy_price`, `shared_network_charge` and
   `surplus_price` required with no default, 0 allowed; one network charge
   for the community (K3, K7).
8. **Ownership:** shares over participants, or an external owner (K7).
9. **References:** each participant alone with the same equipment and
   tariff, plus the no-system baseline (K8).
10. **Out of 0.8.0:** Monte Carlo, the optimizer, hierarchical sharing and
    `revalue` for communities (K1, K9).
11. **Release and order:** ships in 0.8.0, implemented after #419 and #420
    (see [Implementation plan and effort](#implementation-plan-and-effort)).
    Changed from the proposal, which had no order.

Decided 2026-10-22:

12. **Inverter-draw netting belongs in App.** A separate App change on its
    own branch nets an idle PV inverter's night draw against AC bus surplus
    before export in the PV-only and AC-battery paths, preventing simultaneous
    import and export caused by inverter draw. ACC depends on that fix and
    settles App's meter balances directly (K4–K6); Q3 holds exactly, without
    an exception (A6).
13. **Shared asset imports are billed.** Record actual asset imports and
    require an explicit import price, with no default and 0 allowed;
    supply-contract fixed charges are likewise explicit, required when the
    asset has a supply contract, with no default and 0 allowed. Split these
    costs by ownership like the asset's other costs and revenues (K7).
    The asset has no invented no-system baseline (A7).
14. **Households without PV are supported through Community.** Community
    builds their 15-minute meter import from their load, with zero export,
    and uses App's tariff code; sharing is a balance on metered kWh, not
    routed energy. Their alone reference bills the same load on the same
    tariff without sharing, as today's no-system billing does. Q2 checks
    the load-only meter flows; Q3 checks that no-sharing bill. App's
    validation is unchanged (A2).
15. **Annual network-credit caps use supplier network charges only.** Cap
    the credit against the household's remaining supplier network charges,
    including the fixed part, as App does. Report any change caused by
    sharing as a separate annual line (A10).

## Consequences

- A community of households sharing PV, with or without batteries behind
  their meters, can be simulated and settled per 15-minute interval with
  each participant's own tariff, and every participant's benefit is
  reported against the same household alone.
- `breos.App` and its results are unchanged; the community is a second
  facade built on the same single-site physics.
- The settlement ledger is a new frame with its own schema version; the
  dispatch ledger and the result format do not change.
- Community storage, hierarchical and dynamic sharing, community Monte
  Carlo and optimization stay outside 0.8.0; K10 records the storage models
  a later amendment chooses from.
