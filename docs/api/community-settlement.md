# Community settlement and results

`Community.settlement()` returns the first project year's 15-minute meter
settlement, independent of App's [dispatch ledger](energy-balance.md#ledger-schema).
It returns a copy, including frame attributes; editing it does not edit the
stored settlement. `attrs["settlement_schema_version"]` is **`"1.0"`**,
`energy_unit` is `"Wh"`, `interval_minutes` is 15, and `project_year` is 1.
Dispatch power has already been converted once: Wh = average W × 0.25.

## Index and columns

The MultiIndex levels are `interval`, `site_type`, `site_id`. Every interval
contains all sites in sorted id order, then a community row (`site_type =
"community"`, `site_id = ""`). Site types are `"participant"` and `"asset"`.
The timezone-aware interval labels follow the common simulation/weather
clock. Project years replay that commissioning calendar; they are not
restamped to different civil years.

| Column | Unit | Meaning |
|---|---|---|
| `Metered_Import` | Wh | Site import after its own equipment, including inverter draw and grid charging. |
| `Metered_Export` | Wh | Site export after its own equipment. A site never imports and exports in the same interval. |
| `Pooled_Injection` | Wh | Export offered for sharing: every asset's export and participants' exports when pooling is enabled. |
| `Coefficient` | fraction | Receiver coefficient for this interval, after excluding exporters and renormalising; zero for assets. |
| `Shared_Allocated` | Wh | Pooled energy allocated by that coefficient. |
| `Shared_Used` | Wh | `min(Shared_Allocated, Metered_Import)`. |
| `Participant_Surplus` | Wh | `Shared_Allocated - Shared_Used`; never redistributed. |
| `Supplier_Import` | Wh | Remaining import bought from the site's supplier. Assets' actual import remains here in full. |
| `Own_Export` | Wh | Unpooled participant export sold under its own export price; zero on assets. |
| `Attributed_Shared_Used` | Wh | Total shared use attributed to this injector in proportion to its pooled injection. |
| `Attributed_Surplus` | Wh | Community surplus attributed by the same injection shares. |
| `Energy_For_Sharing` | Wh | Community row only: total pooled injection. |
| `Shared_Used_Total` | Wh | Community row only: total shared use. |
| `Community_Surplus` | Wh | Community row only: pooled injection minus use, including unallocated energy. |

The first eleven columns are zero on community rows. The last three are
zero on site rows. All numerical columns are float64; only `Coefficient`
is dimensionless. Asset imports are recorded and billed but never receive
a share. An exporting participant receives none even if pooling is disabled.

## Energy closure

Per interval, let `S` be pooled injection and sum over the corresponding
site or participant rows:

1. `S = Σ Pooled_Injection = Energy_For_Sharing`.
2. `Σ Shared_Allocated = S` when the receiver denominator is positive,
   otherwise zero. For each receiver,
   `Shared_Allocated = Shared_Used + Participant_Surplus`.
3. `Community_Surplus = S - Σ Shared_Used`, also equal to
   `Σ Participant_Surplus + S - Σ Shared_Allocated`.
4. `Metered_Import = Shared_Used + Supplier_Import`, with
   `Shared_Used ≤ min(Shared_Allocated, Metered_Import)`.
5. `Σ Attributed_Shared_Used = Σ Shared_Used` and
   `Σ Attributed_Surplus = Community_Surplus`.
6. `Metered_Export = Pooled_Injection + Own_Export`.

All identities also hold for annual sums, to floating-point precision.
Annual net imports/exports cannot reconstruct interval receivers: annual
totals sum already-settled intervals. The receiver denominator is the sum
of eligible fixed coefficients or eligible metered imports, respectively.
An eligible zero-import household alone does not create a positive
proportional denominator.

## Public result

`Community.result()` returns `CommunityResult`, defined in
`breos.community.results`. Account types are reused from
`breos.community.money`. Fields are unrounded; all money uses `currency`.
Mappings are in sorted site-id order, years in project order. The dataclass
freezes field assignment, not DataFrames or the provenance dict.

| Field | Content |
|---|---|
| `currency` | Shared ISO currency code; no currency conversion. |
| `participants` | Id → `ParticipantAccount`, including owned asset shares once. |
| `assets` | Id → whole `AssetAccount`, including external assets; do not add these accounts to participant totals again. |
| `yearly` | Community nominal money, energy, coverage and closure per project year. |
| `totals` | Participant/community NPVs, initial investment, reference costs and separate external-asset NPC. |
| `ownership` | Asset id → participant shares or `"external"`. |
| `settlement_totals` | Project year → site/community Wh sums, indexed by `site_type`, `site_id`; `Coefficient` is not summed. |
| `interval_money` | First-year money ledger, indexed by interval/type/id, with site rows only. |
| `interval_closure` | First-year shared-payment and boundary residuals, indexed by interval. |
| `provenance` | JSON-safe community settings, source and per-site records described below. |
| `shared_coverage_of_import` | Presented first: yearly shared use / metered participant import, coverage of household grid purchases bounded by 100%; zero for zero import. |
| `shared_coverage_of_load` | Yearly shared use / gross participant load, as a Series; can exceed 100% with battery grid charging; zero for zero load. |

There is no unqualified `coverage` field. Both ratios describe shared use,
not own-PV self-consumption. Settlement caps use at metered import, which
includes battery grid charging and inverter draw. This bounds import
coverage by 100%; gross-load coverage can exceed 100% when shared use also
offsets battery grid charging. Both properties return yearly Series.

### Participant and asset accounts

`ParticipantAccount` contains `costs`, `owned_asset_investment`, `energy`,
`yearly`, `cashflows`, `alone_cashflows`, `npv_savings`, `npv_alone`,
`community_benefit`, `benefit_parts`, `benefit_residual`, `terminal_health`
and `terminal_adjusted_npv`.

- `energy`: `Year`, the site's settlement sums in Wh, `Load_Wh` and
  `Own_PV_Production_Wh`. Own PV production uses App's delivered-PV year
  definition, including PV-origin battery delivery; it is separate from
  shared use and raw inverter production.
- `yearly`: physical/priced rows at year-1 prices. `Import_kWh` and
  `Export_kWh` are the supplier balances after sharing; `energy` preserves
  the original meter balances. `Import_Cost`, `Export_Revenue`,
  `Fixed_Charge`, baseline bills and optional network credit fields use
  App's pricing conventions.
- `cashflows`: App's nominal component costs, year-end discounted and
  cumulative costs/savings, plus `Cost_Shared_Energy`,
  `Cost_Shared_Network`, `Revenue_Shared_Energy`, `Revenue_Surplus`,
  `Network_Credit_Change`, `Asset_*` ownership components and
  `Benefit_*` annual nominal components. Asset CAPEX is added at t=0.
- `alone_cashflows`: identical physical site without sharing or asset
  ownership. `npv_alone` and `npv_savings` both compare with the same
  no-system baseline. `community_benefit = npv_savings - npv_alone`.
- `benefit_parts`: NPVs of `supplier_import_saving`,
  `own_export_revenue_change`, `shared_energy_payment`,
  `shared_network_charge`, `shared_energy_receipt`, `surplus_receipt`,
  `network_credit_change` and `asset_ownership`. The last includes initial
  investment. The sum closes to community benefit via `benefit_residual`.
- `terminal_health` and `terminal_adjusted_npv`: optional pack-only residual
  credit sensitivity, separate from unadjusted NPV and community benefit.

`AssetAccount` contains `costs`, `energy`, `yearly`, `cashflows` and
`net_present_cost`. Its complete investment, O&M, actual imports, fixed
charges and injection revenues are reported without a no-system baseline
or savings NPV. Participant accounts split these by ownership. External
asset costs remain outside the participants' community NPV.

Community `yearly` includes `Cost_System_Annual`, `Cost_Alone_Annual`,
`Cost_No_Sys_Annual`, shared payments/receipts/network fees, surplus revenue,
`External_Owner_Receipts`, `Supplier_Net_Cost`, `Equipment_Operation_Cost`,
`Replacement_Cost`, `Network_Credit_Change`, `Boundary_Cost`, residuals,
`Energy_For_Sharing_Wh`, `Shared_Used_Total_Wh`, `Community_Surplus_Wh`,
`Shared_Coverage_Of_Metered_Import` and `Shared_Coverage_Of_Load`.

`totals` keys are `npv_savings`, `npv_alone`, `community_benefit`,
`initial_investment`, `no_system_cost_npv`, `alone_cost_npv`,
`community_cost_npv`, `external_asset_net_present_cost`,
`terminal_health_credit_npv` and `terminal_adjusted_npv`.

### Money ledger and closure

`interval_money` columns, in the community currency at year-1 prices, are:

| Column | Meaning |
|---|---|
| `Supplier_Import_Cost` | Supplier import at the site's own price. |
| `Own_Export_Revenue` | Unpooled export at the site's own export price. |
| `Shared_Energy_Payment` | Shared use × shared energy price. |
| `Shared_Network_Charge` | Shared use × community network charge. |
| `Shared_Energy_Receipt` | Attributed shared use × shared price. |
| `Surplus_Revenue` | Attributed surplus × surplus price. |
| `Net_Variable_Cost` | Import − own export + shared payment + network − shared receipt − surplus. |

The money schema is separately tagged `money_schema_version = "1.0"`.
Fixed charges, annual network credits, O&M and replacements are booked
annually rather than spread across intervals. Shared payments/receipts and
network charges escalate at the import rate; own export and surplus at
the sell rate. Pooled energy is never also sold as own export.

`Shared_Residual` is receivers' shared payments minus all injectors' shared
receipts. `Boundary_Residual` compares participant/owned-asset net costs
with supplier costs + shared-network fees − surplus sales + external-owner
receipts. External assets' own supplier expenses are excluded from that
participant boundary. Annual boundary costs additionally include fixed
supplier charges, supplier-only network credits, O&M and replacement
outlays. Initial investment is in NPV, outside annual operating closure.
The signed `Network_Credit_Change` is after-sharing minus alone credit.

An explicit annual network credit on a producer-only asset is accepted,
capped against that asset's eligible supplier network charges and split by
ownership as before. Configuration emits one `UserWarning` per such asset,
naming its id: producer-only connections rarely qualify for a network credit
and BREOS does not judge eligibility. Participants do not receive this warning.

Per-participant annual cost savings close against the `Benefit_*` parts
through `Benefit_Residual`; their NPVs plus owned initial investment close
through `benefit_residual`. The community reports `Shared_Residual`,
`Boundary_Residual`, `Max_Interval_Shared_Residual` and
`Max_Interval_Boundary_Residual` each year. Residuals expose floating-point
roundoff; cash is not rounded or redistributed to conceal it.

### Provenance and retention

`provenance` records `breos_version`, settlement and money schema versions,
the resolved `community` settings (agreements, allocation coefficients and
shared clock/economics), `participant_ids`, `asset_ids`, sorted `site_ids`,
site configuration hashes, `allocation`, `ownership`, calendar bounds,
energy/power units and the retention policy.

`sites[id]` contains the resolved site configuration and source records.
PV sites retain App's weather/load, resolved tariff/reference, PV timing and
bifacial, degradation, execution/toolchain, economics, supported
smart-charging and optional terminal-credit provenance. They also record
resolved electrical topology. Load-only households record their load,
supplier/reference tariff and economics without invented PV/weather or
dispatch records. Provenance is JSON-safe, including `"inf"` for unlimited
equipment bounds; the account object itself is not a JSON dict.

The facade keeps first-year settlement/money/residual frames and every
year's totals. It releases the physical projections after pricing and
provenance. Internal runners retain an explicit all-year frame option for
studies that need it; the public facade's default avoids retaining those
large frames. Annual maximum interval residuals are checked before later
frames are released.
