# Electrical solver diagnostics

Single-diode results pass through `breos.pv._solver`. The primary method stays
Newton for MPPs and Lambert W for IV evaluations. Invalid lit points retry
with Brent's method, after which the same checks apply. Above open circuit, the
voltage fallback brackets the junction voltage explicitly: pvlib's default
Brent bracket does not include negative currents. Its analytic upper endpoint
is stepped outward by a factor of 1 + 1e-8 when roundoff leaves the residual
positive, with at most eight expansions. Failed fallbacks raise
`ValueError` with the solve stage, original step, equipment and five diode
parameters. Non-finite diode parameters at lit or indeterminate-light steps
raise a separate input `ValueError` before solving, without fallback or recovery
counts. Non-finite parameters at known dark steps may still produce zero.
No peak constants change.

## Physical dark floor

`DARK_FRACTION = 1e-9` in `breos.pv._solver` defines a shared photocurrent
floor. `floor_photocurrent` sets only photocurrent to exactly zero when it is
strictly below `DARK_FRACTION * I_L_ref`; equality and larger values are kept.
The reference is the module's CEC reference photocurrent from `_get_cec_params`,
approximately its reference short-circuit current. Both runtime
`calcparams_cec` generation sites apply the helper: `IrradianceParts.diode_parameters`
and `solar._module_dc_before_losses`. Substrings, chains and half-cells inherit
the resulting zero through their existing parameter shares. They do not apply
a second threshold to their smaller currents. Other diode parameters are kept.

The floor is approximately 1e-6 W/m² equivalent irradiance. It ends numerical
excursions at the floating-point limits of the diode equation and changes no
realistic reported result. The originally failing 1e-13–1e-15 W/m² cases carry
less than a nanowatt-hour per 400 W module-year. That bound concerns those
failures, not every irradiance below the new floor: even holding a 400 W
module just below the floor for a whole year has a conservative constant-
efficiency energy bound of about 0.0035 Wh, a few milliwatt-hours. The App
goldens and first_home gallery remain unchanged.

Dark MPPs and open-circuit voltages are exactly zero. Dark parts retain their
IV curve: negative current can produce positive voltage, positive current can
produce negative voltage through the shunt, and the represented breakdown or
bypass clamp still applies. If Lambert W gives an invalid dark reverse-current
voltage, the ideal-diode endpoint is checked first; negligible leakage or
infinite Rsh makes it the root. Otherwise the checked bracket solves it.
Dark roots never count as recovered failures and never raise a solver error.
Their retries are separate from lit retries, so an infeasible dark forward
point cannot spoil a lit point's solution.

The robustness sweep covers 434 power results: 31 irradiance scales, 1e-30
through 1, with two steps for each of four represented layouts, a single array,
parallel arrays and parallel shaded strings. Every result is finite and
non-negative. The worst decrease as irradiance increases is zero watts;
all steps below the floor have zero power and zero recoveries. Some lit
points above the floor still use the checked solver fallback (up to ten
scale-step recoveries in the parallel-string sweep). The existing 1e-16 and
1e-18 scale regressions are now dark, with zero recoveries.

The MPP check uses finite values, power consistency, analytic Voc/Isc upper
bounds and the diode equation residual, with small roundoff tolerances. A
positive voltage and current on that equation imply V ≤ Voc and I ≤ Isc.
This avoids extra endpoint solves on the valid path and remains accurate at
vanishing light, where Lambert W can cancel Voc to zero. IV checks allow
physical negative currents and voltages. MPP power consistency and diode
residual checks use 1e-7 relative and 1e-8 absolute tolerances. The endpoint
bounds allow 1e-7 times the larger of 1 and the bound; MPP signs allow 1e-8
absolute roundoff near zero. The bypass mask excludes points already replaced by a breakdown clamp. Non-finite dark
results alone may become zero without recovery.

Constrained solves establish voltage feasibility against Voc before evaluating
an operating point, and check current limits against Isc before inversion.
Parallel searches skip wholly infeasible windows and keep discarded rows at
bounded voltages in mixed batches. A window above Voc returns zero without
recording a recovery.

IV validation evaluates the single-diode residual for both primary and fallback
values. With junction voltage U = V + I Rs, it checks
`Io exp(U/n) + U/Rsh + I - IL - Io`, scaled by the largest of `|IL|`, `|I|`,
`|Io exp(U/n)|`, `|U/Rsh|` and 0.1 A. The relative tolerance is 1e-7, giving
an absolute floor of 1e-8 A. This admits valid vanishing-light roundoff and
large currents while rejecting finite roots that violate the diode equation.
The calculation uses one vectorised exponential and reuses its array buffers.

The collector belongs to PV preparation. Its plain result travels in
`PVProductionBreakdown`, including cached preparations, then in
`SimulationArtifacts` to `electrical.solver`. Revaluation retains it. Counts
are distinct original timesteps, overall and by stage, rather than solve calls
or repeated projection years. It does not mutate the design topology or leak
between runs. Monte Carlo keeps the same records in the PV year cache and
returns them as `solver.by_weather_year`, keyed by source year, with their
sum in `solver.total`. Aggregation happens in the parent, independently of
worker start method or trajectory sampling. CLI JSON and provenance files
carry the same block. See the result guide for keys and stage names.

## Sanitisation audit

All former `nan_to_num` sites in these two files now consume checked results:

| File and method | Checked values |
|---|---|
| `pv/mismatch.py`, `_uniform_curve` | Bare and cabled MPP power, voltage and current; Voc |
| `_substring_elements.element` | Part short-circuit current |
| `_half_section.element` | Half-chain part short-circuit current |
| `_uniform_constrained` | Voltage at a current limit; current at a voltage limit |
| `current` | Uniform string's IV evaluation |
| `own_mean_power` | Reference MPP power |
| `_part_voltage` | Forward part voltage, excluding the breakdown/bypass region |
| `_HalfSection.__init__` | Half-chain table voltages |
| `electrical.py`, `string_cable_loss` | Module MPP current, checked upstream in `solar.py` |
| `_StringGroup.operating` | Cabled Newton MPP; photocurrent no longer sanitised for the lit mask |
| `_StringGroup.power_at_mpp` | MPP voltage/current, checked at the module or cable solve |
| `_single_array_power` | Checked MPP values; voltage/current at binding limits |
| `_parallel_arrays_power` | Checked MPP values, IV currents and Voc |
| `_GroupSource.__init__`, `current` | Voc and IV current |
| `_open_circuit_voltage` | Photocurrent lit mask, Voc and summed IV current |

`solar._module_dc_before_losses` also routes its suppressed Newton MPP through
the helper. The irradiance/temperature and CEC parameter warning scopes in
`solar.py` remain: these calculate inputs, not operating points. The unrelated
irradiance, ratio and inverter-power sanitisation there is outside this change.

The remaining `errstate(divide="ignore", invalid="ignore")` scopes in the
two target files cover explicit guarded ratios: concentration with zero
limiting shade; zero power/current and availability; and repeated points in
interpolation tables. They do not wrap a solver. Runtime warning suppression
for the solvers themselves is centralised in `_solver._quiet`.

## Search evidence

`tests/test_solver_convergence.py` generates represented chain means and
limiting fractions directly. It covers 2, 3 and 6 bypass diodes, full and
half-cut chains, no breakdown and −3, −5.5 and −15 V breakdown, diffuse and
fully dark cells, competing bypass peaks, cables, binding limits and free
operation. Three differently shaded strings share each parallel input.

The reference searches 8,193 points over the whole feasible range and polishes
the best grid point. String references also include both sides of each
represented breakdown discontinuity and solve voltage boundaries independently.
They evaluate the same part/half-chain curve as production; parallel references
evaluate the same string tables as production's refinements. Neither uses the
cell reference or changes the shading representation.

The default sweep uses seed 80621 (80622 and 80623 for the additional parallel
strings): 128 string and 128 parallel comparisons, plus three comparisons per
search with the global optimum at the voltage edge. The declared relative
tolerance is 1e-4, with 1e-6 W absolute tolerance near zero. This is empirical
convergence evidence for these represented cases, not a proof for every input.

On the seeded sweep, the worst relative shortfall was 3.25e-10 for strings
and 5.20e-16 for parallel inputs. The largest relative excess was 3.27e-15
and 2.47e-14 respectively. The voltage-edge cases also passed. No case
missed the declared tolerance, so neither peak count was raised and no new
expected failure was needed.

## Peak-ranking regressions

Two represented inputs rank their true best peak below another sampled peak.
Both are generated by `represented` in `tests/test_solver_convergence.py` with
64 steps, then isolated to one step before constructing the curve. This keeps
the tests small without changing the represented electrical model. All strings
have four modules, a 0.2 ohm cable and no binding inverter limits.

| Search | Represented input | Dense reference (W) | One peak (W) | Production (W) | Relative gap at one / production |
|---|---|---:|---:|---:|---|
| String | seed 1001, step 52; 3 diodes, 2 parallel halves, −3 V breakdown | 468.042956773 | 466.292274948 | 468.042956690 | 0.003740429 / 1.77e-10 |
| Parallel | seeds 1003–1005, step 41; 6 diodes, 1 chain, −15 V breakdown | 502.255755081 | 501.748340868 | 502.255755081 | 0.001010271 / roundoff |

The string's top sampled peaks are 466.2923 W and 465.7380 W; the latter's
bracket refines to the global 468.0430 W peak. Beam irradiance is 647.5342 W/m²,
sky diffuse 80 W/m², temperature 2.6351 °C and current scale 0.876704.
The parallel input's top sampled peaks are 501.6397 W at 102.0498 V and
501.4856 W at 73.7027 V. The second refines to 502.2558 W at 73.9281 V,
beating the broad peak's 501.7483 W. Its sources' beam irradiances are
545.6506, 564.7805 and 385.3020 W/m², with no diffuse light; temperatures
are 24.3954, −3.0563 and 46.2691 °C and scales 0.860487, 0.877083 and 0.880274.

A sensitivity test reduces each peak count to one independently and asserts
that these same convergence assertions fail. Neither production peak count
has changed, and neither case misses the production tolerance.
