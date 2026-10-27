# Electrical solver diagnostics

Single-diode results pass through `breos.pv._solver`. The primary method stays
Newton for MPPs and Lambert W for IV evaluations. Only invalid lit points use
Brent's method, after which the same checks apply. Above open circuit, the
voltage fallback brackets the junction voltage explicitly: pvlib's default
Brent bracket does not include negative currents. Failed fallbacks raise
`ValueError` with the solve stage, original step, equipment and five diode
parameters. Non-finite diode parameters at lit or indeterminate-light steps
raise a separate input `ValueError` before solving, without fallback or recovery
counts. Non-finite parameters at known dark steps may still produce zero.
No peak constants change.

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
