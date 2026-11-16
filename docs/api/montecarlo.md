# Monte Carlo

The Python interface behind `breos montecarlo`. A study takes an App
configuration dict and a {py:class}`~breos.montecarlo.MonteCarloSettings`
with the `[montecarlo]` controls, and returns a
{py:class}`~breos.montecarlo.MonteCarloResult` with one row per run in
`runs`, the statistics in `summary`, the P50/P90 annual yield in
`yield_exceedance`, `solver` recovery counts by source weather year and in
total, and `provenance`. The
[Monte Carlo guide](../getting-started/monte-carlo.md) explains the
settings, the outputs and the reuse of one weather file across designs.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.montecarlo.run_montecarlo
   breos.montecarlo.MonteCarloSettings
   breos.montecarlo.MonteCarloResult
   breos.montecarlo.build_year_cache
   breos.montecarlo.MonteCarloYearCache
   breos.exceedance.yield_exceedance
   breos.exceedance.YieldExceedance
```
