# Third-Party Module Wrapping

**Status:** Proposed — not yet implemented
**Tracked by:** [#11](https://github.com/Str4vinci/breos/issues/11)
**Owner:** TBD

## Problem

BREOS imports several third-party libraries directly throughout the package
(`pvlib`, `pandas`, `numpy`, `scipy`, `numba`, `rainflow`, `geopy`,
`openmeteo-requests`, `timezonefinder`, `nrel-pysam`, `pymoo`,
`requests-cache`, `joblib`, `matplotlib`, `openpyxl`).

Direct usage means those libraries co-own the BREOS public API. When they
change — and `pvlib` in particular has historically broken APIs across
minor releases — every call site has to be updated and every consumer of
BREOS may break. Concretely today:

- `pvlib.Location`, `pvlib.PVSystem`, and `pvlib.irradiance` / `pvlib.iam`
  / `pvlib.temperature` / `pvlib.ivtools` / `pvlib.pvsystem` /
  `pvlib.inverter` / `pvlib.iotools` are referenced from at least 5
  modules.
- `Location` from `pvlib.location` appears in the public signatures of
  `solar.calculate_pv_production_dc`, `solar.calculate_pv_production_ac`,
  `solar.calculate_multi_array_production`, `acc_sizer`, `app`, and
  `optimization`. Anyone calling BREOS must construct a `pvlib.Location`
  themselves.
- `rainflow` is imported in `battery.py`, `scipy.optimize.minimize` in
  `acc.py`, `scipy.interpolate.Akima1DInterpolator` in `weather.py`.

## Goal

Own the BREOS public API. Concentrate every third-party touchpoint in a
small adapter layer so that:

1. Library upgrades (e.g. `pvlib` 0.14 → 0.15) require changes in a single
   place rather than across the package.
2. The BREOS surface area stays narrow and domain-specific (we use ~10
   pvlib calls; we don't need to expose pvlib's hundreds of routines).
3. Alternative implementations (e.g. swap `scipy.optimize.minimize` for a
   different solver, swap `pvlib` for a future in-house model) become a
   single-class change.

## Pattern

For each wrapped concept, introduce:

```python
# breos/adapters/location.py
from abc import ABC, abstractmethod

class SolarPosition: ...   # plain BREOS dataclass

class Location(ABC):
    @abstractmethod
    def solar_position(self, times: TimeIndex) -> SolarPosition: ...

class PvlibLocation(Location):
    def __init__(self, latitude, longitude, tz, altitude=0):
        import pvlib
        self._inner = pvlib.location.Location(latitude, longitude, tz, altitude)

    def solar_position(self, times: TimeIndex) -> SolarPosition:
        df = self._inner.get_solarposition(times=times.to_pandas())
        return SolarPosition.from_pvlib(df)
```

Core BREOS modules depend only on the abstraction (`Location`). Concrete
implementations (`PvlibLocation`) are instantiated at the system boundary
(CLI, `app.py`, tests) and injected.

```python
# breos/cli.py (composition root)
location: Location = PvlibLocation(lat, lon, tz)
run_simulation(location, ...)
```

## Scope and phases

Wrapping every dependency at once is a multi-week refactor. Recommended
phasing, ordered by API-churn risk × surface area:

### Phase 1 — pvlib (highest priority)

Narrow surface (~10 calls), high churn risk, already shows up in public
signatures.

| Concept            | Wrapped in                       | Replaces                                         |
| ------------------ | -------------------------------- | ------------------------------------------------ |
| `Location`         | `breos.adapters.location`        | `pvlib.location.Location`                        |
| `PVSystem`         | `breos.adapters.pv_system`       | `pvlib.pvsystem.PVSystem`                        |
| `irradiance.aoi`   | `breos.adapters.irradiance`      | `pvlib.irradiance.aoi`, `pvlib.iam.ashrae`       |
| `cell_temperature` | `breos.adapters.thermal`         | `pvlib.temperature.faiman`                       |
| `cec_model`        | `breos.adapters.pv_model`        | `pvlib.ivtools.sdm.fit_cec_sam`, `calcparams_cec`, `max_power_point`, `pvwatts_losses` |
| `inverter`         | `breos.adapters.inverter`        | `pvlib.inverter.pvwatts`                         |
| `weather_io`       | `breos.adapters.weather_io`      | `pvlib.iotools.get_pvgis_tmy`, `get_nsrdb_psm4_tmy`, `read_epw` |

### Phase 2 — small scientific deps

- `rainflow` → `breos.adapters.cycle_counting` (used in `battery.py`).
- `scipy.optimize.minimize` → `breos.adapters.optimizer` (used in
  `acc.py`).
- `scipy.interpolate.Akima1DInterpolator` → `breos.adapters.interpolation`
  (used in `weather.py`).
- `numba.jit/prange` → keep direct (perf-critical, tightly coupled to
  kernel internals; abstracting adds no portability value).

### Phase 3 — IO / external services

- `openmeteo-requests`, `requests-cache` → `breos.adapters.weather_client`.
- `geopy`, `timezonefinder` → `breos.adapters.geo`.
- `nrel-pysam` → `breos.adapters.sam_model` (if/when used beyond optional
  paths).
- `pymoo` → `breos.adapters.multi_objective` (used in `optimization.py`).
- `openpyxl` → encapsulated inside `breos.io` already; keep as-is.

### Out of scope: pandas and numpy

`pandas.DataFrame`, `pandas.Series`, `numpy.ndarray` are treated as data
primitives, not wrapped. Rationale:

- They are the lingua franca of the scientific Python ecosystem; wrapping
  them would force every caller to convert at the boundary.
- Their APIs are stable across years.
- The cost (rewriting every module + every test + every example) far
  exceeds the insulation benefit.

If we later want stronger schema guarantees, introduce typed wrappers at
specific boundaries (e.g. a `WeatherFrame` dataclass that validates
columns) rather than a global abstraction.

`matplotlib` is also kept direct — `plotting.py` is intentionally an
output module, not a load-bearing core dependency.

## Migration mechanics

1. Create `breos/adapters/` package, one module per concept.
2. Add the abstraction + the `Pvlib*` (etc.) implementation side by side.
3. Migrate call sites one module at a time. Each migration is a small PR.
4. Add a `ruff` rule (or simple CI grep) that blocks new direct imports of
   wrapped libraries outside `breos/adapters/`.
5. Update tests to inject fakes via the abstraction instead of patching
   `pvlib`.
6. Once all call sites are migrated, document the public API as the
   adapter layer.

## Risks and non-goals

- **Performance:** the wrapper layer must not introduce per-timestep
  overhead. Keep wrappers thin — pass arrays through, do not iterate.
- **Test churn:** existing tests import `pvlib.Location` directly. They
  will need to construct via the adapter or via a fake. Plan one PR per
  test module to spread the cost.
- **Not a full hexagonal rewrite.** This is API insulation, not a port +
  adapter restructure of the whole package. Domain logic stays where it
  is; only the dependency direction at the edges changes.
- **Not a replacement of pvlib.** We continue to use it; we just stop
  exposing it.

## Effort estimate

- Phase 1 (pvlib): ~1–2 weeks of focused work, 5–7 PRs.
- Phase 2 (scipy / rainflow): ~2–3 days, 2–3 PRs.
- Phase 3 (IO / external services): ~1 week, 4–5 PRs.

Total: ~3–4 weeks if pursued continuously; more realistic as a background
refactor over a couple of months.
