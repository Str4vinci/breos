"""
BREOS public facade - single entry point for PV + battery simulations.

Usage:
    import breos

    app = breos.App({
        "location": "porto",
        "n_modules": 10,
        "annual_consumption_kwh": 4000,
        "battery_kwh": 5.0,
        "cost_preset": "residential_pt",
        "emissions_country": "PT",
    })
    app.simulate()
    result = app.result()
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from breos.app_config import APP_CONFIG_FIELDS, resolve_app_config
from breos.app_inputs import AppRuntimeDependencies
from breos.app_results import build_result as build_app_result
from breos.load_profiles import load_profile
from breos.repair import input_repair_records
from breos.runners.app import SimulationArtifacts, revalue_app_simulation, run_app_simulation
from breos.utils import deep_merge
from breos.weather import build_battery_temperature_series, fetch_tmy_weather_data, load_weather, resample_to_15min

# The keys App.revalue may change: the economics section of the resolved
# configuration, except the horizon, which sets how many years are simulated.
REVALUATION_KEYS = frozenset(
    key
    for key, field in APP_CONFIG_FIELDS.items()
    if field.summary is not None and field.summary.startswith("economics.") and key != "projection_years"
)


class App:
    """
    Single entry point for BREOS simulations.

    Parameters
    ----------
    config : dict
        Simulation configuration. Required keys:

        - ``location`` - preset key (``"porto"``) **or** dict with
          ``latitude``, ``longitude``, ``timezone``.
        - ``n_modules`` - number of PV modules (int, > 0), unless
          ``pv_arrays`` is provided.
        - ``annual_consumption_kwh`` - yearly electricity demand (float, > 0).

        Optional keys include battery size, PV arrays, module selection, load
        profile, tracking, sky-diffusion model (``transposition_model``),
        custom terrain shading (``horizon_profile``),
        opt-in bifacial rear gain (``bifacial_model`` plus row geometry),
        resolution, projection years, cost and emissions presets, degradation,
        and inverter assumptions.
    input_repairs : InputRepairReport, dict, or list of them, optional
        Reports from :func:`breos.io.repair_series <breos.repair.repair_series>` for input series repaired
        before this run, for example a measured load written to the
        ``rlp_directory`` file the config points at. They are recorded
        unchanged in ``result()["provenance"]["input_repairs"]``. App does not
        repair anything itself and does not check the reports against the
        input it loads. Omitted (None), provenance has no ``input_repairs``
        key, exactly as before.
    """

    def __init__(self, config: dict, *, input_repairs: Any = None) -> None:
        self._config = deepcopy(config)
        self._resolved = resolve_app_config(config)
        self._cfg = self._resolved.cfg
        self._input_repairs = input_repair_records(input_repairs)
        self._result: dict[str, Any] | None = None
        self._artifacts: SimulationArtifacts | None = None

    def simulate(self) -> None:
        """Run the full simulation pipeline."""
        artifacts = run_app_simulation(self._cfg, self._resolved, self._runtime_dependencies())
        self._result = build_app_result(self._cfg, self._resolved, artifacts, input_repairs=self._input_repairs)
        self._artifacts = artifacts

    def revalue(self, changes: Mapping[str, Any]) -> dict[str, Any]:
        """Return the result this run would give at other prices, without changing this App.

        ``changes`` holds configuration keys, merged into this App's
        configuration as a CLI override is: a nested table such as ``costs``
        or ``tariff`` replaces only the keys it sets. Only the economics keys
        in :data:`REVALUATION_KEYS` may change: ``costs``, ``cost_preset``,
        ``tariff``, the discount rate and the escalators.

        When the new prices cannot change the dispatch, the stored simulation
        is re-priced: flat prices, a tariff removed, or a tariff on the same
        schedule whose smart-charging instructions stay the same. Otherwise (a
        tariff added, a different schedule) the run is simulated again. The
        result records which in ``provenance["revaluation"]``, with the keys
        that changed. A flat-price revaluation gives the same floats as a new
        simulation; a re-priced tariff sums energy by period instead of by
        step, so it agrees to rounding.

        Raises:
            RuntimeError: If :meth:`simulate` has not been called.
            ValueError: If ``changes`` sets a key outside
                :data:`REVALUATION_KEYS`; build a new App for those.
        """
        if self._artifacts is None:
            raise RuntimeError("Call simulate() before revalue().")
        config = deep_merge(self._config, dict(changes))
        changed = sorted(key for key in config.keys() | self._config.keys() if config.get(key) != self._config.get(key))
        outside = [key for key in changed if key not in REVALUATION_KEYS]
        if outside:
            raise ValueError(
                f"revalue() changes prices only; {', '.join(outside)} would change the simulation. "
                f"Build a new App for that. Keys revalue() accepts: {', '.join(sorted(REVALUATION_KEYS))}."
            )
        resolved = resolve_app_config(config)
        artifacts, method = revalue_app_simulation(
            resolved.cfg, resolved, self._artifacts, self._runtime_dependencies()
        )
        result = build_app_result(resolved.cfg, resolved, artifacts, input_repairs=self._input_repairs)
        result["provenance"]["revaluation"] = {"method": method, "changed_keys": changed}
        return result

    def result(self) -> dict[str, Any]:
        """
        Return simulation results as a plain dict.

        Raises ``RuntimeError`` if :meth:`simulate` has not been called.
        """
        if self._result is None:
            raise RuntimeError("Call simulate() before result().")
        return self._result

    @staticmethod
    def _runtime_dependencies() -> AppRuntimeDependencies:
        return AppRuntimeDependencies(
            load_profile=load_profile,
            load_weather=load_weather,
            fetch_tmy_weather_data=fetch_tmy_weather_data,
            resample_to_15min=resample_to_15min,
            build_battery_temperature_series=build_battery_temperature_series,
        )
