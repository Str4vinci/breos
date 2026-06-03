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

from typing import Any

import pandas as pd

from breos.app_config import (
    DEFAULTS as _DEFAULTS,
)
from breos.app_config import (
    ResolvedAppConfig,
    build_costs_dict,
    normalise_pv_arrays,
    resolve_app_config,
    resolve_costs,
    validate_config,
)
from breos.app_config import (
    load_json as _load_json,
)
from breos.app_inputs import (
    AppRuntimeDependencies,
    load_weather_for_simulation,
)
from breos.app_inputs import (
    remap_tmy_year as _remap_tmy_year,
)
from breos.app_results import build_result as build_app_result
from breos.app_results import financial_to_dicts, monthly_to_dicts, yearly_to_dicts
from breos.app_simulation import run_app_simulation
from breos.load_profiles import load_profile
from breos.weather import build_battery_temperature_series, fetch_tmy_weather_data, load_weather, resample_to_15min


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
        profile, tracking, resolution, projection years, cost and emissions
        presets, degradation, and inverter assumptions.
    """

    def __init__(self, config: dict) -> None:
        self._resolved = resolve_app_config(config)
        self._cfg = self._resolved.cfg
        self._sync_legacy_attrs(self._resolved)
        self._result: dict[str, Any] | None = None

    def simulate(self) -> None:
        """Run the full simulation pipeline."""
        artifacts = run_app_simulation(self._cfg, self._resolved, self._runtime_dependencies())
        self._result = build_app_result(self._cfg, self._resolved, artifacts)

    def result(self) -> dict[str, Any]:
        """
        Return simulation results as a plain dict.

        Raises ``RuntimeError`` if :meth:`simulate` has not been called.
        """
        if self._result is None:
            raise RuntimeError("Call simulate() before result().")
        return self._result

    def _sync_legacy_attrs(self, resolved: ResolvedAppConfig) -> None:
        """Expose existing private attributes for in-repo tests and callers."""
        self._lat = resolved.lat
        self._lon = resolved.lon
        self._tz = resolved.timezone
        self._loc_key = resolved.loc_key
        self._pv_arrays = resolved.pv_arrays
        self._pv_params = resolved.pv_params
        self._avg_module_power_w = resolved.avg_module_power_w
        self._system_kwp = resolved.system_kwp
        self._tilt = resolved.tilt
        self._azimuth = resolved.azimuth
        self._tracking = resolved.tracking
        self._axis_azimuth = resolved.axis_azimuth
        self._cost_params = resolved.cost_params
        self._emissions_params = resolved.emissions_params

    @staticmethod
    def _runtime_dependencies() -> AppRuntimeDependencies:
        return AppRuntimeDependencies(
            load_profile=load_profile,
            load_weather=load_weather,
            fetch_tmy_weather_data=fetch_tmy_weather_data,
            resample_to_15min=resample_to_15min,
            build_battery_temperature_series=build_battery_temperature_series,
        )

    # Compatibility wrappers for private helpers that used to live here.
    @staticmethod
    def _validate(cfg: dict) -> None:
        validate_config(cfg)

    def _normalise_pv_arrays(self, arrays: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        return normalise_pv_arrays(arrays, self._cfg, self._lat)

    def _load_weather(self, freq: str, start_year: int) -> pd.DataFrame:
        return load_weather_for_simulation(self._resolved, freq, start_year, self._runtime_dependencies())

    @staticmethod
    def _resolve_costs(cfg: dict):
        return resolve_costs(cfg)

    def _build_costs_dict(self) -> dict[str, float]:
        return build_costs_dict(self._cfg, self._resolved)

    @staticmethod
    def _monthly_to_dicts(results_df: pd.DataFrame, freq: str) -> list[dict[str, Any]]:
        return monthly_to_dicts(results_df, freq)

    @staticmethod
    def _financial_to_dicts(cost_proj: pd.DataFrame, total_initial_cost: float) -> list[dict[str, Any]]:
        return financial_to_dicts(cost_proj, total_initial_cost)

    @staticmethod
    def _yearly_to_dicts(yearly_df: pd.DataFrame) -> list[dict[str, Any]]:
        return yearly_to_dicts(yearly_df)
