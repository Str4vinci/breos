"""
BREOS public facade — single entry point for PV + battery simulations.

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

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from pvlib.location import Location

from breos.battery import BatteryConfig, apply_indoor_temperature_model, simulate_energy_balance
from breos.economics import (
    BATTERY_REPLACEMENT_COST_PER_KWH,
    CostParams,
    calculate_costs,
    calculate_lcoe,
    cost_analysis_projection,
    find_payback_year,
)
from breos.emissions import EmissionsParams, calculate_co2_savings
from breos.load_profiles import load_profile
from breos.pv_modules import MODULES, get_module
from breos.solar import (
    calculate_multi_array_production,
    calculate_pv_production_dc,
    calculate_pv_production_dc_tracking,
    estimate_optimal_tilt,
)
from breos.solar import (
    default_azimuth as default_azimuth_fn,
)
from breos.utils import get_hours_per_step
from breos.weather import extract_ambient_temperature, fetch_tmy_weather_data, load_weather, resample_to_15min

_CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")

# --- Defaults -----------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    "battery_kwh": 0.0,
    "pv_arrays": None,
    "pv_module": None,
    "load_profile": "6",
    "tilt": None,
    "azimuth": None,
    "tracking": "fixed",
    "axis_tilt": 0.0,
    "axis_azimuth": None,
    "max_angle": 60.0,
    "backtrack": True,
    "gcr": 0.35,
    "cross_axis_tilt": 0.0,
    "dual_axis_max_tilt": 90.0,
    "resolution": "h",
    "projection_years": 20,
    "cost_preset": None,
    "inflation_rate": 0.02,
    "discount_rate": 0.03,
    "emissions_country": None,
    "pv_degradation_rate": 0.005,
    "calendar_model": "naumann_lam_field_calibrated",
    "dc_coupled": True,
    "inverter_efficiency": 0.96,
    "inverter_loading_ratio": 1.25,
    "start_date": "2023-01-01",
}


# --- Helpers ------------------------------------------------------------------


def _load_json(name: str) -> dict:
    path = os.path.join(_CONFIGS_DIR, name)
    with open(path) as f:
        return json.load(f)


def _remap_tmy_year(df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """Remap TMY DatetimeIndex to *target_year*."""
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) == 0:
        return df
    was_tz = idx.tz
    idx_utc = idx.tz_convert("UTC") if was_tz is not None else idx.tz_localize("UTC")
    dominant_year = idx_utc.year.value_counts().idxmax()
    offset = target_year - dominant_year
    if offset == 0:
        return df
    new_idx = idx_utc.map(lambda dt: dt.replace(year=dt.year + offset))
    new_idx = new_idx.tz_convert(was_tz) if was_tz is not None else new_idx.tz_localize(None)
    df = df.copy()
    df.index = new_idx
    return df


# --- App class ----------------------------------------------------------------


class App:
    """
    Single entry point for BREOS simulations.

    Parameters
    ----------
    config : dict
        Simulation configuration.  Required keys:

        - ``location`` — preset key (``"porto"``) **or** dict with
          ``latitude``, ``longitude``, ``timezone``.
        - ``n_modules`` — number of PV modules (int, > 0), unless
          ``pv_arrays`` is provided.
        - ``annual_consumption_kwh`` — yearly electricity demand (float, > 0).

        Optional keys (with defaults):

        - ``battery_kwh`` (0.0) — battery capacity; 0 = no battery.
        - ``pv_arrays`` (None) — list of arrays with ``modules``, ``module``,
          ``tilt``, and ``azimuth``. When set, BREOS calculates each
          array separately and combines production before the energy balance.
        - ``pv_module`` (None) — PV module name from the catalogue; None = first available.
        - ``load_profile`` ("6") — load profile type ("1"–"8").
        - ``tilt`` (None) — tilt angle in degrees; None = auto from latitude (fixed only).
        - ``azimuth`` (None) — surface azimuth; None = auto from latitude (fixed only).
        - ``tracking`` ("fixed") — ``"fixed"``, ``"single_axis"``, or ``"dual_axis"``.
        - ``axis_tilt`` (0.0) — single-axis rotation-axis tilt (degrees, 0 = HSAT).
        - ``axis_azimuth`` (None) — single-axis rotation-axis azimuth (None = auto from latitude).
        - ``max_angle`` (60.0) — single-axis maximum rotation from horizontal (±deg).
        - ``backtrack`` (True) — single-axis backtracking to avoid row-shading.
        - ``gcr`` (0.35) — single-axis ground coverage ratio.
        - ``cross_axis_tilt`` (0.0) — terrain slope perpendicular to rotation axis.
        - ``dual_axis_max_tilt`` (90.0) — dual-axis panel tilt cap (degrees).
        - ``resolution`` ("h") — time resolution ("h" or "15min").
        - ``projection_years`` (20) — economic projection horizon.
        - ``cost_preset`` (None) — key into configs/costs.json.
        - ``inflation_rate`` (0.02) — annual electricity inflation.
        - ``discount_rate`` (0.03) — discount rate for NPV.
        - ``emissions_country`` (None) — key into configs/emissions.json.
        - ``pv_degradation_rate`` (0.005) — annual PV degradation.
        - ``calendar_model`` ("naumann_lam_field_calibrated") — battery calendar model.
        - ``dc_coupled`` (True) — DC-coupled / hybrid inverter.
        - ``inverter_efficiency`` (0.96) — inverter efficiency.
        - ``inverter_loading_ratio`` (1.25) — DC/AC oversizing ratio.
    """

    def __init__(self, config: dict) -> None:
        cfg = {**_DEFAULTS, **config}
        self._validate(cfg)
        self._cfg = cfg

        # Resolve location
        loc = cfg["location"]
        if isinstance(loc, str):
            locations = _load_json("locations.json")
            if loc not in locations:
                available = ", ".join(sorted(locations))
                raise ValueError(f"Unknown location '{loc}'. Available: {available}")
            loc_data = locations[loc]
            self._lat = loc_data["latitude"]
            self._lon = loc_data["longitude"]
            self._tz = loc_data["timezone"]
            self._loc_key: Optional[str] = loc
        else:
            self._lat = loc["latitude"]
            self._lon = loc["longitude"]
            self._tz = loc["timezone"]
            self._loc_key = None

        # Resolve PV arrays, if provided. Multi-array systems are the preferred
        # representation for Designer rooftops because mixed faces cannot be
        # reduced to one tilt/azimuth pair without losing production fidelity.
        self._pv_arrays = self._normalise_pv_arrays(cfg.get("pv_arrays"))
        if self._pv_arrays:
            cfg["n_modules"] = sum(arr["modules"] for arr in self._pv_arrays)
            total_power_w = sum(arr["modules"] * get_module(arr["module"]).Mpp for arr in self._pv_arrays)
            self._avg_module_power_w = total_power_w / cfg["n_modules"]
            self._system_kwp = total_power_w / 1000
            module_name = self._pv_arrays[0]["module"]
        else:
            module_name = cfg["pv_module"]

        # Resolve PV module
        if module_name is None:
            module_name = next(iter(MODULES))
        self._pv_params = get_module(module_name)
        if not self._pv_arrays:
            self._avg_module_power_w = self._pv_params.Mpp
            self._system_kwp = cfg["n_modules"] * self._pv_params.Mpp / 1000

        # Resolve tilt / azimuth (used for fixed-tilt arrays; ignored for tracking)
        self._tilt = cfg["tilt"] if cfg["tilt"] is not None else estimate_optimal_tilt(self._lat)
        self._azimuth = cfg["azimuth"] if cfg["azimuth"] is not None else default_azimuth_fn(self._lat)

        # Resolve tracking
        tracking = cfg["tracking"]
        if tracking not in ("fixed", "single_axis", "dual_axis"):
            raise ValueError(
                f"tracking must be 'fixed', 'single_axis', or 'dual_axis', got {tracking!r}"
            )
        self._tracking = tracking
        # axis_azimuth defaults to hemisphere-appropriate orientation
        self._axis_azimuth = (
            cfg["axis_azimuth"] if cfg["axis_azimuth"] is not None else default_azimuth_fn(self._lat)
        )

        # Resolve cost preset
        self._cost_params = self._resolve_costs(cfg)

        # Resolve emissions
        self._emissions_params: Optional[EmissionsParams] = None
        if cfg["emissions_country"]:
            emissions_db = _load_json("emissions.json")
            key = cfg["emissions_country"]
            if key not in emissions_db:
                available = ", ".join(sorted(emissions_db))
                raise ValueError(f"Unknown emissions country '{key}'. Available: {available}")
            self._emissions_params = EmissionsParams(**emissions_db[key])

        self._result: Optional[Dict[str, Any]] = None

    # --- Public API -----------------------------------------------------------

    def simulate(self) -> None:
        """Run the full simulation pipeline."""
        cfg = self._cfg
        freq = cfg["resolution"]
        n_modules = cfg["n_modules"]
        battery_kwh = cfg["battery_kwh"]
        battery_wh = battery_kwh * 1000
        has_battery = battery_kwh > 0
        projection_years = cfg["projection_years"]
        degradation_rate = cfg["pv_degradation_rate"]
        hours_per_step = get_hours_per_step(freq)
        start_year = int(cfg["start_date"][:4])

        # 1. Weather
        weather = self._load_weather(freq, start_year)

        # 2. PV — 1-module DC production
        location = Location(self._lat, self._lon, tz=self._tz)
        if self._pv_arrays:
            dc_system_base = calculate_multi_array_production(
                weather_data=weather,
                location=location,
                arrays=self._pv_arrays,
                freq=freq,
            )
        else:
            if self._tracking == "fixed":
                dc_1mod = calculate_pv_production_dc(
                    weather_data=weather,
                    location=location,
                    tilt=self._tilt,
                    surface_azimuth=self._azimuth,
                    n_modules=1,
                    pv_params=self._pv_params,
                    freq=freq,
                )
            else:
                dc_1mod = calculate_pv_production_dc_tracking(
                    weather_data=weather,
                    location=location,
                    n_modules=1,
                    tracking=self._tracking,
                    axis_tilt=cfg["axis_tilt"],
                    axis_azimuth=self._axis_azimuth,
                    max_angle=cfg["max_angle"],
                    backtrack=cfg["backtrack"],
                    gcr=cfg["gcr"],
                    cross_axis_tilt=cfg["cross_axis_tilt"],
                    dual_axis_max_tilt=cfg["dual_axis_max_tilt"],
                    pv_params=self._pv_params,
                    freq=freq,
                )
            dc_system_base = dc_1mod * n_modules

        # 3. Load profile
        load_data = load_profile(
            profile_type=cfg["load_profile"],
            annual_consumption_kwh=cfg["annual_consumption_kwh"],
            start_date=cfg["start_date"],
            freq=freq,
            num_years=1,
            timezone="UTC",
        )

        # 4. Temperature series (indoor model for battery)
        ambient_temp = extract_ambient_temperature(weather)
        if ambient_temp is not None:
            temp_series = apply_indoor_temperature_model(ambient_temp)
        else:
            temp_series = pd.Series(25.0, index=dc_1mod.index)

        # 5. Multi-year propagation
        replacement_cost_per_kwh = self._cost_params.battery_cost_per_kwh
        replacement_cost = replacement_cost_per_kwh * battery_kwh

        cumulative_fec = 0.0
        cumulative_cal_seconds = 0.0
        cumulative_resistance_growth = 0.0
        cumulative_cycle_deg = 0.0
        cumulative_cal_deg = 0.0
        current_soh = 100.0
        total_replacements = 0
        total_replacement_cost = 0.0
        yearly_summaries = []
        first_year_results_df = None

        for year_idx in range(projection_years):
            pv_degradation_factor = (1 - degradation_rate) ** year_idx
            dc_power = dc_system_base * pv_degradation_factor

            if has_battery:
                batt_cfg = BatteryConfig(
                    nominal_energy_wh=battery_wh,
                    initial_soh=current_soh,
                    eol_percentage=0.70,
                    max_soc=0.90,
                    min_soc=0.10,
                    dc_coupled=cfg["dc_coupled"],
                    inverter_efficiency=cfg["inverter_efficiency"],
                    enable_replacement=True,
                    replacement_cost=replacement_cost,
                    calendar_model=cfg["calendar_model"],
                )
            else:
                batt_cfg = None

            results_df, total_pv, summary_df, year_rep_cost, year_n_rep, degradation_df = simulate_energy_balance(
                pv_dc=dc_power,
                houseload=load_data,
                battery_config=batt_cfg,
                freq=freq,
                temperature_series=temp_series if has_battery else None,
                initial_fec=cumulative_fec,
                initial_calendar_seconds=cumulative_cal_seconds,
                initial_resistance_growth=cumulative_resistance_growth,
                initial_cumulative_cycle_deg=cumulative_cycle_deg,
                initial_cumulative_cal_deg=cumulative_cal_deg,
            )

            if first_year_results_df is None:
                first_year_results_df = results_df

            # Update carryover state
            if has_battery and not degradation_df.empty:
                cumulative_fec = degradation_df["Cumulative_FEC"].iloc[-1]
                cumulative_cal_seconds = degradation_df["Cumulative_Calendar_Seconds"].iloc[-1]
                cumulative_cycle_deg = degradation_df["Cumulative_Cycle_Degradation"].iloc[-1]
                cumulative_cal_deg = degradation_df["Cumulative_Calendar_Degradation"].iloc[-1]
                current_soh = degradation_df["SOH"].iloc[-1]
                if "Resistance_Growth" in degradation_df.columns:
                    cumulative_resistance_growth = degradation_df["Resistance_Growth"].iloc[-1]

            total_replacements += year_n_rep
            total_replacement_cost += year_rep_cost

            # Yearly summary
            total_pv_kwh = total_pv / 1000
            total_load = (results_df["Houseload"].sum() / 1000) * hours_per_step
            total_import = (results_df["Import_From_Grid"].sum() / 1000) * hours_per_step
            total_export = (results_df["Sell_To_Grid"].sum() / 1000) * hours_per_step
            grid_indep = (1 - total_import / total_load) * 100 if total_load > 0 else 0

            yearly_summaries.append(
                {
                    "Year": year_idx + 1,
                    "PV_Production_kWh": total_pv_kwh,
                    "Load_kWh": total_load,
                    "Import_kWh": total_import,
                    "Export_kWh": total_export,
                    "Grid_Independence_%": grid_indep,
                    "Battery_SOH_%": current_soh if has_battery else None,
                    "Replacements": year_n_rep,
                    "Replacement_Cost": year_rep_cost,
                    "PV_Degradation_Factor": pv_degradation_factor,
                }
            )

        yearly_df = pd.DataFrame(yearly_summaries)

        # 6. Economics
        costs_dict = self._build_costs_dict()

        cost_proj = cost_analysis_projection(
            results_df=first_year_results_df,
            costs=costs_dict,
            num_years=projection_years,
            inflation_rate=cfg["inflation_rate"],
            discount_rate=cfg["discount_rate"],
            freq=freq,
            yearly_summary_df=yearly_df,
            total_replacement_cost=total_replacement_cost,
            emissions_params=self._emissions_params,
        )

        payback = find_payback_year(cost_proj)

        # 7. Derived metrics
        year1 = yearly_df.iloc[0]
        yr1_pv = year1["PV_Production_kWh"]
        yr1_export = year1["Export_kWh"]
        yr1_import = year1["Import_kWh"]
        yr1_load = year1["Load_kWh"]
        self_consumption_kwh = yr1_pv - yr1_export
        self_consumption_pct = (self_consumption_kwh / yr1_pv * 100) if yr1_pv > 0 else 0.0
        grid_indep_y1 = year1["Grid_Independence_%"]

        total_initial = costs_dict["total_initial_cost"]
        npv_savings = float(cost_proj["Savings_Cumulative_NPV"].iloc[-1])
        system_kwp = self._system_kwp

        lcoe = calculate_lcoe(
            total_investment=total_initial,
            annual_production_kwh=yr1_pv,
            annual_operation_cost=costs_dict["annual_operation_cost"],
            lifetime_years=projection_years,
            discount_rate=cfg["discount_rate"],
            degradation_rate=degradation_rate,
        )

        # 8. Build result dict
        result: Dict[str, Any] = {
            # System
            "n_modules": n_modules,
            "pv_kwp": round(system_kwp, 3),
            "battery_kwh": battery_kwh,
            # Year 1 energy
            "pv_production_kwh": round(float(yr1_pv), 2),
            "consumption_kwh": round(float(yr1_load), 2),
            "self_consumption_kwh": round(float(self_consumption_kwh), 2),
            "grid_import_kwh": round(float(yr1_import), 2),
            "grid_export_kwh": round(float(yr1_export), 2),
            "grid_independence_pct": round(float(grid_indep_y1), 2),
            "self_consumption_pct": round(float(self_consumption_pct), 2),
            # Economics
            "total_investment_eur": round(float(total_initial), 2),
            "payback_year": int(payback) if payback is not None else None,
            "npv_savings_eur": round(float(npv_savings), 2),
            "lcoe_eur_kwh": round(float(lcoe), 4),
            # Yearly breakdown
            "yearly": self._yearly_to_dicts(yearly_df),
            "monthly": self._monthly_to_dicts(first_year_results_df, freq),
            "financial": self._financial_to_dicts(cost_proj, total_initial),
        }

        if self._pv_arrays:
            result["pv_arrays"] = [
                {
                    "modules": arr["modules"],
                    "module": arr["module"],
                    "tilt": arr["tilt"],
                    "azimuth": arr["azimuth"],
                }
                for arr in self._pv_arrays
            ]

        # Battery (only if present)
        if has_battery:
            result["battery_soh_end_pct"] = round(float(current_soh), 2)
            result["battery_replacements"] = total_replacements
            result["battery_replacement_cost_eur"] = round(float(total_replacement_cost), 2)

        # Emissions
        if self._emissions_params is not None:
            co2 = calculate_co2_savings(yr1_pv, self_consumption_kwh, self._emissions_params)
            lifetime_co2 = float(cost_proj["CO2_Avoided_Total_Cumulative_kg"].iloc[-1])
            result["co2_avoided_year1_kg"] = round(co2["CO2_Avoided_Total_kg"], 2)
            result["co2_avoided_total_kg"] = round(lifetime_co2, 2)

        self._result = result

    def result(self) -> Dict[str, Any]:
        """
        Return simulation results as a plain dict.

        Raises ``RuntimeError`` if :meth:`simulate` has not been called.
        """
        if self._result is None:
            raise RuntimeError("Call simulate() before result().")
        return self._result

    # --- Internal helpers -----------------------------------------------------

    @staticmethod
    def _validate(cfg: dict) -> None:
        # Required keys
        for key in ("location", "annual_consumption_kwh"):
            if key not in cfg:
                raise ValueError(f"Missing required config key: '{key}'")

        has_arrays = bool(cfg.get("pv_arrays"))
        if not has_arrays and "n_modules" not in cfg:
            raise ValueError("Missing required config key: 'n_modules'")

        loc = cfg["location"]
        if isinstance(loc, dict):
            for field in ("latitude", "longitude", "timezone"):
                if field not in loc:
                    raise ValueError(f"Custom location must include '{field}'")
        elif not isinstance(loc, str):
            raise TypeError("'location' must be a string key or a dict with latitude/longitude/timezone")

        if not has_arrays and cfg["n_modules"] < 1:
            raise ValueError("'n_modules' must be >= 1")
        if has_arrays:
            if not isinstance(cfg["pv_arrays"], list):
                raise TypeError("'pv_arrays' must be a list")
            for i, arr in enumerate(cfg["pv_arrays"]):
                if not isinstance(arr, dict):
                    raise TypeError(f"'pv_arrays[{i}]' must be a dict")
                modules = arr.get("modules", 0)
                if modules < 1:
                    raise ValueError(f"'pv_arrays[{i}].modules' must be >= 1")
                tilt = arr.get("tilt", cfg.get("tilt"))
                azimuth = arr.get("azimuth", cfg.get("azimuth"))
                if tilt is not None and not 0 <= tilt <= 90:
                    raise ValueError(f"'pv_arrays[{i}].tilt' must be between 0 and 90")
                if azimuth is not None and not 0 <= azimuth <= 360:
                    raise ValueError(f"'pv_arrays[{i}].azimuth' must be between 0 and 360")
        if cfg["annual_consumption_kwh"] <= 0:
            raise ValueError("'annual_consumption_kwh' must be > 0")
        if cfg["resolution"] not in ("h", "15min"):
            raise ValueError("'resolution' must be 'h' or '15min'")

    def _normalise_pv_arrays(self, arrays: Optional[list]) -> list[dict]:
        if not arrays:
            return []

        default_module = self._cfg.get("pv_module") or next(iter(MODULES))
        default_tilt = self._cfg.get("tilt") if self._cfg.get("tilt") is not None else estimate_optimal_tilt(self._lat)
        default_azimuth = (
            self._cfg.get("azimuth") if self._cfg.get("azimuth") is not None else default_azimuth_fn(self._lat)
        )

        normalized = []
        for arr in arrays:
            modules = int(arr["modules"])
            module = arr.get("module") or default_module
            tilt = arr.get("tilt", default_tilt)
            azimuth = arr.get("azimuth", default_azimuth)
            normalized.append(
                {
                    "modules": modules,
                    "module": module,
                    "tilt": float(tilt),
                    "azimuth": float(azimuth),
                }
            )
        return normalized

    def _load_weather(self, freq: str, start_year: int) -> pd.DataFrame:
        """Load TMY weather, falling back to PVGIS fetch."""
        weather = None
        weather_dir = os.path.join(os.path.dirname(_CONFIGS_DIR), "weather")

        # Try local files first (faster) for preset locations
        if self._loc_key and os.path.isdir(weather_dir):
            weather = load_weather(location=self._loc_key, data_type="tmy", weather_dir=weather_dir)

        if weather is None:
            weather, _ = fetch_tmy_weather_data(
                latitude=self._lat,
                longitude=self._lon,
                sample_year=start_year,
                freq="h",
            )

        if weather.index.tz is None:
            weather.index = weather.index.tz_localize("UTC")
        weather = _remap_tmy_year(weather, start_year)

        # Resample to 15min if needed
        if freq == "15min":
            inferred = pd.infer_freq(weather.index[:10])
            if inferred and "h" in inferred.lower() and "15" not in inferred:
                weather = resample_to_15min(weather, latitude=self._lat, longitude=self._lon)

        return weather

    def _resolve_costs(self, cfg: dict) -> CostParams:
        """Build CostParams from preset + overrides + financials."""
        params: Dict[str, Any] = {}

        # Load preset if provided
        if cfg.get("cost_preset"):
            costs_db = _load_json("costs.json")
            preset_key = cfg["cost_preset"]
            if preset_key not in costs_db:
                available = ", ".join(sorted(costs_db))
                raise ValueError(f"Unknown cost preset '{preset_key}'. Available: {available}")
            preset = costs_db[preset_key]

            params["electricity_cost"] = preset.get("electricity_cost", 0.27)
            params["electricity_sold_cost"] = preset.get("electricity_sold_cost", 0.06)
            params["daily_power_cost"] = preset.get("daily_power_cost", 0.30)
            params["module_cost_per_w"] = preset.get("module_cost_per_w", 0.125)
            params["battery_cost_per_kwh"] = preset.get("storage_cost_per_kwh", BATTERY_REPLACEMENT_COST_PER_KWH)
            params["inverter_cost_per_kw"] = preset.get("inverter_cost_per_kw_hybrid", 102.58)
            params["inverter_cost_per_kw_nobatt"] = preset.get("inverter_cost_per_kw_simple", 48.37)
            params["installation_cost_per_module"] = preset.get("installation_cost_per_module", 350.0)
            params["battery_installation_cost"] = preset.get("installation_cost_battery", 350.0)
            params["maintenance_cost_per_panel"] = preset.get("maintenance_cost_per_panel", 0.0)
            params["maintenance_cost_fixed"] = preset.get("maintenance_cost", 50.0)
            params["other_cost_per_module"] = preset.get("other_cost_per_module", 0.0)
            params["other_cost_fixed"] = preset.get("other_costs", 50.0)

        # Financial defaults
        params["dc_ac_ratio"] = cfg["inverter_loading_ratio"]
        params.setdefault("inflation_rate", cfg["inflation_rate"])
        params.setdefault("discount_rate", cfg["discount_rate"])
        params["pv_degradation_rate"] = cfg["pv_degradation_rate"]

        return CostParams(**params)

    def _build_costs_dict(self) -> Dict[str, float]:
        """Build the costs dict for cost_analysis_projection."""
        cfg = self._cfg
        n_modules = cfg["n_modules"]
        battery_kwh = cfg["battery_kwh"]

        return calculate_costs(
            n_modules=n_modules,
            module_power_w=self._avg_module_power_w,
            battery_capacity_wh=battery_kwh * 1000,
            cost_params=self._cost_params,
        )

    @staticmethod
    def _monthly_to_dicts(results_df: pd.DataFrame, freq: str) -> list:
        """Convert first-year timestep results into monthly energy rows."""
        hours_per_step = get_hours_per_step(freq)
        df = results_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Datetime" in df.columns:
                df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
                df.set_index("Datetime", inplace=True)
            else:
                raise ValueError("results_df must have a DatetimeIndex or Datetime column")

        monthly = df[["PV_Production", "Houseload", "Import_From_Grid", "Sell_To_Grid"]].resample("ME").sum()
        monthly = monthly * hours_per_step / 1000

        rows = []
        for idx, row in monthly.iterrows():
            pv = float(row["PV_Production"])
            consumption = float(row["Houseload"])
            export = float(row["Sell_To_Grid"])
            imported = float(row["Import_From_Grid"])
            self_consumption = pv - export
            rows.append(
                {
                    "month": idx.strftime("%b"),
                    "pv_kwh": round(pv, 2),
                    "consumption_kwh": round(consumption, 2),
                    "self_consumption_kwh": round(self_consumption, 2),
                    "import_kwh": round(imported, 2),
                    "export_kwh": round(export, 2),
                    "grid_independence_pct": round((1 - imported / consumption) * 100, 2) if consumption > 0 else 0.0,
                }
            )
        return rows

    @staticmethod
    def _financial_to_dicts(cost_proj: pd.DataFrame, total_initial_cost: float) -> list:
        """Convert BREOS cost projection into the dashboard line-chart shape."""
        rows = [{"year": 0, "balance": round(-float(total_initial_cost), 2), "reference": 0.0}]
        for _, row in cost_proj.iterrows():
            rows.append(
                {
                    "year": int(row["Year"]),
                    "balance": round(float(row["Savings_Cumulative_NPV"]), 2),
                    "reference": 0.0,
                    "cost_with_system": round(float(row["Cost_System_Cumulative_NPV"]), 2),
                    "cost_without_system": round(float(row["Cost_No_Sys_Cumulative_NPV"]), 2),
                }
            )
        return rows

    @staticmethod
    def _yearly_to_dicts(yearly_df: pd.DataFrame) -> list:
        """Convert yearly summary DataFrame to a list of plain dicts."""
        rows = []
        for _, row in yearly_df.iterrows():
            d: Dict[str, Any] = {
                "year": int(row["Year"]),
                "pv_kwh": round(float(row["PV_Production_kWh"]), 2),
                "consumption_kwh": round(float(row["Load_kWh"]), 2),
                "self_consumption_kwh": round(float(row["PV_Production_kWh"] - row["Export_kWh"]), 2),
                "import_kwh": round(float(row["Import_kWh"]), 2),
                "export_kwh": round(float(row["Export_kWh"]), 2),
                "grid_independence_pct": round(float(row["Grid_Independence_%"]), 2),
            }
            if row["Battery_SOH_%"] is not None:
                d["soh_pct"] = round(float(row["Battery_SOH_%"]), 2)
            rows.append(d)
        return rows
