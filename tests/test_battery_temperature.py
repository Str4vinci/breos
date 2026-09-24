"""Battery temperature input: unusable input is an error, never 25 °C (#153)."""

import numpy as np
import pandas as pd
import pytest

from breos.battery import align_simulation_inputs
from breos.weather import build_battery_temperature_series

NO_INDOOR = {"enabled": False}


def _hourly(year: int, periods: int = 48) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", periods=periods, freq="h", tz="UTC")


def _write_csv(path, stamps, values, columns=("date", "temp")):
    pd.DataFrame({columns[0]: stamps.tz_localize(None), columns[1]: values}).to_csv(path, index=False)
    return str(path)


def test_missing_temperature_file_raises():
    with pytest.raises(FileNotFoundError, match="battery temperature file not found"):
        build_battery_temperature_series("no/such/temperature.csv", index=_hourly(2025))


def test_temperature_file_without_recognised_columns_raises(tmp_path):
    path = _write_csv(tmp_path / "t.csv", _hourly(2025), np.full(48, 5.0), columns=("stamp", "celsius"))

    with pytest.raises(ValueError, match="needs a timestamp column"):
        build_battery_temperature_series(path, index=_hourly(2025))


@pytest.mark.filterwarnings("ignore:Could not infer format")
def test_temperature_file_with_unreadable_timestamps_raises(tmp_path):
    path = tmp_path / "t.csv"
    pd.DataFrame({"date": ["not a date", "nor this"], "temp": [5.0, 6.0]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="unreadable timestamps"):
        build_battery_temperature_series(str(path), index=_hourly(2025, 2))


def test_temperature_file_from_another_year_raises_instead_of_using_25c(tmp_path):
    path = _write_csv(tmp_path / "t.csv", _hourly(2023), np.full(48, 5.0))

    with pytest.raises(ValueError, match="from 2023 and the simulation runs in 2025; restamp it to 2025"):
        build_battery_temperature_series(path, index=_hourly(2025), indoor_model=NO_INDOOR)


def test_temperature_file_in_the_simulation_year_is_used(tmp_path):
    values = np.linspace(0.0, 10.0, 48)
    path = _write_csv(tmp_path / "t.csv", _hourly(2025), values)

    temps = build_battery_temperature_series(path, index=_hourly(2025), indoor_model=NO_INDOOR)

    np.testing.assert_allclose(temps.to_numpy(), values)


def test_hourly_temperatures_hold_across_a_15min_index_but_do_not_bridge_gaps(tmp_path):
    hourly = _hourly(2025, 4)
    path = _write_csv(tmp_path / "t.csv", hourly, [1.0, 2.0, 3.0, 4.0])
    quarter = pd.date_range(hourly[0], hourly[-1] + pd.Timedelta("45min"), freq="15min")

    temps = build_battery_temperature_series(path, index=quarter, indoor_model=NO_INDOOR)
    np.testing.assert_allclose(temps.to_numpy(), np.repeat([1.0, 2.0, 3.0, 4.0], 4))

    day = _hourly(2025, 8)
    gapped = _write_csv(tmp_path / "gap.csv", day.delete([2, 3]), np.full(6, 5.0))
    day_quarter = pd.date_range(day[0], day[-1] + pd.Timedelta("45min"), freq="15min")
    with pytest.raises(ValueError, match="does not cover 8 of 32 simulation steps"):
        build_battery_temperature_series(gapped, index=day_quarter, indoor_model=NO_INDOOR)


def test_weather_temperatures_from_another_year_raise_unless_aligned():
    weather = pd.DataFrame({"temp_air": np.linspace(0.0, 5.0, 48)}, index=_hourly(2023))

    with pytest.raises(ValueError, match="weather temperature does not cover 48 of 48"):
        build_battery_temperature_series("weather", index=_hourly(2025), weather_df=weather)

    temps = build_battery_temperature_series(
        "weather", index=_hourly(2025), weather_df=weather, indoor_model=NO_INDOOR, align_weather_year=True
    )
    np.testing.assert_allclose(temps.to_numpy(), weather["temp_air"].to_numpy())


def test_weather_temperatures_with_nan_raise():
    weather = pd.DataFrame({"temp_air": [5.0, np.nan, 6.0]}, index=_hourly(2025, 3))

    with pytest.raises(ValueError, match="1 readings that are not finite"):
        build_battery_temperature_series("weather", index=_hourly(2025, 3), weather_df=weather)


@pytest.mark.parametrize("config", [True, ["weather"], {"path": "t.csv"}])
def test_unsupported_temperature_config_raises(config):
    with pytest.raises(TypeError, match="battery temperature must be"):
        build_battery_temperature_series(config, index=_hourly(2025))


def test_non_finite_fixed_temperature_raises():
    with pytest.raises(ValueError, match="must be finite"):
        build_battery_temperature_series(float("nan"), index=_hourly(2025))


def test_simulation_boundary_rejects_a_temperature_series_with_gaps():
    index = _hourly(2025, 24)
    pv = pd.Series(1000.0, index=index)
    load = pd.DataFrame({"Load": np.full(24, 500.0)}, index=index)
    temperature = pd.Series(5.0, index=index[:12])

    with pytest.raises(ValueError, match="temperature_series has no finite value for 12 of 24"):
        align_simulation_inputs(pv, load, temperature)

    aligned = align_simulation_inputs(pv, load)
    assert set(aligned.temperature_c) == {25.0}


def test_projected_weather_sequence_keeps_the_representative_temperatures(monkeypatch):
    """A weather sequence from another year used to leave the battery at 25 °C."""
    from breos.optimization import evaluate_projected_design

    tmy_index = _hourly(2023, 24)
    sequence_index = _hourly(2025, 24)
    tmy = pd.DataFrame({"temp_air": np.linspace(2.0, 8.0, 24)}, index=tmy_index)
    sequence = pd.DataFrame({"temp_air": np.full(24, 30.0)}, index=sequence_index)
    load = pd.DataFrame({"Load": np.full(24, 500.0)}, index=sequence_index)
    seen = {}

    monkeypatch.setattr(
        "breos.optimization.calculate_pv_production_dc",
        lambda **kwargs: pd.Series(1000.0, index=kwargs["weather_data"].index),
    )

    def _capture(**kwargs):
        seen["temperature"] = kwargs["temperature_series"]
        return {"_yearly_summary_df": pd.DataFrame(), "_cost_projection_df": pd.DataFrame()}

    monkeypatch.setattr("breos.optimization._evaluate_projected_design_metrics", _capture)
    config = {
        "location": {"latitude": 41.15, "longitude": -8.63},
        "pv": {"module": "Suntech_STP550S_STC"},
        "battery": {"indoor_model": NO_INDOOR},
        "simulation": {"resolution": "h", "years_projection": 2},
        "financials": {"project_lifespan": 2},
        "costs": {"dc_ac_ratio": 1.25},
    }

    evaluate_projected_design(
        tmy, load, config, n_modules=9, battery_kwh=5.0, tilt=35.0, azimuth=200.0, weather_by_year=[sequence] * 2
    )

    temperature = seen["temperature"]
    pd.testing.assert_index_equal(temperature.index, sequence_index)
    np.testing.assert_allclose(temperature.to_numpy(), tmy["temp_air"].to_numpy())
