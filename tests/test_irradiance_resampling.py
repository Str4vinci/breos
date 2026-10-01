"""Hourly irradiance policy, dawn/dusk support and public runner wiring."""

import numpy as np
import pandas as pd
import pytest
from pvlib.location import Location

from breos import App
from breos.app_config import resolve_app_config
from breos.app_inputs import resample_hourly_weather
from breos.cli import build_parser
from breos.weather import WEATHER_METADATA_KEY, resample_to_15min

COORDINATES = {"latitude": 41.1579, "longitude": -8.6291, "altitude": 0.0}


def _weather(basis="interval_mean"):
    index = pd.date_range("2025-06-21 11:00", periods=5, freq="h", tz="UTC")
    frame = pd.DataFrame({"ghi": 500.0, "dni": 50.0, "dhi": 500.0}, index=index)
    if basis is not None:
        frame.attrs[WEATHER_METADATA_KEY] = {"radiation_time_basis": basis, "timestamp_label_basis": "left"}
    return frame


@pytest.mark.parametrize("policy", ["clear_sky", "clear_sky_energy_conserving"])
def test_overcast_diffuse_is_not_capped(policy):
    source = _weather()
    sky = Location(**COORDINATES).get_clearsky(source.index + pd.Timedelta(minutes=30))
    assert (source.dhi.to_numpy() > 1.5 * sky.dhi.to_numpy()).all()
    result = resample_to_15min(source, irradiance_resampling=policy, **COORDINATES)
    means = result.dhi.to_numpy().reshape(-1, 4).mean(axis=1)
    np.testing.assert_allclose(means, source.dhi, rtol=0.04 if policy == "clear_sky" else 1e-9)
    assert (result.dhi > result.ghi).any() or np.allclose(result.dhi, result.ghi)


@pytest.mark.parametrize(
    "names", [("ghi", "dni", "dhi"), ("shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation")]
)
def test_independent_component_hourly_conservation(names):
    source = _weather().rename(columns=dict(zip(("ghi", "dni", "dhi"), names, strict=True)))
    source.iloc[:] = np.array([[500, 400, 300], [700, 600, 350], [600, 500, 450], [300, 200, 250], [400, 100, 700]])
    result = resample_to_15min(source, **COORDINATES)
    np.testing.assert_allclose(result.to_numpy().reshape(-1, 4, 3).mean(axis=1), source, rtol=1e-9, atol=1e-6)
    metadata = result.attrs[WEATHER_METADATA_KEY]
    assert metadata["irradiance_resampling"] == "auto"
    assert metadata["irradiance_resampling_resolved"] == "clear_sky_energy_conserving"
    assert "preserve_irradiance_energy" not in metadata
    assert metadata["irradiance_closure"]["after"]["absolute_residual_energy_fraction"] > 0


@pytest.mark.parametrize(
    "basis,expected", [("interval_mean", "clear_sky_energy_conserving"), ("instant", "clear_sky"), (None, "clear_sky")]
)
def test_auto_resolves_from_declared_basis(basis, expected):
    result = resample_to_15min(_weather(basis), **COORDINATES)
    assert result.attrs[WEATHER_METADATA_KEY]["irradiance_resampling_resolved"] == expected


@pytest.mark.parametrize("basis", ["instant", None])
def test_conservation_rejects_non_mean_input(basis):
    with pytest.raises(ValueError, match=f"irradiance_resampling.*{basis or 'undeclared'}"):
        resample_to_15min(_weather(basis), irradiance_resampling="clear_sky_energy_conserving", **COORDINATES)


def test_dawn_dusk_linear_fallback_and_night(monkeypatch):
    import breos.weather as weather_module

    index = pd.date_range("2025-06-21", periods=7, freq="h", tz="UTC")
    cs = np.array([0.0, 4.0, 100.0, 100.0, 100.0, 4.0, 0.0])
    source = pd.DataFrame(
        {"ghi": [0, 50, 90, 40, 80, 30, 0], "dni": [0, 50, 90, 40, 80, 30, 0], "dhi": [0, 50, 90, 40, 80, 30, 0]},
        index=index,
    )

    class FakeLocation:
        def __init__(self, *args, **kwargs):
            pass

        def get_clearsky(self, times, solar_position=None):
            values = np.interp((times - index[0]) / pd.Timedelta(hours=1), np.arange(len(index)), cs)
            return pd.DataFrame(dict.fromkeys(("ghi", "dni", "dhi"), values), index=times)

        def get_solarposition(self, times):
            return pd.DataFrame({"apparent_zenith": 60.0}, index=times)

    monkeypatch.setattr(weather_module, "Location", FakeLocation)
    result = resample_to_15min(source, **COORDINATES)
    for component in source:
        for hour in [0, 1, 4, 5]:
            block = result[component].iloc[hour * 4 : (hour + 1) * 4]
            assert block.max() <= source[component].iloc[hour : hour + 2].max()
        assert result[component].iloc[-4:].eq(0).all()
        assert result[component].ge(0).all()
    assert result.attrs[WEATHER_METADATA_KEY]["irradiance_resampling_fallback_counts"] == dict.fromkeys(
        ("ghi", "dni", "dhi"), 15
    )


def test_zero_support_positive_hours_are_counted():
    source = _weather()
    source = source.iloc[:3].copy()
    source.index = pd.date_range("2025-06-21 00:00", periods=3, freq="h", tz="UTC")
    # Replace the midday rows with night rows carrying anomalous positive input.
    result = resample_to_15min(source, **COORDINATES)
    np.testing.assert_allclose(result.to_numpy().reshape(-1, 4, 3).mean(axis=1), source, rtol=1e-9, atol=1e-6)
    assert result.attrs[WEATHER_METADATA_KEY]["irradiance_resampling_zero_support_counts"] == dict.fromkeys(
        ("ghi", "dni", "dhi"), 3
    )


def test_negative_direct_interpolation_is_clipped():
    source = _weather(None)
    source.iloc[:] = -1
    assert resample_to_15min(source).ge(0).all().all()


@pytest.mark.parametrize("value", ["bogus", True, None])
def test_app_validates_policy(value):
    with pytest.raises(ValueError, match="irradiance_resampling.*auto, clear_sky, clear_sky_energy_conserving"):
        resolve_app_config(
            {"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000, "irradiance_resampling": value}
        )


def test_app_passes_and_records_explicit_policy(monkeypatch):
    from tools.generate_app_golden import synthetic_weather

    source = synthetic_weather()
    source.attrs[WEATHER_METADATA_KEY] = {"radiation_time_basis": "interval_mean", "timestamp_label_basis": "left"}
    monkeypatch.setattr("breos.app.load_weather", lambda *args, **kwargs: source)
    received = []

    def resample(frame, **kwargs):
        received.append(kwargs["irradiance_resampling"])
        return resample_to_15min(frame, **kwargs, altitude=0)

    monkeypatch.setattr("breos.app.resample_to_15min", resample)
    app = App(
        {
            "location": "porto",
            "n_modules": 10,
            "annual_consumption_kwh": 4000,
            "start_date": "2025-01-01",
            "resolution": "15min",
            "projection_years": 1,
            "irradiance_resampling": "clear_sky_energy_conserving",
        }
    )
    app.simulate()
    provenance = app.result()["provenance"]
    assert received == ["clear_sky_energy_conserving"]
    assert provenance["weather"]["irradiance_resampling"] == received[0]
    assert provenance["weather"]["irradiance_resampling_resolved"] == received[0]
    assert "preserve_irradiance_energy" not in provenance["weather"]
    assert provenance["resolved_config"]["irradiance_resampling"] == received[0]


@pytest.mark.parametrize("command", ["run", "montecarlo"])
def test_cli_exposes_shared_app_policy(command):
    args = build_parser().parse_args([command, "--config", "config.toml", "--irradiance-resampling", "clear_sky"])
    assert args.irradiance_resampling == "clear_sky"


def test_helper_keeps_native_quarter_hour_weather():
    source = resample_to_15min(_weather(), **COORDINATES)
    assert (
        resample_hourly_weather(
            source,
            "15min",
            latitude=41,
            longitude=-8,
            resample=lambda *args, **kwargs: pytest.fail("already 15-minute"),
        )
        is source
    )


@pytest.mark.parametrize("runner", ["fixed", "search", "weather_sequence"])
def test_optimizer_prepares_hourly_weather_before_pv(monkeypatch, runner):
    import breos.optimization as optimization
    from tests.test_optimization_parity import _problem_config

    source = _weather()
    config = _problem_config()
    config["simulation"].update(resolution="15min", irradiance_resampling="clear_sky_energy_conserving")
    load = pd.DataFrame({"Load": 100.0}, index=source.index)
    expected = resample_to_15min(source, latitude=41.15, longitude=-8.61)
    calls = []

    class StopAtPV(RuntimeError):
        pass

    def spy(**kwargs):
        calls.append(kwargs["weather_data"])
        raise StopAtPV

    monkeypatch.setattr(optimization, "calculate_pv_production_dc", spy)
    with pytest.raises(StopAtPV):
        if runner == "search":
            pytest.importorskip("pymoo")
            problem = optimization.SolarDesignProblem(source, load, config)
            problem._evaluate(np.array([2, 0, 30, 180]), {})
        else:
            optimization.evaluate_projected_design(
                tmy_data=source,
                houseload=load,
                config=config,
                n_modules=2,
                battery_kwh=0,
                tilt=30,
                azimuth=180,
                weather_by_year=[source] if runner == "weather_sequence" else None,
            )
    pd.testing.assert_frame_equal(calls[0], expected)
    assert calls[0].attrs[WEATHER_METADATA_KEY]["irradiance_resampling"] == "clear_sky_energy_conserving"


def test_closure_diagnostics_are_observational():
    source = _weather()
    source.iloc[:] = np.array([[500, 400, 300], [700, 600, 350], [600, 500, 450], [300, 200, 250], [400, 100, 700]])
    unconserved = resample_to_15min(source, irradiance_resampling="clear_sky", **COORDINATES)
    conserved = resample_to_15min(source, **COORDINATES)
    diagnostics = conserved.attrs[WEATHER_METADATA_KEY]["irradiance_closure"]
    assert diagnostics["before"] == unconserved.attrs[WEATHER_METADATA_KEY]["irradiance_closure"]["after"]
    site = Location(**COORDINATES)
    position = site.get_solarposition(conserved.index + pd.Timedelta(minutes=7.5))
    cos_zenith = np.maximum(np.cos(np.deg2rad(position.apparent_zenith.to_numpy())), 0)
    residual = np.abs(conserved.ghi - conserved.dni * cos_zenith - conserved.dhi)
    assert diagnostics["after"]["absolute_residual_energy_fraction"] == pytest.approx(
        residual.sum() / conserved.ghi.sum()
    )
    assert diagnostics["after"]["ghi_weighted_mean_absolute_residual_w_m2"] == pytest.approx(
        (residual * conserved.ghi).sum() / conserved.ghi.sum()
    )
    assert conserved.dhi.iloc[-4:].mean() > conserved.ghi.iloc[-4:].mean()


def test_montecarlo_records_the_app_policy(tmp_path):
    from breos.montecarlo import MonteCarloSettings, run_montecarlo
    from breos.weather import save_weather_csv
    from tools.generate_app_golden import synthetic_weather

    source = synthetic_weather()
    source.attrs[WEATHER_METADATA_KEY] = {"radiation_time_basis": "interval_mean", "timestamp_label_basis": "left"}
    source.index.name = "date"
    path = tmp_path / "weather.csv"
    save_weather_csv(source, path)
    result = run_montecarlo(
        {
            "location": "porto",
            "n_modules": 10,
            "annual_consumption_kwh": 4000,
            "resolution": "15min",
            "irradiance_resampling": "clear_sky_energy_conserving",
        },
        MonteCarloSettings(weather_file=str(path), n_runs=1, years_per_run=1, target_year=2025, seed=0),
    )
    provenance = result.provenance
    metadata = provenance["runtime_weather"]["metadata"]
    assert provenance["resolved_config"]["irradiance_resampling"] == "clear_sky_energy_conserving"
    assert metadata["irradiance_resampling"] == "clear_sky_energy_conserving"
    assert metadata["irradiance_resampling_resolved"] == "clear_sky_energy_conserving"
    assert "preserve_irradiance_energy" not in metadata
    assert "preserve_irradiance_energy" not in provenance["settings"]
    assert not hasattr(result.settings, "preserve_irradiance_energy")


def test_optimizer_records_policy_and_resamples_at_the_configured_altitude(monkeypatch):
    import breos.optimization as optimization
    from tests.test_optimization_parity import _problem_config
    from tools.generate_app_golden import synthetic_weather

    source = synthetic_weather()
    source.attrs[WEATHER_METADATA_KEY] = {"radiation_time_basis": "instant", "irradiance_time_offset_hours": 0.0}
    config = _problem_config()
    config["location"]["altitude"] = 120.0
    config["simulation"]["resolution"] = "15min"
    load = pd.DataFrame({"Load": 400.0}, index=pd.date_range(source.index[0], periods=4 * len(source), freq="15min"))
    altitudes = []

    def resample(frame, **kwargs):
        altitudes.append(kwargs["altitude"])
        return resample_to_15min(frame, **kwargs)

    monkeypatch.setattr(optimization, "resample_to_15min", resample)
    result = optimization.evaluate_projected_design(
        source, load, config, n_modules=2, battery_kwh=0, tilt=30, azimuth=180, weather_by_year=[source]
    )
    assert altitudes == [120.0, 120.0]
    provenance = result.provenance
    assert provenance["simulation"]["irradiance_resampling"] == "auto"
    for record in (provenance["weather"], *provenance["weather_by_year"]):
        assert record["irradiance_resampling"] == "auto"
        assert record["irradiance_resampling_resolved"] == "clear_sky"
        assert record["output_resolution"] == "15min"
