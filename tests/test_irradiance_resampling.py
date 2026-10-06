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
    with pytest.raises(
        ValueError, match="irradiance_resampling.*auto, clear_sky, clear_sky_energy_conserving, quadrature"
    ):
        resolve_app_config(
            {"location": "porto", "n_modules": 10, "annual_consumption_kwh": 4000, "irradiance_resampling": value}
        )


@pytest.mark.parametrize("policy", ["clear_sky_energy_conserving", "quadrature"])
def test_app_passes_and_records_explicit_policy(monkeypatch, policy):
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
            "irradiance_resampling": policy,
        }
    )
    app.simulate()
    provenance = app.result()["provenance"]
    assert received == [policy]
    assert provenance["weather"]["irradiance_resampling"] == received[0]
    assert provenance["weather"]["irradiance_resampling_resolved"] == received[0]
    assert "preserve_irradiance_energy" not in provenance["weather"]
    assert provenance["resolved_config"]["irradiance_resampling"] == received[0]


@pytest.mark.parametrize("command", ["run", "montecarlo"])
@pytest.mark.parametrize("policy", ["clear_sky", "quadrature"])
def test_cli_exposes_shared_app_policy(command, policy):
    args = build_parser().parse_args([command, "--config", "config.toml", "--irradiance-resampling", policy])
    assert args.irradiance_resampling == policy


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


@pytest.mark.parametrize("policy", ["clear_sky_energy_conserving", "quadrature"])
@pytest.mark.parametrize("runner", ["fixed", "search", "weather_sequence"])
def test_optimizer_prepares_hourly_weather_before_pv(monkeypatch, runner, policy):
    import breos.optimization as optimization
    from tests.test_optimization_parity import _problem_config

    source = _weather()
    config = _problem_config()
    config["simulation"].update(resolution="15min", irradiance_resampling=policy)
    load = pd.DataFrame({"Load": 100.0}, index=source.index)
    expected = resample_to_15min(source, latitude=41.15, longitude=-8.61, irradiance_resampling=policy)
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
    assert calls[0].attrs[WEATHER_METADATA_KEY]["irradiance_resampling"] == policy


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


@pytest.mark.parametrize("policy", ["clear_sky_energy_conserving", "quadrature"])
def test_montecarlo_records_the_app_policy(tmp_path, policy):
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
            "irradiance_resampling": policy,
        },
        MonteCarloSettings(weather_file=str(path), n_runs=1, years_per_run=1, target_year=2025, seed=0),
    )
    provenance = result.provenance
    metadata = provenance["runtime_weather"]["metadata"]
    assert provenance["resolved_config"]["irradiance_resampling"] == policy
    assert metadata["irradiance_resampling"] == policy
    assert metadata["irradiance_resampling_resolved"] == policy
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


def _minute_geometry(result, centre_offset):
    """Apparent-zenith cosine and clear sky at the 15 one-minute samples of each step."""
    times = (result.index + centre_offset).repeat(15) + pd.to_timedelta(np.tile(np.arange(15) - 7, len(result)), "min")
    site = Location(**COORDINATES)
    position = site.get_solarposition(times)
    sky = site.get_clearsky(times, solar_position=position)
    cos_zenith = np.maximum(np.cos(np.deg2rad(position.apparent_zenith.to_numpy())), 0.0)
    return cos_zenith, sky


def test_quadrature_conserves_interval_mean_dni_and_dhi_and_closes_per_sample():
    source = _weather()
    source.iloc[:] = np.array([[500, 400, 300], [700, 600, 350], [600, 500, 450], [300, 200, 250], [400, 100, 700]])
    result = resample_to_15min(source, irradiance_resampling="quadrature", **COORDINATES)
    means = result.to_numpy().reshape(-1, 4, 3).mean(axis=1)
    np.testing.assert_allclose(means[:, 1:], source[["dni", "dhi"]], rtol=1e-12)

    # Each hour follows its clear-sky shape. GHI closes in every one-minute
    # sample, so a step's GHI is its DHI plus the mean of DNI·cos(zenith) over
    # its samples, not its mean DNI times one cosine.
    cos_zenith, sky = _minute_geometry(result, pd.Timedelta(minutes=7.5))
    minutes = {}
    for component in ("dni", "dhi"):
        weights = sky[component].to_numpy().reshape(len(source), 60)
        minutes[component] = (source[component].to_numpy()[:, None] * weights / weights.mean(axis=1)[:, None]).ravel()
    np.testing.assert_allclose(result.dni, minutes["dni"].reshape(-1, 15).mean(axis=1), rtol=1e-12)
    expected_ghi = (minutes["dni"] * cos_zenith + minutes["dhi"]).reshape(-1, 15).mean(axis=1)
    np.testing.assert_allclose(result.ghi, expected_ghi, rtol=1e-12)
    np.testing.assert_allclose(
        result.ghi - result.dhi, (minutes["dni"] * cos_zenith).reshape(-1, 15).mean(axis=1), rtol=1e-12
    )

    metadata = result.attrs[WEATHER_METADATA_KEY]
    assert metadata["irradiance_resampling"] == metadata["irradiance_resampling_resolved"] == "quadrature"
    assert metadata["irradiance_resampling_reconstruction"] == "one_minute_quadrature"
    quadrature = metadata["irradiance_quadrature"]
    assert quadrature["conserved_components"] == ["dni", "dhi"]
    assert quadrature["derived_components"] == ["ghi"]
    derived = means[:, 0]
    assert quadrature["ghi_reconciliation"]["energy_change_fraction"] == pytest.approx(
        derived.sum() / source.ghi.sum() - 1
    )
    assert quadrature["ghi_reconciliation"]["absolute_residual_energy_fraction"] == pytest.approx(
        np.abs(derived - source.ghi).sum() / source.ghi.sum()
    )
    assert metadata["irradiance_closure"]["before"] == metadata["irradiance_closure"]["after"]


def test_quadrature_open_meteo_names_and_right_labels():
    names = ("shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation")
    source = _weather().rename(columns=dict(zip(("ghi", "dni", "dhi"), names, strict=True)))
    source.index = source.index + pd.Timedelta(hours=1)
    source.attrs[WEATHER_METADATA_KEY]["timestamp_label_basis"] = "right"
    result = resample_to_15min(source, irradiance_resampling="quadrature", **COORDINATES)
    assert result.index[0] == source.index[0] - pd.Timedelta(hours=1)
    np.testing.assert_allclose(
        result[list(names[1:])].to_numpy().reshape(-1, 4, 2).mean(axis=1), source[list(names[1:])]
    )
    assert (result[names[0]] >= result[names[2]]).all()


def test_quadrature_interpolates_instant_samples_without_conservation():
    # Samples on a constant clear-sky ratio come back on that ratio at every minute.
    source = _weather("instant")
    source.attrs[WEATHER_METADATA_KEY]["irradiance_time_offset_hours"] = 0.25
    sky = Location(**COORDINATES).get_clearsky(source.index + pd.Timedelta(minutes=15))
    source["dni"] = 0.5 * (sky.dni.to_numpy() + 5.0)
    source["dhi"] = 1.5 * (sky.dhi.to_numpy() + 5.0)
    result = resample_to_15min(source, irradiance_resampling="quadrature", **COORDINATES)
    cos_zenith, minute_sky = _minute_geometry(result, pd.Timedelta(minutes=15))
    dni = 0.5 * (minute_sky.dni.to_numpy() + 5.0)
    dhi = 1.5 * (minute_sky.dhi.to_numpy() + 5.0)
    np.testing.assert_allclose(result.dni, dni.reshape(-1, 15).mean(axis=1), rtol=1e-9)
    np.testing.assert_allclose(result.ghi, (dni * cos_zenith + dhi).reshape(-1, 15).mean(axis=1), rtol=1e-9)
    quadrature = result.attrs[WEATHER_METADATA_KEY]["irradiance_quadrature"]
    assert quadrature["conserved_components"] == []
    # The quarter-hour centred on each sample is compared with it.
    assert quadrature["ghi_reconciliation"]["energy_change_fraction"] == pytest.approx(
        result.ghi.iloc[::4].sum() / source.ghi.sum() - 1
    )


def test_quadrature_night_is_zero_and_unsupported_hours_are_counted():
    source = _weather()
    source = source.iloc[:3].copy()
    source.index = pd.date_range("2025-06-21 00:00", periods=3, freq="h", tz="UTC")
    result = resample_to_15min(source, irradiance_resampling="quadrature", **COORDINATES)
    np.testing.assert_allclose(result[["dni", "dhi"]].to_numpy().reshape(-1, 4, 2).mean(axis=1), source[["dni", "dhi"]])
    np.testing.assert_allclose(result.ghi, result.dhi)
    assert result.attrs[WEATHER_METADATA_KEY]["irradiance_resampling_zero_support_counts"] == {
        "ghi": 0,
        "dni": 3,
        "dhi": 3,
    }

    source.iloc[:] = 0.0
    dark = resample_to_15min(source, irradiance_resampling="quadrature", **COORDINATES)
    assert dark.eq(0).all().all()


def test_quadrature_skips_only_dark_steps(monkeypatch):
    import breos.weather as weather_module

    index = pd.date_range("2025-03-20 04:00", periods=18, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    source = pd.DataFrame(rng.uniform(0, 600, (len(index), 3)), index=index, columns=["ghi", "dni", "dhi"])
    for basis in ("interval_mean", "instant"):
        source.attrs[WEATHER_METADATA_KEY] = {"radiation_time_basis": basis, "timestamp_label_basis": "left"}
        fast = resample_to_15min(source, irradiance_resampling="quadrature", **COORDINATES)
        monkeypatch.setattr(weather_module, "_QUADRATURE_DARK_ZENITH", 180.0)
        full = resample_to_15min(source, irradiance_resampling="quadrature", **COORDINATES)
        monkeypatch.undo()
        pd.testing.assert_frame_equal(fast, full)
        assert fast.attrs == full.attrs


def test_quadrature_requires_coordinates_components_and_hourly_index():
    with pytest.raises(ValueError, match="quadrature.*latitude and longitude"):
        resample_to_15min(_weather(), irradiance_resampling="quadrature")
    with pytest.raises(ValueError, match="quadrature.*missing: ghi"):
        resample_to_15min(_weather().drop(columns="ghi"), irradiance_resampling="quadrature", **COORDINATES)
    irregular = _weather().drop(index=_weather().index[2])
    with pytest.raises(ValueError, match="quadrature requires a regular hourly index"):
        resample_to_15min(irregular, irradiance_resampling="quadrature", **COORDINATES)


def test_quadrature_leaves_other_columns_to_interpolation():
    source = _weather()
    source["temp_air"] = [10.0, 12.0, 14.0, 13.0, 11.0]
    quadrature = resample_to_15min(source, irradiance_resampling="quadrature", **COORDINATES)
    conserving = resample_to_15min(source, **COORDINATES)
    assert list(quadrature.columns) == list(source.columns)
    pd.testing.assert_series_equal(quadrature.temp_air, conserving.temp_air)


@pytest.mark.parametrize("basis", ["interval_mean", "instant"])
@pytest.mark.parametrize("component", ["ghi", "dni", "dhi"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_quadrature_refuses_non_finite_irradiance(basis, component, bad):
    source = _weather(basis)
    source.loc[source.index[2], component] = bad
    with pytest.raises(ValueError, match=f"quadrature.*finite irradiance.*'{component}' has 1 missing"):
        resample_to_15min(source, irradiance_resampling="quadrature", **COORDINATES)


def test_quadrature_reconciliation_is_undefined_without_input_ghi():
    source = _weather()
    source[["ghi", "dni", "dhi"]] = [0.0, 400.0, 100.0]
    result = resample_to_15min(source, irradiance_resampling="quadrature", **COORDINATES)
    assert result.ghi.sum() > 0
    reconciliation = result.attrs[WEATHER_METADATA_KEY]["irradiance_quadrature"]["ghi_reconciliation"]
    assert reconciliation == {"energy_change_fraction": None, "absolute_residual_energy_fraction": None}


def test_quadrature_step_means_do_not_close_with_one_cosine():
    # Porto, 20 March, a dawn quarter-hour: the step's GHI is its DHI plus the
    # mean of DNI·cos(zenith) over the samples, which differs from its mean
    # DNI times the sample-mean cosine.
    index = pd.date_range("2025-03-20 06:00", periods=4, freq="h", tz="UTC")
    source = pd.DataFrame({"ghi": 300.0, "dni": 400.0, "dhi": 100.0}, index=index)
    source.attrs[WEATHER_METADATA_KEY] = {"radiation_time_basis": "interval_mean", "timestamp_label_basis": "left"}
    result = resample_to_15min(source, irradiance_resampling="quadrature", **COORDINATES)
    cos_zenith, _ = _minute_geometry(result, pd.Timedelta(minutes=7.5))
    step_cos = cos_zenith.reshape(-1, 15).mean(axis=1)
    residual = np.abs(result.ghi - result.dni * step_cos - result.dhi)
    assert residual.iloc[3] > 5.0
