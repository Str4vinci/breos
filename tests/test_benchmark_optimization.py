"""The optimizer backend benchmark checks its arguments, compares results exactly and reports a fixed shape."""

import gzip
import json

import numpy as np
import pandas as pd
import pytest

from tools import benchmark_optimization as bench


def test_defaults_match_the_documented_study():
    args = bench.parse_args([])

    assert args.weather_file == bench.DEFAULT_WEATHER_FILE and args.weather_file.is_file()
    assert args.resolution == ["h", "15min"]
    assert (args.projection_years, args.pop_size, args.n_gen, args.seed) == (3, 8, 2, 42)
    assert (args.n_procs, args.warm_repeats, args.output) == (1, 3, None)


def test_app_witness_records_every_controller_year(tmp_path):
    staged, location, _ = bench.stage_weather(bench.DEFAULT_WEATHER_FILE, tmp_path)
    settings = dict(latitude=41.1579, longitude=-8.6291, annual_consumption_kwh=3500.0)
    weather, load, _ = bench.prepare_case_inputs(staged, location, "h", **settings)
    app_config, optimizer_config = bench.study_configs("h", 3, **settings)

    run = bench._run_app(
        dict(weather=weather, load=load, app_config=app_config, optimizer_config=optimizer_config), "python"
    )

    assert len(run["ledgers"]) == 3
    assert all(len(ledger["results"]) == len(weather) for ledger in run["ledgers"])
    grid_kwh = sum(ledger["results"]["Grid_AC_To_Battery"].sum() / 1000.0 for ledger in run["ledgers"])
    assert grid_kwh > 0.0
    assert grid_kwh == pytest.approx(run["yearly"]["Grid_AC_To_Battery_kWh"].sum(), rel=1e-15)
    assert run["total_replacements"] > 0


@pytest.mark.parametrize(
    "argv",
    [
        ["--pop-size", "0"],
        ["--pop-size", "3"],
        ["--warm-repeats", "2"],
        ["--n-gen", "-1"],
        ["--warm-repeats", "0"],
        ["--projection-years", "two"],
        ["--n-procs", "0"],
        ["--seed", "-1"],
        ["--annual-consumption-kwh", "0"],
        ["--resolution", "30min"],
        ["--resolution", "h", "h"],
        ["--latitude", "91"],
        ["--longitude", "-181"],
        ["--weather-file", "missing_tmy_2020_2020_x.csv"],
    ],
)
def test_invalid_arguments_exit_with_usage_error(argv, capsys):
    with pytest.raises(SystemExit) as excinfo:
        bench.parse_args(argv)
    assert excinfo.value.code == 2
    assert "error" in capsys.readouterr().err


def test_fewer_warm_repeats_need_the_smoke_flag():
    args = bench.parse_args(["--smoke", "--warm-repeats", "1"])
    assert args.smoke and args.warm_repeats == 1
    assert bench.parse_args(["--pop-size", "4"]).pop_size == 4


def test_weather_file_must_be_csv(tmp_path):
    other = tmp_path / "porto.parquet"
    other.write_bytes(b"")
    with pytest.raises(SystemExit):
        bench.parse_args(["--weather-file", str(other)])


def test_staged_default_weather_keeps_its_bound_metadata(tmp_path):
    from breos.weather import WEATHER_METADATA_KEY, load_weather

    staged, location, record = bench.stage_weather(bench.DEFAULT_WEATHER_FILE, tmp_path)

    assert (staged.name, location) == ("porto_tmy_2005_2023_pvgis-sarah3.csv", "porto")
    assert record["compressed"] and record["sha256"] == bench.sha256_file(bench.DEFAULT_WEATHER_FILE)
    assert record["sidecar"]["status"] == "bound"
    assert record["staged_sha256"] == bench.sha256_file(staged) != record["sha256"]
    metadata = load_weather(location=location, weather_dir=str(tmp_path)).attrs[WEATHER_METADATA_KEY]
    # The sidecar is bound to the staged CSV, so the loader reads its timing fields.
    assert metadata["upstream_source"] == "PVGIS_TMY"
    assert (
        metadata["irradiance_time_offset_hours"]
        == record["sidecar"]["payload"]["breos_weather_metadata"]["irradiance_time_offset_hours"]
    )


def _sidecar(path, sha256, schema_version=1):
    payload = {
        "schema_version": schema_version,
        "weather_sha256": sha256,
        "breos_weather_metadata": {"irradiance_time_offset_hours": 0.25},
    }
    path.write_text(json.dumps(payload))


@pytest.mark.parametrize(
    ("sha256", "schema_version", "message"),
    [("0" * 64, 1, "matches neither"), (None, 2, "schema_version")],
)
def test_sidecar_that_fails_a_check_stops_the_benchmark(tmp_path, sha256, schema_version, message):
    source = tmp_path / "site_tmy_2020_2020_local.csv"
    source.write_text("time,ghi\n")
    sidecar = tmp_path / "site_tmy_2020_2020_local.csv.metadata.json"
    _sidecar(sidecar, sha256 or bench.sha256_file(source), schema_version)

    with pytest.raises(ValueError, match=message):
        bench.stage_weather(source, tmp_path / "staged")


def test_gzipped_csv_uses_the_sidecar_bound_to_its_csv(tmp_path):
    plain = tmp_path / "site_tmy_2020_2020_local.csv"
    plain.write_text("time,ghi\n")
    _sidecar(tmp_path / f"{plain.name}.metadata.json", bench.sha256_file(plain))
    compressed = tmp_path / f"{plain.name}.gz"
    compressed.write_bytes(gzip.compress(plain.read_bytes()))
    plain.unlink()

    staged, _, record = bench.stage_weather(compressed, tmp_path / "staged")

    assert record["sidecar"]["status"] == "bound" and record["sidecar"]["bound_to"] == "decompressed CSV"
    restaged = json.loads((staged.parent / f"{staged.name}.metadata.json").read_text())
    assert restaged["weather_sha256"] == bench.sha256_file(staged)


def test_weather_without_a_sidecar_is_recorded_as_absent(tmp_path):
    source = tmp_path / "site_tmy_2020_2020_local.csv"
    source.write_text("time,ghi\n")
    staged, location, record = bench.stage_weather(source, tmp_path / "staged")
    assert location == "site" and record["sidecar"] == {"status": "absent"}
    assert not (staged.parent / f"{staged.name}.metadata.json").exists()


def _quarter_hour_weather(path, periods=None):
    index = pd.date_range("2026-01-01", "2027-01-01", freq="15min", inclusive="left", tz="UTC")[:periods]
    frame = pd.DataFrame({"ghi": 0.0, "dni": 0.0, "dhi": 0.0, "temp_air": 15.0, "wind_speed": 2.0}, index=index)
    frame.index.name = "time"
    frame.to_csv(path)
    return path


def test_hourly_case_rejects_quarter_hour_weather(tmp_path):
    staged = _quarter_hour_weather(tmp_path / "site_tmy_2026_2026_quarter.csv", periods=12)
    with pytest.raises(ValueError, match="hourly case needs hourly weather"):
        bench.prepare_case_inputs(staged, "site", "h", latitude=41.0, longitude=-8.0, annual_consumption_kwh=3500.0)


@pytest.fixture(scope="module")
def quarter_hour_year(tmp_path_factory):
    return _quarter_hour_weather(tmp_path_factory.mktemp("weather") / "site_tmy_2026_2026_quarter.csv")


def test_every_case_is_checked_before_the_first_child(quarter_hour_year, monkeypatch):
    monkeypatch.setattr(bench, "run_child", lambda *args: pytest.fail("a child started before the inputs were checked"))
    argv = ["--weather-file", str(quarter_hour_year), "--resolution", "15min", "h", "--smoke", "--warm-repeats", "1"]
    with pytest.raises(ValueError, match="hourly case needs hourly weather"):
        bench.main(argv)


def test_a_failed_child_still_writes_a_partial_report(quarter_hour_year, monkeypatch, tmp_path):
    calls = []

    def child(kind, spec, workdir, cache):
        calls.append(kind)
        if kind == "parity":
            return _parity()
        raise RuntimeError("child measure-15min-cold-python-0 failed")

    monkeypatch.setattr(bench, "run_child", child)
    output = tmp_path / "out" / "report.json"
    argv = ["--weather-file", str(quarter_hour_year), "--resolution", "15min", "--smoke", "--warm-repeats", "1"]

    assert bench.main([*argv, "--output", str(output)]) == 1

    report = json.loads(output.read_text())
    assert calls == ["parity", "measure"]
    assert report["status"] == "failed" and "measure-15min-cold-python-0" in report["error"]
    assert report["cases"][0]["parity"]["status"] == "passed" and report["cases"][0]["status"] == "failed"
    assert report["smoke"] and report["warm_repeats_below_contract"]


def _year(freq="h"):
    index = pd.date_range("2026-01-01", "2027-01-01", freq=freq, inclusive="left", tz="UTC")
    return pd.DataFrame({"value": np.arange(len(index), dtype=float)}, index=index)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda frame: frame.drop(frame.index[100]), "regular"),
        (lambda frame: frame.iloc[:-1], "rows"),
        (lambda frame: pd.concat([frame, frame.iloc[:1]]), "repeats"),
        (lambda frame: frame.tz_localize(None), "timezone"),
        (lambda frame: frame.iloc[::-1], "increasing"),
    ],
)
def test_incomplete_or_ambiguous_year_is_rejected(damage, message):
    bench.check_complete_year(_year(), "h", "weather")
    bench.check_complete_year(_year("15min"), "15min", "weather")
    with pytest.raises(ValueError, match=message):
        bench.check_complete_year(damage(_year()), "h", "weather")


def _pareto():
    return pd.DataFrame(
        {
            "Modules": [12, 8, 8],
            "Battery_kWh": [0.0, 5.0, 3.0],
            "Tilt": [30.0, 35.0, 35.0],
            "Azimuth": [180.0, 180.0, 180.0],
            "Projected_NPV": [1500.0, np.nan, 900.0],
        }
    )


def test_pareto_rows_compare_in_design_order():
    pareto = _pareto()
    shuffled = pareto.iloc[[2, 0, 1]].copy()
    shuffled.attrs["currency"] = "EUR"

    canonical = bench.canonical_pareto(shuffled)

    assert canonical[["Modules", "Battery_kWh"]].to_numpy().tolist() == [[8, 3.0], [8, 5.0], [12, 0.0]]
    assert canonical.attrs == {} and list(canonical.index) == [0, 1, 2]
    # NaN (a design that never breaks even) is equal to NaN.
    assert bench.value_differences(bench.canonical_pareto(pareto), canonical, "pareto") == []


def test_every_difference_is_listed():
    left = bench.canonical_pareto(_pareto())
    changed = left.copy()
    changed.loc[1, "Projected_NPV"] = 0.0
    assert bench.value_differences(left, changed, "pareto") == ["pareto[Projected_NPV]: values differ"]

    assert "columns differ" in bench.value_differences(left, left.drop(columns="Tilt"), "p")[0]
    assert "dtype" in bench.value_differences(left, left.astype({"Modules": float}), "p")[0]
    assert "rows" in bench.value_differences(left, left.iloc[:2], "p")[0]
    assert bench.value_differences({"a": [1.0, np.nan]}, {"a": [1.0, np.nan]}, "d") == []
    assert bench.value_differences({"a": 1}, {"b": 1}, "d") == ["d: keys differ (only left ['a'], only right ['b'])"]
    assert bench.value_differences({"soh": 0.9}, {"soh": 0.9 + 1e-15}, "d")[0].startswith("d.soh: 0.9 against")
    assert bench.value_differences([left], [left, left], "ledgers") == ["ledgers: length 1 against 2"]


def _measurement(backend, phase, total, construction=0.5, repeat=0):
    return {
        "phase": phase,
        "backend": backend,
        "repeat": repeat,
        "total_s": total,
        "problem_construction_s": construction,
        "residual_search_s": total - construction,
        "peak_rss_mib": 300.0 if backend == "python" else 450.0,
        "evaluations": 16,
        "generations": 2,
    }


def test_summary_reports_cold_warm_and_speedups():
    measurements = [
        _measurement("python", "cold", 10.5),
        _measurement("numba", "cold", 30.5),
        *(_measurement("python", "warm", total, repeat=i) for i, total in enumerate((10.5, 12.5, 8.5))),
        *(_measurement("numba", "warm", total, repeat=i) for i, total in enumerate((2.5, 2.0, 3.0))),
    ]

    summary = bench.summarize_case(measurements)

    python, numba = summary["backends"]["python"], summary["backends"]["numba"]
    assert python["warm"]["total_s"] == {"median": 10.5, "min": 8.5}
    assert numba["cold"]["total_s"] == 30.5 and numba["warm"]["repeats"] == 3
    assert numba["warm"]["residual_search_s"] == {"median": 2.0, "min": 1.5}
    assert numba["peak_rss_mib_max"] == 450.0
    speedup = summary["speedup_vs_python_warm_median"]
    assert speedup["python"]["warm_total"] == 1.0
    assert speedup["numba"]["warm_total"] == pytest.approx(10.5 / 2.5)
    assert speedup["numba"]["warm_residual_search"] == pytest.approx(10.0 / 2.0)
    assert speedup["numba"]["cold_total"] == pytest.approx(10.5 / 30.5)
    assert summary["evaluation_counts_consistent"]
    assert bench.summarize_case(measurements, {"evaluations": 16, "generations": 2})["evaluation_counts_consistent"]
    measurements[0] = {**measurements[0], "evaluations": 15}
    differing = bench.summarize_case(measurements, {"evaluations": 16, "generations": 2})
    assert differing["count_mismatches"] == ["cold python 0: 15 evaluations, 2 generation(s)"]
    case = {"parity": _parity(), "summary": differing}
    # One worker gates the case on the counts; several workers only record them.
    assert (bench.case_status(case, n_procs=1), bench.case_status(case, n_procs=2)) == ("failed", "passed")
    assert any(line.startswith("  COUNTS DIFFER") for line in bench.format_case({"resolution": "h", **case}))


@pytest.mark.parametrize(
    ("platform", "status", "raw", "expected"),
    [
        ("linux", "Name:\tpython\nVmHWM:\t   14336 kB\nVmRSS:\t 1 kB\n", 851968, (14.0, "linux_vmhwm")),
        ("linux", None, 2048, (2.0, "ru_maxrss")),
        ("darwin", None, 2 * 2**20, (2.0, "ru_maxrss")),
        ("win32", None, None, (None, "unavailable")),
    ],
)
def test_peak_rss_names_its_source_and_unit(tmp_path, monkeypatch, platform, status, raw, expected):
    status_path = tmp_path / "status"
    if status is not None:
        status_path.write_text(status)
    monkeypatch.setattr(bench, "_ru_maxrss", lambda: raw)

    assert bench.peak_rss(platform, status_path) == expected


def _parity(status="passed"):
    failed = status != "passed"
    return {
        "status": status,
        "checks": [{"name": "witness_replaces_battery", "passed": not failed, "differences": []}],
        "counts": {"python": {"evaluations": 16, "generations": 2, "pareto_designs": 3, "feasible_pareto_designs": 3}},
        "witness": {"grid_ac_to_battery_kwh": 900.0, "projected_total_replacements": 1.0},
    }


def test_report_has_a_fixed_shape_and_fails_on_parity():
    args = bench.parse_args(["--resolution", "h"])
    case = {
        "resolution": "h",
        "status": "passed",
        "parity": _parity(),
        "summary": bench.summarize_case([_measurement("python", "warm", 2)]),
    }

    report = bench.build_report(args, {"python": "3.x"}, {"sha256": "abc"}, [case])

    assert report["schema"] == bench.REPORT_SCHEMA and report["status"] == "passed"
    assert (report["smoke"], report["warm_repeats_below_contract"]) == (False, False)
    assert report["entrypoint"] == "breos.optimization.optimize_system_multi_objective"
    assert set(report) >= {"machine", "settings", "timing_boundaries", "study", "weather_file", "cases"}
    assert report["settings"]["parity_n_procs"] == 1 and report["settings"]["seed"] == 42
    assert "not a pure dispatch time" in report["timing_boundaries"]["residual_search_s"]
    json.dumps(report)
    assert any("python warm" in line for line in bench.format_case(case))

    failed = {"resolution": "h", "parity": _parity("failed")}
    assert bench.build_report(args, {}, {}, [failed])["status"] == "failed"
    assert any(line.startswith("  FAILED witness_replaces_battery") for line in bench.format_case(failed))
