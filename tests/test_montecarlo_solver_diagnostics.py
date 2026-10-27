"""Monte Carlo retains the parent's recovery counts across years and workers."""

import json
import multiprocessing
import os

import numpy as np
import pvlib
import pytest

import breos.montecarlo as mc
from breos import cli
from tests.test_montecarlo import _base_config


@pytest.mark.parametrize("start_method", [None, "fork", "spawn"], ids=["serial", "fork", "spawn"])
@pytest.mark.parametrize("cached", [False, True], ids=["fresh", "cached"])
def test_recovery_belongs_to_its_source_weather_year_and_survives_workers(
    tmp_path, monkeypatch, write_multiyear_weather, start_method, cached
):
    if start_method is not None and start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{start_method} is unavailable")
    weather = write_multiyear_weather(tmp_path / "weather.csv")
    settings = mc.MonteCarloSettings(
        weather_file=str(weather),
        n_runs=2,
        years_per_run=1,
        seed=11,
        n_procs=1 if start_method is None else 2,
    )
    config = {**_base_config(), "battery_kwh": 0.0}
    original = pvlib.pvsystem.max_power_point
    injected = False
    fallback_processes = []

    def forced(*args, method="newton", **kwargs):
        nonlocal injected
        result = original(*args, method=method, **kwargs)
        if method == "newton" and not injected:
            row = np.flatnonzero(np.asarray(args[0]) > 0.1)[0]
            result["p_mp"][row] = np.nan
            injected = True
        elif method == "brentq":
            fallback_processes.append(os.getpid())
        return result

    monkeypatch.setattr(pvlib.pvsystem, "max_power_point", forced)
    if start_method is not None:
        monkeypatch.setattr(mc, "Pool", multiprocessing.get_context(start_method).Pool)
        monkeypatch.setattr(mc.multiprocessing, "get_start_method", lambda: start_method)
    cache = mc.build_year_cache(config, settings) if cached else None
    with monkeypatch.context() as reuse:
        if cached:
            # The record must come from the cached preparation, not a new solve.
            def unexpected(*args, **kwargs):
                pytest.fail("cached PV inputs were rebuilt")

            reuse.setattr(mc, "build_pv_production_breakdown", unexpected)
        result = mc.run_montecarlo(config, settings, year_cache=cache)
    expected = {
        "by_weather_year": {
            2021: {"recovered_steps": 1, "by_stage": {"module_mpp": 1}},
            2022: {"recovered_steps": 0, "by_stage": {}},
        },
        "total": {"recovered_steps": 1, "by_stage": {"module_mpp": 1}},
    }
    assert injected
    assert fallback_processes == [os.getpid()]
    assert result.solver == expected
    assert result.provenance["solver"] == expected
    if cached and start_method is None:
        result.solver["by_weather_year"][2021]["by_stage"].clear()
        reused = mc.run_montecarlo(config, settings, year_cache=cache)
        assert reused.solver == expected
        # A different PV design rebuilds the counts together with the DC inputs.
        rebuilt = mc.run_montecarlo({**config, "n_modules": 10}, settings, year_cache=cache)
        assert rebuilt.solver["total"] == {"recovered_steps": 0, "by_stage": {}}


def test_zero_counts_are_in_the_python_result_and_cli_json(tmp_path, monkeypatch, capsys, write_multiyear_weather):
    weather = write_multiyear_weather(tmp_path / "weather.csv")
    config = {**_base_config(), "battery_kwh": 0.0}
    settings = mc.MonteCarloSettings(weather_file=str(weather), n_runs=1, years_per_run=1, seed=11)
    result = mc.run_montecarlo(config, settings)
    zero = {"recovered_steps": 0, "by_stage": {}}
    assert result.solver == {"by_weather_year": {2021: zero, 2022: zero}, "total": zero}
    assert result.provenance["solver"] == result.solver
    # Exercise serialization with a recovery as well as zero entries.
    result.solver["by_weather_year"][2021] = {"recovered_steps": 1, "by_stage": {"module_mpp": 1}}
    result.solver["total"] = {"recovered_steps": 1, "by_stage": {"module_mpp": 1}}
    monkeypatch.setattr(mc, "run_montecarlo", lambda *a, **kw: result)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({**config, "montecarlo": {"weather_file": str(weather)}}))
    assert cli.main(["montecarlo", "--config", str(config_path), "--output", str(tmp_path / "runs.csv"), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    expected = json.loads(json.dumps(result.solver))
    assert payload["solver"] == expected
    assert json.loads((tmp_path / "runs.provenance.json").read_text())["solver"] == expected
