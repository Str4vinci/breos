"""Every JSON file or stream the CLI writes is strict JSON (no NaN/Infinity tokens)."""

import csv
import json

import numpy as np
import pandas as pd
import pytest

import breos.montecarlo as montecarlo_module
from breos import cli
from breos.io import nonfinite_to_none
from breos.montecarlo import MonteCarloResult, MonteCarloSettings, _summarize, run_montecarlo


def _reject_constant(token):
    raise ValueError(f"non-standard JSON token {token}")


def _strict_loads(text):
    """Parse ``text`` as standard JSON: ``NaN``, ``Infinity`` and ``-Infinity`` raise."""
    return json.loads(text, parse_constant=_reject_constant)


def test_strict_parser_rejects_the_tokens_python_json_writes():
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="non-standard JSON token"):
            _strict_loads(json.dumps({"x": value}))


def test_nonfinite_to_none_walks_nested_payloads():
    payload = {
        "lcoe": float("inf"),
        "numpy": np.float64("nan"),
        "float32": np.float32(0.5),
        "negative": -np.inf,
        "rows": [{"a": 1.0, "b": float("nan")}, (2, float("-inf"))],
        "kept": [1, "text", None, True],
    }

    assert nonfinite_to_none(payload) == {
        "lcoe": None,
        "numpy": None,
        "float32": 0.5,
        "negative": None,
        "rows": [{"a": 1.0, "b": None}, [2, None]],
        "kept": [1, "text", None, True],
    }
    assert type(nonfinite_to_none(np.float64(1.5))) is float


class _UndefinedLcoeApp:
    """An App whose result has the undefined metrics a zero-production run gives."""

    def __init__(self, config):
        self.config = config

    def simulate(self):
        return None

    def result(self):
        return {
            "n_modules": self.config.get("n_modules"),
            "grid_independence_pct": 0.0,
            "lcoe_eur_kwh": float("inf"),
            "yearly": [{"year": 1, "lcoe_eur_kwh": np.float64("inf"), "ratio": float("nan")}],
        }


def _write_config(tmp_path, body):
    path = tmp_path / "config.toml"
    path.write_text(body.strip() + "\n", encoding="utf-8")
    return path


_BASE_TOML = """
location = "porto"
n_modules = 8
annual_consumption_kwh = 3500
battery_kwh = 0
cost_preset = "residential_pt"
"""


def test_run_writes_undefined_metrics_as_null_to_stdout_and_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "App", _UndefinedLcoeApp)
    config = _write_config(tmp_path, _BASE_TOML)

    assert cli.main(["run", "--config", str(config)]) == 0
    stdout = _strict_loads(capsys.readouterr().out)

    output = tmp_path / "result.json"
    assert cli.main(["run", "--config", str(config), "--output", str(output), "--indent", "0"]) == 0
    written = _strict_loads(output.read_text(encoding="utf-8"))

    for payload in (stdout, written):
        assert payload["lcoe_eur_kwh"] is None
        assert payload["yearly"] == [{"year": 1, "lcoe_eur_kwh": None, "ratio": None}]
        assert payload["grid_independence_pct"] == 0.0


def test_run_dry_run_writes_strict_json(tmp_path, capsys):
    config = _write_config(tmp_path, _BASE_TOML)

    assert cli.main(["run", "--config", str(config), "--dry-run"]) == 0

    assert _strict_loads(capsys.readouterr().out)["valid"] is True


def test_leftover_nonfinite_value_fails_with_a_clear_error():
    with pytest.raises(ValueError, match="Cannot write the config summary as JSON: it contains a non-finite number"):
        cli._json_text({"x": float("nan")}, "the config summary")


def test_sweep_json_reports_undefined_metrics_as_null(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "App", _UndefinedLcoeApp)
    config = _write_config(tmp_path, _BASE_TOML + "\n[sweep]\nn_modules = [8, 10]\n")
    output = tmp_path / "sweep.csv"

    assert cli.main(["sweep", "--config", str(config), "--output", str(output), "--json"]) == 0

    payload = _strict_loads(capsys.readouterr().out)
    assert payload["runs"] == 2
    assert [row["lcoe_eur_kwh"] for row in payload["rows"]] == [None, None]
    # The CSV keeps its own spelling of an undefined value.
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert [row["lcoe_eur_kwh"] for row in rows] == ["inf", "inf"]


@pytest.mark.parametrize(
    "category", ["locations", "modules", "cost-presets", "emissions", "load-profiles", "battery-models"]
)
def test_list_json_is_strict(category, capsys):
    assert cli.main(["list", category, "--json"]) == 0

    assert _strict_loads(capsys.readouterr().out)


def test_validate_config_json_is_strict(tmp_path, capsys):
    config = _write_config(tmp_path, _BASE_TOML)

    assert cli.main(["validate-config", str(config), "--json"]) == 0

    assert _strict_loads(capsys.readouterr().out)["valid"] is True


def _montecarlo_config(tmp_path, extra=""):
    weather = tmp_path / "weather.csv"
    weather.write_text("date\n")
    return _write_config(tmp_path, _BASE_TOML + f'\n[montecarlo]\nweather_file = "{weather}"\n{extra}')


def test_montecarlo_provenance_and_json_write_nonfinite_statistics_as_null(monkeypatch, tmp_path, capsys):
    config = _montecarlo_config(tmp_path)
    runs = pd.DataFrame(
        {
            "run": [1, 2, 3],
            "npv_savings_eur": [-100.0, 0.0, 100.0],
            "lcoe_eur_kwh": [0.12, float("inf"), 0.14],
        }
    )

    def fake_run(_config, settings):
        summary = _summarize(runs)
        # A statistic with no defined value, the case the CLI must still write.
        summary["npv_savings_eur"]["std"] = float("nan")
        summary["lcoe_eur_kwh"]["max"] = float("inf")
        return MonteCarloResult(
            runs=runs,
            summary=summary,
            settings=settings,
            available_years=[2021],
            provenance={"settings": {"max_load_scale": settings.max_load_scale}},
        )

    monkeypatch.setattr(montecarlo_module, "run_montecarlo", fake_run)
    output = tmp_path / "out.csv"

    assert cli.main(["montecarlo", "--config", str(config), "--output", str(output), "--json"]) == 0

    payload = _strict_loads(capsys.readouterr().out)
    provenance = _strict_loads((tmp_path / "out.provenance.json").read_text())
    for summary in (payload["summary"], provenance["summary"]):
        assert summary["npv_savings_eur"]["std"] is None
        assert summary["lcoe_eur_kwh"]["max"] is None
        assert summary["lcoe_eur_kwh"]["count"] == 2
        assert summary["lcoe_eur_kwh"]["mean"] == pytest.approx(0.13)
    assert payload["settings"]["max_load_scale"] is None
    assert provenance["settings"]["max_load_scale"] is None


def test_montecarlo_cli_rejects_an_infinite_max_load_scale(tmp_path, capsys):
    config = _montecarlo_config(tmp_path, "max_load_scale = inf")

    assert cli.main(["montecarlo", "--config", str(config), "--output", str(tmp_path / "out.csv"), "--json"]) == 1

    err = capsys.readouterr().err
    assert "max_load_scale must be a finite number, or None to leave the load scale unbounded, got inf" in err
    assert not (tmp_path / "out.provenance.json").exists()


@pytest.mark.parametrize("bound", [float("inf"), float("-inf"), float("nan")])
def test_run_montecarlo_rejects_a_nonfinite_max_load_scale(tmp_path, bound):
    settings = MonteCarloSettings(weather_file=str(tmp_path / "unused.csv"), n_runs=1, max_load_scale=bound)

    with pytest.raises(ValueError, match="max_load_scale must be a finite number, or None"):
        run_montecarlo({"location": "porto", "n_modules": 8, "annual_consumption_kwh": 4000}, settings)


def test_run_montecarlo_accepts_an_unbounded_max_load_scale(tmp_path, write_multiyear_weather):
    weather = write_multiyear_weather(tmp_path / "multi.csv", years=(2021,))
    settings = MonteCarloSettings(weather_file=str(weather), n_runs=1, years_per_run=1, seed=0, max_load_scale=None)
    config = {"location": "porto", "n_modules": 8, "annual_consumption_kwh": 4000, "projection_years": 1}

    result = run_montecarlo(config, settings)

    assert len(result.runs) == 1
    assert result.provenance["settings"]["max_load_scale"] is None
    _strict_loads(cli._json_text(nonfinite_to_none(result.summary), "the summary"))
