"""Tests for the BREOS command line interface."""

import json

from breos import cli


class FakeApp:
    seen_config = None
    simulated = False

    def __init__(self, config):
        self.config = config
        FakeApp.seen_config = config

    def simulate(self):
        FakeApp.simulated = True

    def result(self):
        return {
            "grid_independence_pct": 42.0,
            "config": self.config,
        }


def test_run_from_flags_outputs_json(monkeypatch, capsys):
    monkeypatch.setattr(cli, "App", FakeApp)

    exit_code = cli.main(
        [
            "run",
            "--location",
            "porto",
            "--n-modules",
            "10",
            "--annual-consumption-kwh",
            "4000",
            "--battery-kwh",
            "5.0",
            "--cost-preset",
            "residential-pt",
            "--emissions-country",
            "pt",
            "--load-profile",
            "6",
            "--rlp-directory",
            "/tmp/external-rlp",
        ]
    )

    assert exit_code == 0
    assert FakeApp.simulated is True
    assert FakeApp.seen_config["location"] == "porto"
    assert FakeApp.seen_config["n_modules"] == 10
    assert FakeApp.seen_config["annual_consumption_kwh"] == 4000
    assert FakeApp.seen_config["battery_kwh"] == 5.0
    assert FakeApp.seen_config["cost_preset"] == "residential_pt"
    assert FakeApp.seen_config["emissions_country"] == "PT"
    assert FakeApp.seen_config["load_profile"] == "6"
    assert FakeApp.seen_config["rlp_directory"] == "/tmp/external-rlp"

    output = json.loads(capsys.readouterr().out)
    assert output["grid_independence_pct"] == 42.0


def test_run_from_toml_config_with_cli_override(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "App", FakeApp)
    config_path = tmp_path / "experiment.toml"
    config_path.write_text(
        """
location = "porto"
n_modules = 8
annual_consumption_kwh = 3500
battery_kwh = 0
cost_preset = "residential_pt"
""".strip(),
        encoding="utf-8",
    )
    output_path = tmp_path / "result.json"

    exit_code = cli.main(
        [
            "run",
            "--config",
            str(config_path),
            "--battery-kwh",
            "5.0",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert FakeApp.seen_config["n_modules"] == 8
    assert FakeApp.seen_config["battery_kwh"] == 5.0
    assert json.loads(output_path.read_text(encoding="utf-8"))["grid_independence_pct"] == 42.0


def test_invalid_json_config_reports_filename(tmp_path, capsys):
    config_path = tmp_path / "broken_config.json"
    config_path.write_text('{"location": "porto",', encoding="utf-8")

    exit_code = cli.main(["run", "--config", str(config_path)])

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "Invalid JSON in" in stderr
    assert "broken_config.json" in stderr
