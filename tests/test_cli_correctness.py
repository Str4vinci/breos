"""CLI coverage for correctness-release battery contracts."""

import pytest

from breos import cli


class _FakeApp:
    seen = None

    def __init__(self, config):
        type(self).seen = config

    def simulate(self):
        return None

    def result(self):
        return {"ok": True}


def test_cli_threads_battery_power_limits(monkeypatch, capsys):
    monkeypatch.setattr(cli, "App", _FakeApp)

    assert (
        cli.main(
            [
                "run",
                "--location",
                "porto",
                "--n-modules",
                "10",
                "--annual-consumption-kwh",
                "4000",
                "--battery-max-charge-power-w",
                "2500",
                "--battery-max-discharge-power-w",
                "1800",
                "--export-emissions-factor-gco2-kwh",
                "120",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert _FakeApp.seen["battery_max_charge_power_w"] == 2500
    assert _FakeApp.seen["battery_max_discharge_power_w"] == 1800
    assert _FakeApp.seen["export_emissions_factor_gco2_kwh"] == 120


def test_cli_does_not_advertise_unsupported_ac_coupling(capsys):
    with pytest.raises(SystemExit):
        cli.main(["run", "--ac-coupled"])
    assert "unrecognized arguments: --ac-coupled" in capsys.readouterr().err
