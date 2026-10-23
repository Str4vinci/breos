"""Standalone site identity, full-horizon carry and settlement integration."""

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import breos
import breos.app_inputs as input_module
import breos.projection as projection_module
from breos import App
from breos.community.config import SHARED_APP_KEYS, HouseholdConfig, parse_community_config
from breos.community.runner import run_community
from breos.projection import build_battery_config, build_pv_only_battery_config
from tests.community_conservation import assert_settlement_closure
from tests.conftest import _build_synthetic_weather
from tests.energy_conservation import assert_energy_conservation
from tests.test_ac_several_inverters import APP_AC, TOU
from tests.test_multi_inverter import TWO_INVERTERS


@pytest.fixture
def offline_deps(monkeypatch):
    weather = _build_synthetic_weather(year=2026, freq="15min")
    monkeypatch.setattr("breos.app.fetch_tmy_weather_data", lambda **_kwargs: (weather.copy(deep=True), {}))
    monkeypatch.setattr("breos.app.load_weather", lambda **_kwargs: None)
    return App._runtime_dependencies()


@pytest.fixture
def community_case(trina):
    multiple = {key: deepcopy(value) for key, value in TWO_INVERTERS.items() if key not in SHARED_APP_KEYS}
    # A deliberate start-up-voltage what-if leaves the second inverter idle
    # while the first exports. Its night draw must be netted by M9.
    multiple["inverters"][1].update(startup_voltage_v=900, night_consumption_w=8)
    multiple["ac_cable"] = {"length_m": 40, "cross_section_mm2": 2.5, "phases": 1, "voltage_v": 230}
    return {
        "community": {
            "location": "porto",
            "start_date": "2026-01-01",
            "resolution": "15min",
            "projection_years": 2,
            "allocation": "proportional",
            "shared_energy_price": 0.1,
            "shared_network_charge": 0.02,
            "surplus_price": 0.04,
        },
        "participants": [
            {
                "id": "ac",
                "pv_arrays": deepcopy(multiple["pv_arrays"][:1]),
                "inverters": [{**deepcopy(multiple["inverters"][0]), "night_consumption_w": 5}],
                "pv_strings": deepcopy(multiple["pv_strings"][:1]),
                "electrical_environment": deepcopy(multiple["electrical_environment"]),
                "electrical_fidelity": "string",
                "annual_consumption_kwh": 4000,
                **deepcopy(APP_AC),
                "tariff": deepcopy(TOU),
                "smart_charging": {
                    "mode": "fixed_target",
                    "charge_periods": ["off_peak"],
                    "discharge_periods": ["peak"],
                    "target_usable_fraction": 0.5,
                },
            },
            {"id": "dc", "n_modules": 6, "annual_consumption_kwh": 3500, "battery_kwh": 3, "battery_temperature": 21},
            {"id": "multi", **multiple},
            {"id": "multi-ac", **deepcopy(multiple), **deepcopy(APP_AC)},
            {"id": "load", "annual_consumption_kwh": 3000, "tariff": deepcopy(TOU)},
        ],
        "assets": [
            {
                "id": "roof",
                **{k: deepcopy(v) for k, v in multiple.items() if k not in ("annual_consumption_kwh", "battery_kwh")},
                "ownership": "external",
                "costs": {"electricity_cost": 0.2, "daily_power_cost": 0},
            }
        ],
    }


def assert_bits(actual, expected):
    np.testing.assert_array_equal(
        np.asarray(actual, dtype=float).view(np.uint64), np.asarray(expected, dtype=float).view(np.uint64)
    )


def simulate_app(app, monkeypatch):
    frames = []
    actual = projection_module._run_frame_year

    def record(*args, **kwargs):
        outcome = actual(*args, **kwargs)
        frames.append(outcome.results_df)
        return outcome

    with monkeypatch.context() as context:
        context.setattr(projection_module, "_run_frame_year", record)
        app.simulate()
    return frames


def assert_same_projection(actual, expected, year_frames=None):
    pd.testing.assert_frame_equal(actual.first_year_results_df, expected.first_year_results_df, check_exact=True)
    columns = actual.first_year_results_df.select_dtypes(include=[np.number, bool]).columns
    assert_bits(actual.first_year_results_df[columns].to_numpy(), expected.first_year_results_df[columns].to_numpy())
    pd.testing.assert_frame_equal(actual.yearly_df, expected.yearly_df, check_exact=True)
    expected_flows = expected.priced_flows
    if expected_flows is None:
        assert year_frames is not None, "capture App's years when its controller disables priced-flow retention"
        expected_flows = [{key: frame[key].to_numpy() for key in actual.priced_flows[0]} for frame in year_frames]
    for left, right in zip(actual.priced_flows, expected_flows):
        assert left.keys() == right.keys()
        for column in left:
            assert_bits(left[column], right[column])
    np.testing.assert_equal(actual.carry.degradation_state, expected.carry.degradation_state)
    assert actual.carry.energy_wh == expected.carry.energy_wh
    assert actual.carry.pv_origin_energy_wh == expected.carry.pv_origin_energy_wh
    assert actual.carry.grid_origin_energy_wh == expected.carry.grid_origin_energy_wh
    assert actual.carry.battery_retired == expected.carry.battery_retired
    assert actual.end_of_life_events == expected.end_of_life_events


def test_q2_standalone_identity_closure_and_asset_physics(community_case, offline_deps, monkeypatch):
    parsed = parse_community_config(community_case)
    run = run_community(parsed, deps=offline_deps, retain_site_frames=True, retain_settlement_frames=True)
    assert "Community" in breos.__all__
    assert tuple(run.sites) == ("ac", "dc", "load", "multi", "multi-ac", "roof")
    assert tuple(run.settlements) == (1, 2)
    assert run.provenance["settlement_schema_version"] == "1.0"
    for participant in parsed.participants:
        site_run = run.sites[participant.id]
        if isinstance(participant.site, HouseholdConfig):
            cfg = participant.site.cfg
            expected = input_module.load_consumption_profile(dict(cfg), offline_deps, participant.site.timezone)
            assert site_run.projection is None
            assert_bits(site_run.load_w, expected.iloc[:, 0])
            for meter in site_run.meters:
                assert_bits(meter.import_wh, expected.iloc[:, 0].to_numpy() * 0.25)
                assert not np.any(meter.export_wh)
            continue
        # Build App from its original payload, not the materialised config
        # (which has derived ratings that are refused as repeated sizing).
        original = next(p for p in community_case["participants"] if p["id"] == participant.id)
        app = App({**parsed.community.shared.app_payload(), **{k: v for k, v in original.items() if k != "id"}})
        year_frames = simulate_app(app, monkeypatch)
        assert_same_projection(site_run.projection, app._artifacts.projection, year_frames)
        if participant.id in ("multi", "multi-ac"):
            assert app._artifacts.pv_loss_waterfall["inverter"]["ac_cable"]["loss_kwh"] > 0
        battery = (
            build_battery_config(participant.site.cfg, participant.site, initial_soh=100)
            if participant.site.cfg["battery_kwh"] > 0
            else build_pv_only_battery_config(participant.site.cfg, participant.site)
        )
        assert_energy_conservation(site_run.projection.first_year_results_df, battery)
        for year_idx, meter in enumerate(site_run.meters):
            powers = year_frames[year_idx]
            assert_bits(meter.import_wh, powers["Import_From_Grid"] * 0.25)
            assert_bits(meter.export_wh, powers["PV_AC_Export"] * 0.25)
    roof = run.sites["roof"]
    assert not np.any(roof.load_w)
    assert np.any(roof.meters[0].import_wh > 0), "actual inverter night import is retained"
    assert np.any(roof.meters[0].export_wh > 0)
    assert not np.any((roof.meters[0].import_wh > 0) & (roof.meters[0].export_wh > 0))
    assert_energy_conservation(
        roof.projection.first_year_results_df, build_pv_only_battery_config(roof.config.cfg, roof.config)
    )
    # Independent standalone PV-only projection with the same real zero load.
    inputs = input_module.prepare_simulation_inputs(
        roof.config.cfg,
        roof.config,
        replace(offline_deps, load_profile=lambda **_kwargs: pd.DataFrame({"Load": 0.0}, index=roof.index)),
    )

    def roof_year(year_idx):
        factor = (1 - roof.config.cfg["pv_degradation_rate"]) ** year_idx
        return projection_module.ProjectionYear(
            factor,
            pv_dc=inputs.dc_system_base * factor,
            houseload=inputs.load_data,
            temperature_series=inputs.temperature_series,
            inverter_ac_derate=inputs.inverter_ac_derate,
            inverter_curve_row=inputs.inverter_curve_row,
            inverter_units=inputs.inverter_units,
        )

    alone_roof = projection_module.run_projection(
        roof.config.cfg,
        roof.config,
        2,
        roof_year,
        has_battery=False,
        execution_backend="python",
        record_priced_flows=True,
    )
    assert_same_projection(roof.projection, alone_roof)
    frames = roof.projection.first_year_results_df
    assert np.any(frames["Inverter_Night_Consumption"] > 0)
    assert np.any((frames["Inverter_Night_Consumption"] > 0) & (frames["PV_AC_Export"] > 0)), (
        "M9 daytime netting is exercised"
    )
    assert frames["PV_DC_Curtailed"].sum() > 0, "clipping is exercised"
    assert roof.config.ac_cable.loss_coefficient_per_w > 0
    ac = run.sites["ac"].projection.first_year_results_df
    assert ac["Grid_AC_To_Battery"].sum() > 0
    assert ((ac["Battery_AC_To_Load"] > 0) & (ac["PV_DC"] == 0) & (ac["Import_From_Grid"] > 0)).any()
    for year_idx, settlement in run.settlements.items():
        assert settlement.attrs["project_year"] == year_idx
        assert_settlement_closure(settlement, atol=1e-6)
        for site_id, site in run.sites.items():
            rows = settlement.xs(site_id, level="site_id")
            assert_bits(rows["Metered_Import"], site.meters[year_idx - 1].import_wh)
            assert_bits(rows["Metered_Export"], site.meters[year_idx - 1].export_wh)
        rows = settlement.xs("roof", level="site_id")
        assert_bits(rows["Supplier_Import"], roof.meters[year_idx - 1].import_wh)


@pytest.mark.parametrize("kind", ["pv", "load"])
def test_q3_single_site_no_sharing_is_exactly_alone(kind, offline_deps):
    participant = {"id": "alone", "annual_consumption_kwh": 3500, "tariff": deepcopy(TOU)}
    if kind == "pv":
        participant.update(n_modules=8, battery_kwh=3)
    raw = {
        "community": {
            "location": "porto",
            "start_date": "2026-01-01",
            "resolution": "15min",
            "projection_years": 2,
            "allocation": "proportional",
            "pool_participant_export": False,
            "shared_energy_price": 0,
            "shared_network_charge": 0,
            "surplus_price": 0,
        },
        "participants": [participant],
    }
    parsed = parse_community_config(raw)
    run = run_community(parsed, deps=offline_deps, retain_site_frames=True, retain_settlement_frames=True)
    for year_idx, frame in run.settlements.items():
        rows = frame.xs("participant", level="site_type")
        assert_bits(rows["Supplier_Import"], run.sites["alone"].meters[year_idx - 1].import_wh)
        assert_bits(rows["Own_Export"], run.sites["alone"].meters[year_idx - 1].export_wh)
        assert not np.any(
            rows[
                [
                    "Shared_Allocated",
                    "Shared_Used",
                    "Participant_Surplus",
                    "Attributed_Shared_Used",
                    "Attributed_Surplus",
                ]
            ]
        )
    if kind == "pv":
        app = App({**parsed.community.shared.app_payload(), **{k: v for k, v in participant.items() if k != "id"}})
        app.simulate()
        assert_same_projection(run.sites["alone"].projection, app._artifacts.projection)
    else:
        alone = input_module.load_consumption_profile(
            dict(parsed.participants[0].site.cfg), offline_deps, "Europe/Lisbon"
        )
        assert_bits(run.sites["alone"].meters[0].import_wh, alone.iloc[:, 0].to_numpy() * 0.25)


@pytest.mark.parametrize("allocation", ["fixed", "proportional"])
def test_q4_reordering_preserves_all_bits_and_hashes(community_case, offline_deps, allocation):
    community_case["community"]["projection_years"] = 1
    community_case["community"]["allocation"] = allocation
    if allocation == "fixed":
        for participant in community_case["participants"]:
            participant["coefficient"] = 1 / len(community_case["participants"])
    first = run_community(parse_community_config(community_case), deps=offline_deps)
    community_case["participants"].reverse()
    second = run_community(parse_community_config(community_case), deps=offline_deps)
    assert first.provenance == second.provenance
    assert tuple(first.sites) == tuple(second.sites)
    for site_id in first.sites:
        assert first.sites[site_id].projection is None or first.sites[site_id].projection.first_year_results_df is None
        assert_bits(first.sites[site_id].meters[0].import_wh, second.sites[site_id].meters[0].import_wh)
        assert_bits(first.sites[site_id].meters[0].export_wh, second.sites[site_id].meters[0].export_wh)
    assert_bits(first.settlements[1], second.settlements[1])
    assert_settlement_closure(first.settlements[1], coefficients=first.config.community.coefficients, atol=1e-6)


def test_python_numba_parity_for_the_mixed_community(community_case, offline_deps):
    pytest.importorskip("numba")
    python = run_community(
        parse_community_config(community_case),
        deps=offline_deps,
        retain_site_frames=True,
        retain_settlement_frames=True,
    )
    community_case["community"]["execution_backend"] = "numba"
    numba = run_community(
        parse_community_config(community_case),
        deps=offline_deps,
        retain_site_frames=True,
        retain_settlement_frames=True,
    )
    for site_id, left in python.sites.items():
        right = numba.sites[site_id]
        if left.projection is not None:
            assert_same_projection(left.projection, right.projection)
        for a, b in zip(left.meters, right.meters):
            assert_bits(a.import_wh, b.import_wh)
            assert_bits(a.export_wh, b.export_wh)
    for year_idx in python.settlements:
        pd.testing.assert_frame_equal(python.settlements[year_idx], numba.settlements[year_idx], check_exact=True)
        assert_bits(python.settlements[year_idx], numba.settlements[year_idx])


def test_input_reuse_groups_whole_keys_and_never_forces_hits(offline_deps, monkeypatch):
    base = {"n_modules": 8, "annual_consumption_kwh": 3500}
    raw = {
        "community": {
            "location": "porto",
            "start_date": "2026-01-01",
            "resolution": "15min",
            "projection_years": 1,
            "allocation": "proportional",
            "shared_energy_price": 0,
            "shared_network_charge": 0,
            "surplus_price": 0,
        },
        "participants": [
            {"id": "a", **base},
            {"id": "b", **base, "annual_consumption_kwh": 4000},
            {"id": "c", **base, "battery_kwh": 3},
            {"id": "d", **base, "battery_temperature": 21},
        ],
    }
    prepared = []
    actual = input_module.prepare_simulation_inputs

    def record(cfg, *args):
        prepared.append((cfg["annual_consumption_kwh"], cfg["battery_temperature"]))
        return actual(cfg, *args)

    monkeypatch.setattr(input_module, "prepare_simulation_inputs", record)
    run = run_community(parse_community_config(raw), deps=offline_deps)
    assert len(prepared) == 3  # a/c share; b's load and d's temperature differ.
    assert set(prepared) == {(3500, "weather"), (4000, "weather"), (3500, 21)}
    assert not np.array_equal(run.sites["a"].load_w, run.sites["b"].load_w)
    assert not np.shares_memory(run.sites["a"].meters[0].import_wh, run.sites["c"].meters[0].import_wh)


def test_full_horizon_owns_replacement_clock_and_finalisation(offline_deps, monkeypatch):
    raw = {
        "community": {
            "location": "porto",
            "start_date": "2026-01-01",
            "resolution": "15min",
            "projection_years": 2,
            "allocation": "proportional",
            "pool_participant_export": False,
            "shared_energy_price": 0,
            "shared_network_charge": 0,
            "surplus_price": 0,
        },
        "participants": [
            {
                "id": "battery",
                "n_modules": 8,
                "annual_consumption_kwh": 3500,
                "battery_kwh": 3,
                "battery_eol_percentage": 0.99999,
                "battery_allow_terminal_replacement": False,
            }
        ],
    }
    calls = []
    actual = projection_module.project_years

    def record(years, *args, **kwargs):
        calls.append((years, kwargs["record_priced_flows"]))
        return actual(years, *args, **kwargs)

    monkeypatch.setattr(projection_module, "project_years", record)
    parsed = parse_community_config(raw)
    run = run_community(parsed, deps=offline_deps, retain_site_frames=True, retain_settlement_frames=True)
    assert calls == [(2, True)]
    app = App(
        {**parsed.community.shared.app_payload(), **{k: v for k, v in raw["participants"][0].items() if k != "id"}}
    )
    app.simulate()
    projection = run.sites["battery"].projection
    assert_same_projection(projection, app._artifacts.projection)
    assert any(event["year"] == 2 and event["time_years"] > 1 for event in projection.end_of_life_events)
    assert not projection.end_of_life_events[-1]["action"] == "replaced"


def test_daily_planner_keeps_all_years_meter_flows(offline_deps):
    pytest.importorskip("numba")
    participant = {
        "id": "planner",
        "n_modules": 8,
        "annual_consumption_kwh": 4000,
        "battery_kwh": 5,
        "tariff": deepcopy(TOU),
        "smart_charging": {
            "mode": "daily_persistence",
            "charge_periods": ["off_peak"],
            "discharge_periods": ["peak"],
            "grid_charge_efficiency": 0.95,
            "target_levels": 3,
            "soc_states": 3,
        },
    }
    raw = {
        "community": {
            "location": "porto",
            "start_date": "2026-01-01",
            "resolution": "15min",
            "projection_years": 2,
            "execution_backend": "numba",
            "allocation": "proportional",
            "pool_participant_export": False,
            "shared_energy_price": 0,
            "shared_network_charge": 0,
            "surplus_price": 0,
        },
        "participants": [participant],
    }
    parsed = parse_community_config(raw)
    run = run_community(parsed, deps=offline_deps, retain_site_frames=True, retain_settlement_frames=True)
    projection = run.sites["planner"].projection
    app = App({**parsed.community.shared.app_payload(), **{k: v for k, v in participant.items() if k != "id"}})
    app.simulate()
    assert app._artifacts.projection.priced_flows is None  # App's price-aware retention differs.
    pd.testing.assert_frame_equal(projection.first_year_results_df, app.timeseries(), check_exact=True)
    pd.testing.assert_frame_equal(projection.yearly_df, app._artifacts.projection.yearly_df, check_exact=True)
    assert len(projection.priced_flows) == 2
    assert projection.carry.controller_carry is not None
    for year_idx in (1, 2):
        assert_bits(
            run.settlements[year_idx].xs("participant", level="site_type")["Supplier_Import"],
            projection.priced_flows[year_idx - 1]["Import_From_Grid"] * 0.25,
        )


@pytest.mark.parametrize("year", [2024, 2026])
@pytest.mark.parametrize("clock", ["UTC", timezone(timedelta(hours=1))])
def test_weather_clock_and_leap_year_match_app_without_cropping(offline_deps, monkeypatch, year, clock):
    weather = input_module.remap_tmy_year(_build_synthetic_weather(year=2026, freq="15min"), year)
    weather.index = weather.index.tz_localize(None).tz_localize(clock)

    def fetch(**_kwargs):
        return weather.copy(deep=True), {}

    monkeypatch.setattr("breos.app.fetch_tmy_weather_data", fetch)
    deps = replace(offline_deps, fetch_tmy_weather_data=fetch)
    raw = {
        "community": {
            "location": "berlin",
            "start_date": f"{year}-01-01",
            "resolution": "15min",
            "projection_years": 2,
            "allocation": "proportional",
            "shared_energy_price": 0,
            "shared_network_charge": 0,
            "surplus_price": 0,
        },
        "participants": [
            {"id": "pv", "n_modules": 8, "annual_consumption_kwh": 3500},
            {"id": "load", "annual_consumption_kwh": 3500},
        ],
    }
    parsed = parse_community_config(raw)
    run = run_community(parsed, deps=deps, retain_site_frames=True, retain_settlement_frames=True)
    app = App({**parsed.community.shared.app_payload(), "n_modules": 8, "annual_consumption_kwh": 3500})
    app.simulate()
    assert_same_projection(run.sites["pv"].projection, app._artifacts.projection)
    assert_bits(run.sites["load"].load_w, app.timeseries()["Houseload"])
    expected_steps = (366 if year == 2024 else 365) * 96
    for number, frame in run.settlements.items():
        intervals = frame.index.get_level_values("interval").unique()
        assert len(intervals) == expected_steps
        assert intervals[0] == weather.index[0]
        assert intervals[-1] == weather.index[-1]
        assert frame.attrs["project_year"] == number
        assert_settlement_closure(frame, atol=1e-6)


@pytest.mark.parametrize("bad", ["missing", "negative"])
def test_household_input_errors_name_site(offline_deps, bad):
    load = offline_deps.load_profile

    def corrupt(**kwargs):
        frame = load(**kwargs)
        if bad == "missing":
            return frame.iloc[:-1]
        frame.iloc[0, 0] = -1
        return frame

    raw = {
        "community": {
            "location": "porto",
            "start_date": "2026-01-01",
            "resolution": "15min",
            "projection_years": 1,
            "allocation": "proportional",
            "shared_energy_price": 0,
            "shared_network_charge": 0,
            "surplus_price": 0,
        },
        "participants": [{"id": "bad-load", "annual_consumption_kwh": 3500}],
    }
    with pytest.raises(ValueError, match="bad-load.*" + ("calendar" if bad == "missing" else "nonnegative")):
        run_community(parse_community_config(raw), deps=replace(offline_deps, load_profile=corrupt))
