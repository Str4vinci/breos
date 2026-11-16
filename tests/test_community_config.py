"""ACC configuration contracts without dispatch, money calculations or network."""

import tomllib
import warnings
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields
from itertools import permutations

import pytest

import breos
from breos.app_config import ALLOWED_CONFIG_KEYS, DEFAULTS, ResolvedAppConfig, resolve_app_config
from breos.community.config import SHARED_APP_KEYS, HouseholdConfig, SharedSettings, parse_community_config
from breos.execution import resolve_execution_backend
from tests.offline_app import CREDIT, TOU

EXAMPLE = """
[community]
location = "porto"
start_date = "2026-01-01"
resolution = "15min"
projection_years = 20
currency = "EUR"
allocation = "fixed"
pool_participant_export = true
shared_energy_price = 0.10
shared_network_charge = 0.02
surplus_price = 0.04

[[participants]]
id = "apt-1"
coefficient = 0.5
annual_consumption_kwh = 3500
load_profile = "demandlib_h0"
tariff = { schedule = "pt_mainland_2026_daily_bi", currency = "EUR", import_prices = { peak = 0.28, off_peak = 0.11 }, export_prices = { all = 0.05 } }

[[participants]]
id = "apt_1"
coefficient = 0.5
n_modules = 8
annual_consumption_kwh = 4000
costs = { module_cost_per_w = 0.2 }

[[assets]]
id = "school-roof"
n_modules = 60
ownership = { apt-1 = 0.5, apt_1 = 0.5 }
costs = { electricity_cost = 0.20, daily_power_cost = 0.0 }

[[assets]]
id = "external-roof"
n_modules = 10
ownership = "external"
costs = { electricity_cost = 0.0, daily_power_cost = 0.0 }
"""


@pytest.fixture
def config():
    return tomllib.loads(EXAMPLE)


def test_valid_community_and_standalone_app_equivalence(config):
    original = deepcopy(config)
    defaults, allowed = deepcopy(DEFAULTS), ALLOWED_CONFIG_KEYS.copy()
    parsed = parse_community_config(config)
    assert config == original
    assert DEFAULTS == defaults and ALLOWED_CONFIG_KEYS == allowed
    assert "Community" in breos.__all__
    assert [p.id for p in parsed.participants] == ["apt-1", "apt_1"]
    assert [a.id for a in parsed.assets] == ["external-roof", "school-roof"]
    assert parsed.community.coefficients == {"apt-1": 0.5, "apt_1": 0.5}
    assert parsed.community.shared_energy_price == {"all": 0.10}
    assert parsed.community.shared_network_charge == {"all": 0.02}
    assert parsed.community.surplus_price == {"all": 0.04}
    assert parsed.community.shared.discount_rate == DEFAULTS["discount_rate"]
    household, pv = parsed.participants
    assert isinstance(household.site, HouseholdConfig)
    assert "n_modules" not in household.site.cfg and "pv_arrays" not in household.site.cfg
    assert household.site.cfg["annual_consumption_kwh"] == 3500
    assert household.site.tariff.prices.import_prices == {"peak": 0.28, "off_peak": 0.11}
    assert isinstance(pv.site, ResolvedAppConfig)
    payload = {k: v for k, v in original["participants"][1].items() if k not in ("id", "coefficient")}
    assert pv.site == resolve_app_config({**parsed.community.shared.app_payload(), **payload})
    roof = parsed.assets[1]
    assert roof.ownership == {"apt-1": 0.5, "apt_1": 0.5}
    assert roof.site.cfg["annual_consumption_kwh"] == 0
    assert roof.site.cfg["load_profile"] is None
    assert roof.site.cfg["battery_kwh"] == 0
    assert roof.site.cost_params.electricity_cost == 0.20
    assert parsed.assets[0].ownership == "external"
    assert parsed.assets[0].site.cost_params.daily_power_cost == 0
    # Changing source tables after parsing cannot change the parsed config.
    config["assets"][0]["ownership"]["apt-1"] = 1
    config["participants"][1]["costs"]["module_cost_per_w"] = 999
    assert roof.ownership["apt-1"] == 0.5
    assert pv.site.cfg["costs"]["module_cost_per_w"] == 0.2


def test_frozen_records_and_identity_maps(config):
    parsed = parse_community_config(config)
    for record in (
        parsed,
        parsed.community,
        parsed.community.shared,
        *parsed.participants,
        *parsed.assets,
        parsed.participants[0].site,
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(record, fields(record)[0].name, None)
    for mapping in (
        parsed.community.coefficients,
        parsed.assets[1].ownership,
        parsed.community.shared_energy_price,
        parsed.participants[0].site.cfg,
    ):
        with pytest.raises(TypeError):
            mapping["apt-1"] = 1


@pytest.mark.parametrize("asset_ids", [(), ("school-roof",), ("external-roof",), ("school-roof", "external-roof")])
def test_producer_credit_warns_once_per_asset_and_never_for_participants(config, asset_ids, recwarn):
    warnings.simplefilter("always")
    for participant in config["participants"]:
        participant["tariff"] = {**deepcopy(TOU), "annual_network_credit": deepcopy(CREDIT)}
    for asset in config["assets"]:
        asset.pop("costs")
        asset["tariff"] = deepcopy(TOU)
        if asset["id"] in asset_ids:
            asset["tariff"]["annual_network_credit"] = deepcopy(CREDIT)
    parsed = parse_community_config(config)
    assert len(recwarn) == len(asset_ids)
    for warning, asset_id in zip(recwarn, sorted(asset_ids), strict=True):
        assert warning.category is UserWarning
        assert str(warning.message) == (
            f"asset '{asset_id}': a producer-only connection rarely qualifies for an annual network credit; "
            "BREOS does not judge eligibility."
        )
    for asset in parsed.assets:
        credit = asset.site.tariff.prices.annual_network_credit
        assert (credit is not None) == (asset.id in asset_ids)
        if credit is not None:
            assert credit.record()["amount_per_year"] == CREDIT["amount_per_year"]


def test_reordering_and_hyphen_normalisation_preserve_ids(config):
    expected = parse_community_config(config)
    for participant_order in permutations(config["participants"]):
        for asset_order in permutations(config["assets"]):
            reordered = deepcopy(config)
            reordered["participants"] = list(reversed(deepcopy(participant_order)))
            reordered["assets"] = list(deepcopy(asset_order))
            for row in reordered["assets"]:
                if isinstance(row["ownership"], dict):
                    row["ownership"] = dict(reversed(list(row["ownership"].items())))
            assert parse_community_config(reordered) == expected
    hyphenated = deepcopy(config)
    hyphenated["community"] = {key.replace("_", "-"): value for key, value in config["community"].items()}
    for row in hyphenated["participants"] + hyphenated["assets"]:
        if "costs" in row:
            row["costs"] = {key.replace("_", "-"): value for key, value in row["costs"].items()}
        row.update({key.replace("_", "-"): row.pop(key) for key in list(row) if "_" in key})
    assert parse_community_config(hyphenated) == expected


def test_proportional_zero_agreements_and_no_assets(config):
    config["community"].update(
        allocation="proportional",
        pool_participant_export=False,
        shared_energy_price=0,
        shared_network_charge=0,
        surplus_price=0,
    )
    for row in config["participants"]:
        row.pop("coefficient")
    config.pop("assets")
    parsed = parse_community_config(config)
    assert parsed.community.coefficients is None
    assert not parsed.community.pool_participant_export
    assert parsed.community.shared_energy_price == {"all": 0.0}
    assert parsed.assets == ()


@pytest.mark.parametrize("site_type,index", [("participants", 0), ("participants", 1), ("assets", 0)])
@pytest.mark.parametrize("key", sorted(SHARED_APP_KEYS))
def test_shared_keys_refused_even_if_values_agree(config, site_type, index, key):
    assert {f.name for f in fields(SharedSettings)} == SHARED_APP_KEYS
    site = config[site_type][index]
    site[key.replace("_", "-")] = config["community"].get(key, DEFAULTS.get(key))
    with pytest.raises(ValueError, match=rf"{site['id']}.*shared key.*{key}.*\[community\]"):
        parse_community_config(config)


@pytest.mark.parametrize("where", ["participants", "assets"])
@pytest.mark.parametrize("key", ["vehicle", "period", "montecarlo", "sweep"])
def test_unsupported_site_features(config, where, key):
    site = config[where][0]
    site[key] = {}
    with pytest.raises(ValueError, match=rf"{site['id']}.*{key}.*not supported"):
        parse_community_config(config)


@pytest.mark.parametrize(
    "key",
    [
        "battery",
        "battery_kwh",
        "battery_coupling",
        "degradation_engine",
        "annual_consumption_kwh",
        "load_profile",
        "load_profile_file",
        "load",
        "smart_charging",
        "reference_tariff",
    ],
)
def test_assets_refuse_load_battery_and_controllers(config, key):
    config["assets"][0][key] = 0
    with pytest.raises(ValueError, match=rf"school-roof.*PV-only assets refuse.*{key}"):
        parse_community_config(config)


@pytest.mark.parametrize("key", ["electricity_cost", "daily_power_cost"])
def test_assets_need_explicit_prices_not_preset_defaults(config, key):
    config["assets"][0]["costs"].pop(key)
    config["assets"][0]["cost_preset"] = "residential_pt"
    with pytest.raises(ValueError, match="school-roof.*explicit costs.electricity_cost and costs.daily_power_cost"):
        parse_community_config(config)


def test_asset_tariff_and_required_fixed_charge(config):
    row = config["assets"][0]
    row.pop("costs")
    row["tariff"] = deepcopy(config["participants"][0]["tariff"])
    with pytest.raises(ValueError, match="school-roof.*explicit import_prices and fixed_charge_per_day"):
        parse_community_config(config)
    row["tariff"]["fixed_charge_per_day"] = 0
    assert parse_community_config(config).assets[1].site.tariff.prices.fixed_charge_per_day == 0
    row["tariff"].pop("import_prices")
    with pytest.raises(ValueError, match="school-roof.*explicit import_prices"):
        parse_community_config(config)


@pytest.mark.parametrize("id", [None, "", "Apt-1", "apt.1", " apt-1", "_apt", "a" * 65, 12])
def test_invalid_ids(config, id):
    config["participants"][0]["id"] = id
    with pytest.raises(ValueError, match=r"participants\[0\].id.*must match"):
        parse_community_config(config)


@pytest.mark.parametrize("where", ["participants", "assets"])
def test_missing_and_duplicate_ids(config, where):
    config[where][0].pop("id")
    with pytest.raises(ValueError, match=r".id.*must match"):
        parse_community_config(config)
    config[where][0]["id"] = config["participants"][1]["id"]
    with pytest.raises(ValueError, match="apt_1.*unique across"):
        parse_community_config(config)


@pytest.mark.parametrize(
    "value,match",
    [
        (None, "fixed allocation needs"),
        (-0.5, "coefficient.*>= 0"),
        (float("nan"), "coefficient.*finite"),
        (True, "coefficient.*finite"),
        (0.4, "coefficients .*sum to 1"),
    ],
)
def test_bad_coefficients(config, value, match):
    if value is None:
        config["participants"][0].pop("coefficient")
    else:
        config["participants"][0]["coefficient"] = value
    with pytest.raises((TypeError, ValueError), match=match):
        parse_community_config(config)


def test_proportional_refuses_fixed_coefficient(config):
    config["community"]["allocation"] = "proportional"
    with pytest.raises(ValueError, match="apt-1.*proportional allocation does not take coefficient"):
        parse_community_config(config)


@pytest.mark.parametrize(
    "ownership,match",
    [
        (None, "ownership must"),
        ({}, "ownership must"),
        ("someone", "ownership must"),
        ({"unknown": 1}, "existing participant ids"),
        ({"external-roof": 1}, "existing participant ids"),
        ({"apt-1": -1, "apt_1": 2}, ">= 0"),
        ({"apt-1": 0.8}, "sum to 1"),
        ({"apt-1": float("inf")}, "finite"),
        ({"apt-1": True}, "finite"),
    ],
)
def test_invalid_ownership(config, ownership, match):
    config["assets"][0]["ownership"] = ownership
    with pytest.raises((TypeError, ValueError), match=rf"school-roof.*{match}"):
        parse_community_config(config)


def test_missing_ownership_and_verbatim_owner_lookup(config):
    config["assets"][0].pop("ownership")
    with pytest.raises(ValueError, match="school-roof.*ownership must"):
        parse_community_config(config)
    config["assets"][0]["ownership"] = {"apt-1": 1}
    config["participants"][0]["id"] = "changed-id"
    with pytest.raises(ValueError, match="school-roof.*ids verbatim"):
        parse_community_config(config)


@pytest.mark.parametrize("key", ["shared_energy_price", "shared_network_charge", "surplus_price"])
@pytest.mark.parametrize("value", [None, -0.1, float("inf"), float("nan"), True, {}, {"peak": 0.1}])
def test_bad_or_missing_agreements(config, key, value):
    if value is None:
        config["community"].pop(key)
    else:
        config["community"][key] = value
    with pytest.raises((TypeError, ValueError), match=rf"community.*{key}"):
        parse_community_config(config)


def test_scheduled_agreements_and_period_validation(config):
    community = config["community"]
    community["schedule"] = "pt_mainland_2026_daily_bi"
    community["shared_energy_price"] = {"peak": 0.10, "off_peak": 0.05}
    parsed = parse_community_config(config)
    assert parsed.community.schedule.schedule.identifier == community["schedule"]
    assert parsed.community.shared_energy_price == {"off_peak": 0.05, "peak": 0.10}
    for prices in ({"peak": 0.1}, {"all": 0, "unknown": 1}, {"peak": -1, "off_peak": 0}):
        community["shared_energy_price"] = prices
        with pytest.raises(ValueError, match="community.shared_energy_price"):
            parse_community_config(config)
    community["shared_energy_price"] = 0
    community["location"] = "berlin"
    with pytest.raises(ValueError, match="timezone"):
        parse_community_config(config)


def test_seasonal_agreements_use_app_schedule_validation(config):
    from tests.test_tariff_month_seasons import SEASON_IMPORT, _quarterly

    community = config["community"]
    community["custom_schedule"] = _quarterly(timezone="Europe/Lisbon")
    community["shared_energy_price"] = deepcopy(SEASON_IMPORT)
    parsed = parse_community_config(config)
    assert parsed.community.shared_energy_price == SEASON_IMPORT
    assert parsed.community.schedule.season_names == ("q1", "q2", "q3", "q4")
    community["shared_energy_price"].pop("q4")
    with pytest.raises(ValueError, match="community.shared_energy_price.*q4"):
        parse_community_config(config)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"participants": []}, "at least one participant"),
        ({"participants": {}}, "participants.*array of site tables"),
        ({"assets": {}}, "assets.*array of site tables"),
        ({"participants": [1]}, "participants.*table/mapping"),
        ({"unknown": 1}, "config.*unknown key"),
    ],
)
def test_invalid_config_structure(config, changes, match):
    config.update(changes)
    with pytest.raises((TypeError, ValueError), match=match):
        parse_community_config(config)


def test_load_only_community_with_no_assets_and_no_fake_pv(config):
    config["participants"] = [config["participants"][0]]
    config["participants"][0]["coefficient"] = 1
    config["assets"] = []
    parsed = parse_community_config(config)
    assert isinstance(parsed.participants[0].site, HouseholdConfig)
    assert parsed.assets == ()
    config["participants"][0].pop("annual_consumption_kwh")
    with pytest.raises(ValueError, match="apt-1.*needs annual_consumption_kwh"):
        parse_community_config(config)


@pytest.mark.parametrize("site_type,index", [("participants", 0), ("participants", 1), ("assets", 0)])
def test_site_tariff_currency_matches_community(config, site_type, index):
    row = config[site_type][index]
    row["tariff"] = deepcopy(config["participants"][0]["tariff"])
    row["tariff"].update(currency="USD", fixed_charge_per_day=0)
    row.pop("costs", None)
    with pytest.raises(ValueError, match=rf"{row['id']}.*currenc"):
        parse_community_config(config)


def test_shared_economics_and_custom_location_are_injected_into_sites(config):
    config["community"].update(
        location={"latitude": 41, "longitude": -8, "timezone": "Europe/Lisbon"},
        discount_rate=0.04,
        inflation_rate=0.03,
        import_price_escalation=0.02,
        om_escalation=0.01,
        replacement_cost_learning=0.05,
        sell_price_inflation=0.015,
    )
    parsed = parse_community_config(config)
    for site in [*(p.site for p in parsed.participants), *(a.site for a in parsed.assets)]:
        for key in SHARED_APP_KEYS - {"execution_backend"}:
            assert site.cfg[key] == getattr(parsed.community.shared, key)
        # The shared layer keeps the request; each PV site runs it resolved.
        requested = parsed.community.shared.execution_backend
        assert requested == "auto"
        if isinstance(site, ResolvedAppConfig):
            assert site.requested_execution_backend == requested
            assert site.cfg["execution_backend"] == resolve_execution_backend(requested)
        else:
            assert site.cfg["execution_backend"] == requested
    layer = parsed.community.shared.app_payload()
    layer["location"]["latitude"] = 0
    assert parsed.community.shared.location["latitude"] == 41


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"resolution": "h"}, "resolution.*15min"),
        ({"resolution": "30min"}, "resolution.*15min"),
        ({"allocation": "dynamic"}, "allocation.*fixed.*proportional"),
        ({"pool_participant_export": "false"}, "pool_participant_export.*true or false"),
        ({"discount_rate": -1}, "discount_rate"),
        ({"inflation_rate": float("nan")}, "inflation_rate"),
        ({"import_price_escalation": -1}, "import_price_escalation"),
        ({"replacement_cost_learning": 1}, "replacement_cost_learning"),
        ({"projection_years": 0}, "projection_years"),
        ({"start_date": "2026-02-01"}, "start_date.*1 January"),
        ({"currency": "bad"}, "currency"),
        ({"execution_backend": "bad"}, "execution_backend"),
        ({"weather_source": "a/b"}, "weather_source"),
        ({"location": {"latitude": 100, "longitude": 1, "timezone": "UTC"}}, "latitude"),
        ({"location": {"latitude": 1, "longitude": 1, "timezone": "bad"}}, "timezone"),
        ({"typo": 1}, "unknown key.*typo"),
    ],
)
def test_bad_community_fields(config, changes, match):
    config["community"].update(changes)
    with pytest.raises((TypeError, ValueError), match=match):
        parse_community_config(config)


@pytest.mark.parametrize("field", ["community", "participants"])
def test_missing_required_sections(config, field):
    config.pop(field)
    with pytest.raises((TypeError, ValueError), match=field):
        parse_community_config(config)


@pytest.mark.parametrize("key", ["location", "resolution", "allocation"])
def test_missing_community_fields(config, key):
    config["community"].pop(key)
    with pytest.raises(ValueError, match=key):
        parse_community_config(config)


@pytest.mark.parametrize("key", ["battery_kwh", "smart_charging", "pv_module"])
def test_load_only_households_refuse_equipment(config, key):
    config["participants"][0][key] = 1
    with pytest.raises(ValueError, match=rf"apt-1.*load-only household.*{key}"):
        parse_community_config(config)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"annual_consumption_kwh": 0}, "annual_consumption_kwh"),
        ({"load_profile": "missing"}, "profile"),
        ({"load_profile": "custom", "load_profile_unit": "Wh"}, "load_profile_file"),
        ({"typo": 1}, "unknown key"),
        ({"tariff": {"currency": "USD"}}, "tariff"),
    ],
)
def test_household_uses_app_load_and_tariff_validation(config, changes, match):
    config["participants"][0].update(changes)
    with pytest.raises((TypeError, ValueError), match=rf"apt-1.*{match}"):
        parse_community_config(config)


@pytest.mark.parametrize("kind", ["pv", "household"])
def test_reference_tariff_and_shared_reference_escalation(config, kind):
    row = config["participants"][1 if kind == "pv" else 0]
    row["reference_tariff"] = {"currency": "EUR", "import_prices": {"all": 0.3}, "fixed_charge_per_day": 0.4}
    parsed = parse_community_config(config)
    site = next(p.site for p in parsed.participants if p.id == row["id"])
    assert site.reference_tariff.prices.import_prices == {"all": 0.3}
    row["reference_tariff"]["import_price_escalation"] = 0.01
    with pytest.raises(ValueError, match=rf"{row['id']}.*reference_tariff.import_price_escalation is shared"):
        parse_community_config(config)


def test_app_ac_battery_checks_are_inherited(config):
    row = config["participants"][1]
    row.update(battery_kwh=5, battery_coupling="ac")
    for settings, match in [
        ({}, "inverter's rating"),
        ({"battery_inverter_ac_rating_kw": 2.5}, "battery_inverter_efficiency"),
        ({"battery_inverter_efficiency": 0.95}, "battery_inverter_cost_per_kw"),
    ]:
        row.update(settings)
        with pytest.raises(ValueError, match=rf"apt_1.*{match}"):
            parse_community_config(config)
    row["costs"]["battery_inverter_cost_per_kw"] = 0
    assert parse_community_config(config).participants[1].site.battery_inverter_ac_capacity_w == 2500


@pytest.mark.parametrize("mode", ["dc", "planner", "ac"])
def test_app_several_inverter_checks_are_inherited(config, trina, mode):
    from tests.test_ac_several_inverters import APP_AC, TOU
    from tests.test_multi_inverter import TWO_INVERTERS

    row = config["participants"][1]
    row.update({k: deepcopy(v) for k, v in TWO_INVERTERS.items() if k not in SHARED_APP_KEYS})
    row["battery_kwh"] = 5
    if mode != "dc":
        row.update(deepcopy(APP_AC))
    if mode == "planner":
        row.update(
            tariff=deepcopy(TOU),
            smart_charging={"mode": "daily_persistence", "charge_periods": ["off_peak"], "discharge_periods": ["peak"]},
        )
    if mode == "ac":
        assert len(parse_community_config(config).participants[1].site.electrical.inverters) == 2
    else:
        with pytest.raises(
            ValueError, match="apt_1.*" + ("must be AC-coupled" if mode == "dc" else "plans with one PV inverter")
        ):
            parse_community_config(config)


def test_duplicate_normalised_keys_name_the_site(config):
    config["participants"][1]["n-modules"] = 8
    with pytest.raises(ValueError, match="apt_1.*duplicate key 'n_modules'"):
        parse_community_config(config)
