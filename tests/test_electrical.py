"""Declared string topologies are checked against published datasheet limits."""

import math
from dataclasses import replace

import pytest

from breos.electrical import (
    BIFACIAL_REAR_IRRADIANCE_RATIO,
    MOUNTING_TEMPERATURE_RISE_C,
    ElectricalEnvironment,
    PVString,
    check_topology,
)
from breos.equipment import InverterSpec, MpptSpec
from breos.pv_modules import PVModuleParams

# Trina Solar TSM-450NEG9R.28, datasheet TSM_EN_2024_C, PDF page 2:
# https://static.trinasolar.com/sites/default/files/Datasheet_Vertex%20S%2B_NEG9R.28_EN_2024_C_web.pdf
TRINA_450 = PVModuleParams(
    Mpp=450,
    Vmp=44.6,
    Imp=10.09,
    Voc=52.9,
    Isc=10.74,
    T_Pmax_pct=-0.29,
    T_Voc_pct=-0.24,
    T_Isc_pct=0.04,
    N_Cells=144,
    Max_System_Voltage=1500,
)

# Jinko Solar JKM440N-54HL4R-B, datasheet EU-JKM430-455N-54HL4R-B-G3-EN, PDF page 2:
# https://jinkosolar.eu/wp-content/uploads/EU-JKM430-455N-54HL4R-B-G3-EN.pdf
# The maximum system voltage was not transcribed, so it stays unknown here.
JINKO_440 = PVModuleParams(
    Mpp=440,
    Vmp=32.99,
    Imp=13.34,
    Voc=39.57,
    Isc=13.8,
    T_Pmax_pct=-0.29,
    T_Voc_pct=-0.25,
    T_Isc_pct=0.045,
    N_Cells=108,
)

# Huawei SUN2000-4KTL-L1, European datasheet, PDF page 2:
# https://solar.huawei.com/admin/asset/v1/pro/view/fc2bdc8911d8462dbc1a88dae66314cc.pdf
# 6000 W is the recommended PV input power, not a maximum.
HUAWEI_4KTL = InverterSpec(
    nominal_power_w=4000,
    inverter_efficiency=0.984,
    is_hybrid=True,
    max_dc_voltage_v=600,
    recommended_dc_power_w=6000,
    min_mppt_voltage_v=90,
    max_mppt_voltage_v=560,
    startup_voltage_v=100,
    mppts=(MpptSpec(12.5, 18, 1),) * 2,
    isc_comparison_factor=1.0,
)

# Fronius Symo GEN24 6.0, technical data, retrieved 2026-10-11:
# https://www.fronius.com/en-gb/uk/solar-energy/installers-partners/technical-data/all-products/inverters/fronius-symo-gen24/fronius-symo-gen24-6-0
# Asymmetric inputs: 25 / 12.5 A operating, 40 / 20 A short-circuit, 2 / 1 DC
# connections. The MPPT window is the usable range, 80-800 V; the range at
# rated power, 174-800 V, is narrower. max_dc_power_w is the maximum PV
# generator power.
FRONIUS_GEN24_6 = InverterSpec(
    nominal_power_w=6000,
    inverter_efficiency=0.982,
    is_hybrid=False,
    max_dc_voltage_v=1000,
    max_dc_power_w=9000,
    min_mppt_voltage_v=80,
    max_mppt_voltage_v=800,
    startup_voltage_v=80,
    mppts=(MpptSpec(25, 40, 2), MpptSpec(12.5, 20, 1)),
    isc_comparison_factor=1.0,
)

# Fronius Verto 15.0 Plus, datasheet SE_DS_Fronius_Verto_Plus_EN (preliminary data), retrieved 2026-10-11:
# https://www.fronius.com/-/media/australia/se/downloads/se_ds_fronius_verto_plus_en.pdf
# Three trackers with two DC connections each. Every string connection is
# rated like its tracker (28 A operating, 50 A short-circuit); each tracker
# converts 13 000 W and may carry 20 000 Wp; the inverter converts 22 500 W
# and takes 150 A short-circuit over all inputs. The datasheet states no Isc
# comparison rule, so the factor stays unknown.
VERTO_15_PLUS = InverterSpec(
    nominal_power_w=15000,
    inverter_efficiency=0.9803,
    is_hybrid=True,
    max_dc_voltage_v=1000,
    max_dc_power_w=22500,
    min_mppt_voltage_v=150,
    max_mppt_voltage_v=870,
    startup_voltage_v=150,
    mppts=(
        MpptSpec(
            max_input_current_a=28,
            max_short_circuit_current_a=50,
            max_strings=2,
            max_string_input_current_a=28,
            max_string_short_circuit_current_a=50,
            max_dc_power_w=20000,
            max_usable_dc_power_w=13000,
        ),
    )
    * 3,
    max_short_circuit_current_a=150,
    max_usable_dc_power_w=22500,
)

# The Isc comparison basis was not transcribed; 1 states a direct comparison for these tests.
PORTO = ElectricalEnvironment(min_design_temp_c=-5, max_ambient_temp_c=40)


def codes(check, severity=None):
    return sorted(f.code for f in check.findings if severity is None or f.severity == severity)


def test_judgement_constants_are_pinned():
    assert MOUNTING_TEMPERATURE_RISE_C == {"flush": 30.0, "tilted_rack": 25.0, "ground_mount": 20.0}
    assert BIFACIAL_REAR_IRRADIANCE_RATIO == 0.135
    assert (PORTO.cold_cell_c, PORTO.cold_operating_cell_c, PORTO.hot_cell_c) == (-5, 20, 65)


def test_a_buildable_design_passes():
    check = check_topology(
        [(TRINA_450, 20)],
        [HUAWEI_4KTL],
        [PVString(0, 0, 0, 10), PVString(0, 1, 0, 10)],
        PORTO,
        dc_ac_ratio_range=(0.8, 2.5),
    )
    assert codes(check) == ["dc_power_above_recommended"]
    assert check.status == "warning"


@pytest.mark.parametrize(("modules", "flagged"), [(10, False), (11, True)])
def test_cold_voc_bounds_the_string_length(modules, flagged):
    # 52.9 V * (1 + -30 * -0.24 / 100) = 56.71 V a module at -5 °C: 10 fit in 600 V, 11 do not.
    check = check_topology([(TRINA_450, modules)], [HUAWEI_4KTL], [PVString(0, 0, 0, modules)], PORTO)
    assert ("cold_voc_above_limit" in codes(check, "error")) is flagged
    if flagged:
        finding = next(f for f in check.findings if f.code == "cold_voc_above_limit")
        assert finding.value == pytest.approx(11 * 52.9 * 1.072)
        assert finding.limit == 600
        assert (finding.inverter, finding.mppt, finding.string, finding.array) == (0, 0, 0, 0)


def test_the_lower_of_inverter_and_module_ratings_binds():
    # What-if: the same module with a 1000 V rating on a 1500 V-capable inverter.
    module = replace(TRINA_450, Max_System_Voltage=1000)
    inverter = replace(FRONIUS_GEN24_6, max_dc_voltage_v=1500, max_mppt_voltage_v=1500)
    check = check_topology([(module, 18)], [inverter], [PVString(0, 0, 0, 18)], PORTO)
    finding = next(f for f in check.findings if f.code == "cold_voc_above_limit")
    assert finding.limit == 1000


def test_a_colder_site_shortens_the_safe_string():
    siberia = ElectricalEnvironment(min_design_temp_c=-40, max_ambient_temp_c=40)
    assert codes(check_topology([(TRINA_450, 17)], [FRONIUS_GEN24_6], [PVString(0, 0, 0, 17)], PORTO)) == []
    assert "cold_voc_above_limit" in codes(
        check_topology([(TRINA_450, 17)], [FRONIUS_GEN24_6], [PVString(0, 0, 0, 17)], siberia)
    )
    # 16 modules stay under 1000 V, but their Vmp at the -15 °C operating cell passes 800 V.
    sixteen = check_topology([(TRINA_450, 16)], [FRONIUS_GEN24_6], [PVString(0, 0, 0, 16)], siberia)
    assert codes(sixteen) == ["vmp_above_mppt_max"]
    assert sixteen.status == "warning"


def test_short_strings_cannot_track_or_restart_on_the_hottest_days():
    # Yield and restart problems, not damage: warnings.
    check = check_topology([(TRINA_450, 2)], [HUAWEI_4KTL], [PVString(0, 0, 0, 2)], PORTO)
    assert codes(check) == ["hot_vmp_below_mppt_min", "hot_voc_below_startup"]
    assert check.status == "warning"
    assert {f.limit_field for f in check.findings} == {"min_mppt_voltage_v", "startup_voltage_v"}
    assert check_topology([(TRINA_450, 3)], [HUAWEI_4KTL], [PVString(0, 0, 0, 3)], PORTO).findings == ()


def test_the_vmp_coefficient_is_derived_when_unpublished():
    assert TRINA_450.T_Vmp_pct_effective == pytest.approx(-0.33)
    assert replace(TRINA_450, T_Vmp_pct=-0.25).T_Vmp_pct_effective == -0.25


def test_asymmetric_inputs_apply_their_own_limits():
    arrays = [(TRINA_450, 14), (JINKO_440, 5)]
    check = check_topology(arrays, [FRONIUS_GEN24_6], [PVString(0, 0, 0, 7, parallel=2), PVString(0, 1, 1, 5)], PORTO)
    # Jinko's 13.34 A Imp passes the first input's 25 A but not the second's 12.5 A.
    assert codes(check) == ["imp_above_mppt_limit", "missing_value"]
    assert [(f.code, f.mppt) for f in check.findings if f.severity == "warning"] == [("imp_above_mppt_limit", 1)]
    assert [(f.field, f.array) for f in check.findings if f.severity == "missing"] == [("Max_System_Voltage", 1)]

    swapped = check_topology(arrays, [FRONIUS_GEN24_6], [PVString(0, 1, 0, 7, parallel=2), PVString(0, 0, 1, 5)], PORTO)
    # Two Trina strings carry 2 * 10.91 A design Isc: under 40 A, over 20 A.
    assert codes(swapped, "error") == ["isc_above_mppt_limit", "too_many_strings"]
    assert all(f.mppt == 1 for f in swapped.findings if f.severity == "error")


def test_design_isc_uses_the_hot_cell_and_the_manufacturer_factor():
    hot_isc = 13.8 * (1 + 40 * 0.045 / 100)
    arrays = [(JINKO_440, 20)]
    strings = [PVString(0, 0, 0, 10, parallel=2)]
    assert "isc_above_mppt_limit" not in codes(check_topology(arrays, [FRONIUS_GEN24_6], strings, PORTO))
    # What-if: a comparison factor of 1.5 takes 2 * 14.05 A past the 40 A rating.
    strict = replace(FRONIUS_GEN24_6, isc_comparison_factor=1.5)
    finding = next(
        f for f in check_topology(arrays, [strict], strings, PORTO).findings if f.code == "isc_above_mppt_limit"
    )
    # The factor applies to STC Isc: 1.5 * 13.8 A is above the hot-cell 14.05 A.
    assert finding.value == pytest.approx(1.5 * 2 * 13.8)
    assert (finding.limit, finding.unit, finding.limit_field) == (40, "A", "max_short_circuit_current_a")
    # With a factor of 1 the hot-cell Isc is the larger.
    plain = check_topology(arrays, [replace(FRONIUS_GEN24_6, isc_comparison_factor=1.0)], strings, PORTO)
    assert "isc_above_mppt_limit" not in codes(plain)
    assert hot_isc > 13.8


def test_a_bifacial_module_adds_rear_current():
    bifacial = replace(TRINA_450, bifaciality=0.8)
    check = check_topology([(bifacial, 3)], [HUAWEI_4KTL], [PVString(0, 0, 0, 3)], PORTO)
    assert check.findings == ()
    two = check_topology(
        [(bifacial, 6)],
        [replace(HUAWEI_4KTL, mppts=(MpptSpec(20, 22, 2),))],
        [PVString(0, 0, 0, 3, parallel=2)],
        PORTO,
    )
    finding = next(f for f in two.findings if f.code == "isc_above_mppt_limit")
    assert finding.value == pytest.approx(2 * 10.74 * 1.016 * (1 + 0.8 * 0.135))


def test_parallel_strings_on_one_input_must_match():
    arrays = [(TRINA_450, 17)]
    lengths = check_topology(arrays, [FRONIUS_GEN24_6], [PVString(0, 0, 0, 8), PVString(0, 0, 0, 9)], PORTO)
    assert codes(lengths, "error") == ["parallel_strings_mismatch"]
    # Strings from two arrays of the same module and length share an input;
    # different modules do not.
    faces = check_topology(
        [(TRINA_450, 10), (TRINA_450, 10)], [FRONIUS_GEN24_6], [PVString(0, 0, 0, 10), PVString(0, 0, 1, 10)], PORTO
    )
    assert "parallel_strings_mismatch" not in codes(faces)
    mixed = check_topology(
        [(TRINA_450, 10), (JINKO_440, 10)], [FRONIUS_GEN24_6], [PVString(0, 0, 0, 10), PVString(0, 0, 1, 10)], PORTO
    )
    assert codes(mixed, "error") == ["parallel_strings_mismatch"]


def test_a_verto_plus_design_inside_its_limits_passes_its_checks():
    # Two strings of 8 Trina modules on each tracker: 21 600 Wp, under the
    # 22 500 Wp the inverter may carry, and 21.8 A design Isc a tracker.
    strings = [PVString(0, mppt, 0, 8, parallel=2) for mppt in range(3)]
    check = check_topology([(TRINA_450, 48)], [VERTO_15_PLUS], strings, PORTO)
    assert codes(check, "error") == []
    assert {f.field for f in check.findings if f.severity == "missing"} == {"isc_comparison_factor"}


def test_each_string_connection_is_held_to_its_own_rating():
    # Trina's hot-cell Isc at 65 °C is 10.74 * (1 + 40 * 0.0004) = 10.91 A.
    spec = replace(VERTO_15_PLUS.mppts[0], max_string_short_circuit_current_a=10.8, max_string_input_current_a=10.0)
    inverter = replace(VERTO_15_PLUS, mppts=(spec,) * 3, isc_comparison_factor=1.0)
    check = check_topology([(TRINA_450, 30)], [inverter], [PVString(0, 0, 0, 15, parallel=2)], PORTO)
    isc = next(f for f in check.findings if f.code == "isc_above_string_limit")
    assert (isc.severity, isc.value, isc.limit) == ("error", pytest.approx(10.74 * 1.016), 10.8)
    assert (isc.inverter, isc.mppt, isc.string, isc.array) == (0, 0, 0, 0)
    assert isc.limit_field == "max_string_short_circuit_current_a"
    imp = next(f for f in check.findings if f.code == "imp_above_string_limit")
    assert (imp.severity, imp.value, imp.limit) == ("warning", 10.09, 10.0)
    assert "isc_above_mppt_limit" not in codes(check)


def test_a_tracker_may_carry_its_maximum_generator_power_and_no_more():
    spec = replace(VERTO_15_PLUS.mppts[0], max_dc_power_w=13000)
    inverter = replace(VERTO_15_PLUS, mppts=(spec,) * 3)
    check = check_topology([(TRINA_450, 30)], [inverter], [PVString(0, 1, 0, 15, parallel=2)], PORTO)
    finding = next(f for f in check.findings if f.code == "mppt_dc_power_above_max")
    assert (finding.value, finding.limit, finding.inverter, finding.mppt) == (13500, 13000, 0, 1)
    assert finding.limit_field == "max_dc_power_w"


def test_the_inverter_short_circuit_rating_covers_all_its_inputs():
    # What-if 40 A over the inverter: four Trina strings carry 43.6 A.
    inverter = replace(VERTO_15_PLUS, max_short_circuit_current_a=40, isc_comparison_factor=1.0)
    strings = [PVString(0, 0, 0, 12, parallel=2), PVString(0, 1, 0, 12, parallel=2)]
    check = check_topology([(TRINA_450, 48)], [inverter], strings, PORTO)
    finding = next(f for f in check.findings if f.code == "isc_above_inverter_limit")
    assert (finding.value, finding.limit, finding.inverter) == (pytest.approx(4 * 10.74 * 1.016), 40, 0)
    assert codes(check, "error") == ["isc_above_inverter_limit"]


def test_limits_some_datasheets_publish_are_not_reported_missing():
    bare = replace(VERTO_15_PLUS, mppts=(MpptSpec(28, 50, 2),) * 3, max_short_circuit_current_a=None)
    check = check_topology([(TRINA_450, 30)], [bare], [PVString(0, 0, 0, 15, parallel=2)], PORTO)
    assert {f.field for f in check.findings if f.severity == "missing"} == {"isc_comparison_factor"}


def test_every_module_is_strung_exactly_once():
    short = check_topology([(TRINA_450, 22)], [HUAWEI_4KTL], [PVString(0, 0, 0, 10), PVString(0, 1, 0, 10)], PORTO)
    finding = next(f for f in short.findings if f.code == "unconnected_modules")
    assert (finding.value, finding.array, short.status) == (2, 0, "fail")
    over = check_topology([(TRINA_450, 18)], [HUAWEI_4KTL], [PVString(0, 0, 0, 10), PVString(0, 1, 0, 10)], PORTO)
    assert codes(over, "error") == ["array_overcommitted"]


def test_references_must_exist():
    check = check_topology(
        [(TRINA_450, 30)],
        [HUAWEI_4KTL],
        [PVString(1, 0, 0, 10), PVString(0, 2, 0, 10), PVString(0, 0, 3, 10), PVString(0, 0, 0, 10)],
        PORTO,
    )
    assert codes(check, "error") == ["unknown_array", "unknown_inverter", "unknown_mppt"]
    assert check_topology([(TRINA_450, 1)], [], [], PORTO).status == "fail"
    assert codes(check_topology([], [HUAWEI_4KTL], [], PORTO)) == ["no_strings"]


def test_missing_data_is_reported_and_known_limits_still_apply():
    unrated = replace(TRINA_450, Max_System_Voltage=None)
    check = check_topology([(unrated, 10)], [HUAWEI_4KTL], [PVString(0, 0, 0, 10)], PORTO)
    assert [(f.code, f.field, f.array) for f in check.findings] == [("missing_value", "Max_System_Voltage", 0)]
    assert check.status == "incomplete"
    # The inverter's 600 V still binds while the module rating is unknown.
    eleven = check_topology([(unrated, 11)], [HUAWEI_4KTL], [PVString(0, 0, 0, 11)], PORTO)
    assert eleven.status == "fail"
    assert "cold_voc_above_limit" in codes(eleven)


def test_an_inverter_without_inputs_reports_what_it_lacks():
    bare = InverterSpec(nominal_power_w=4000)
    check = check_topology([(TRINA_450, 10)], [bare], [PVString(0, 0, 0, 10)], PORTO)
    fields = sorted(f.field for f in check.findings)
    assert fields == ["max_dc_voltage_v", "max_mppt_voltage_v", "min_mppt_voltage_v", "mppts", "startup_voltage_v"]
    assert check.status == "incomplete"

    unknown_limits = replace(HUAWEI_4KTL, mppts=(MpptSpec(),) * 2)
    check = check_topology([(TRINA_450, 10)], [unknown_limits], [PVString(0, 0, 0, 10)], PORTO)
    assert sorted(f.field for f in check.findings) == [
        "max_input_current_a",
        "max_short_circuit_current_a",
        "max_strings",
    ]


def test_unbounded_limits_never_trigger():
    ideal = InverterSpec(
        nominal_power_w=4000,
        max_dc_voltage_v=math.inf,
        max_mppt_voltage_v=math.inf,
        min_mppt_voltage_v=0,
        startup_voltage_v=0,
        mppts=(MpptSpec(math.inf, math.inf, 10),),
        isc_comparison_factor=1.0,
    )
    check = check_topology([(TRINA_450, 1)], [ideal], [PVString(0, 0, 0, 1)], PORTO)
    assert check.status == "pass"


def test_dc_ac_ratio_guidance_is_optional():
    strings = [PVString(0, 0, 0, 10), PVString(0, 1, 0, 10)]
    assert "dc_ac_ratio_outside_range" not in codes(check_topology([(TRINA_450, 20)], [HUAWEI_4KTL], strings, PORTO))
    check = check_topology([(TRINA_450, 20)], [HUAWEI_4KTL], strings, PORTO, dc_ac_ratio_range=(0.8, 1.5))
    finding = next(f for f in check.findings if f.code == "dc_ac_ratio_outside_range")
    assert (finding.value, finding.limit) == (pytest.approx(9000 / 4000), 1.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_design_temp_c": 10, "max_ambient_temp_c": 10},
        {"min_design_temp_c": math.nan, "max_ambient_temp_c": 40},
        {"min_design_temp_c": -5, "max_ambient_temp_c": 40, "mounting": "roof"},
    ],
)
def test_invalid_environments_are_refused(kwargs):
    with pytest.raises(ValueError):
        ElectricalEnvironment(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"inverter": -1, "mppt": 0, "array": 0, "modules": 1},
        {"inverter": 0, "mppt": 0, "array": 0, "modules": 0},
        {"inverter": 0, "mppt": 0, "array": 0, "modules": 1, "parallel": 0},
        {"inverter": 0, "mppt": True, "array": 0, "modules": 1},
        {"inverter": 0, "mppt": 0, "array": 0.0, "modules": 1},
    ],
)
def test_invalid_strings_are_refused(kwargs):
    with pytest.raises(ValueError):
        PVString(**kwargs)


def test_coefficient_overrides_drive_the_check_as_they_drive_energy():
    # An absolute Voc coefficient of -0.30 V/°C: 10 * (52.9 + 0.30 * 30) = 619 V at -5 °C.
    module = replace(TRINA_450, beta_voc_abs=-0.30)
    check = check_topology([(module, 10)], [HUAWEI_4KTL], [PVString(0, 0, 0, 10)], PORTO)
    finding = next(f for f in check.findings if f.code == "cold_voc_above_limit")
    assert finding.value == pytest.approx(619.0)
    assert check.status == "fail"
    # alpha_sc_abs raises the hot-cell Isc: 10.74 + 0.02 * 40.
    hot = replace(TRINA_450, alpha_sc_abs=0.02)
    two = check_topology(
        [(hot, 6)], [replace(HUAWEI_4KTL, mppts=(MpptSpec(20, 23, 2),))], [PVString(0, 0, 0, 3, parallel=2)], PORTO
    )
    assert next(f for f in two.findings if f.code == "isc_above_mppt_limit").value == pytest.approx(2 * 11.54)
    # The derived Vmp coefficient follows the overrides too.
    assert replace(TRINA_450, gamma_pmp=-0.35).T_Vmp_pct_effective == pytest.approx(-0.35 - 0.04)


def test_each_exceeded_voltage_ceiling_is_reported_with_its_source():
    module = replace(TRINA_450, Max_System_Voltage=550)
    check = check_topology([(module, 10)], [HUAWEI_4KTL], [PVString(0, 0, 0, 10)], PORTO)
    assert codes(check, "error") == ["cold_voc_above_limit"]
    module = replace(TRINA_450, Max_System_Voltage=500)
    check = check_topology([(module, 11)], [HUAWEI_4KTL], [PVString(0, 0, 0, 11)], PORTO)
    found = sorted((f.limit_field, f.limit, f.unit) for f in check.findings if f.code == "cold_voc_above_limit")
    assert found == [("Max_System_Voltage", 500, "V"), ("max_dc_voltage_v", 600, "V")]


def test_an_unstated_isc_comparison_is_missing_and_the_hot_cell_isc_still_applies():
    unstated = replace(FRONIUS_GEN24_6, isc_comparison_factor=None)
    check = check_topology([(TRINA_450, 7)], [unstated], [PVString(0, 0, 0, 7)], PORTO)
    assert [(f.code, f.field) for f in check.findings] == [("missing_value", "isc_comparison_factor")]
    over = check_topology(
        [(JINKO_440, 6)], [replace(unstated, mppts=(MpptSpec(25, 28, 2),))], [PVString(0, 0, 0, 3, parallel=2)], PORTO
    )
    assert "isc_above_mppt_limit" in codes(over, "error")


def test_maximum_and_recommended_generator_power_differ():
    strings = [PVString(0, 0, 0, 10), PVString(0, 1, 0, 10)]
    hard = replace(HUAWEI_4KTL, recommended_dc_power_w=None, max_dc_power_w=6000)
    check = check_topology([(TRINA_450, 20)], [hard], strings, PORTO)
    assert codes(check, "error") == ["dc_power_above_max"]
    with pytest.raises(ValueError, match="recommended_dc_power_w"):
        replace(HUAWEI_4KTL, max_dc_power_w=5000)


def test_stated_cell_temperatures_replace_the_screening_rises():
    # 3 modules: 113.9 V hot Vmp at the 65 °C screening cell, 106 V at 88 °C.
    floor = replace(HUAWEI_4KTL, min_mppt_voltage_v=110)
    strings = [PVString(0, 0, 0, 3)]
    assert "hot_vmp_below_mppt_min" not in codes(check_topology([(TRINA_450, 3)], [floor], strings, PORTO))
    hot = ElectricalEnvironment(min_design_temp_c=-5, max_ambient_temp_c=40, hot_cell_temp_c=88)
    finding = next(f for f in check_topology([(TRINA_450, 3)], [floor], strings, hot).findings)
    assert (finding.code, finding.value) == ("hot_vmp_below_mppt_min", pytest.approx(3 * 44.6 * (1 - 63 * 0.0033)))
    cold = ElectricalEnvironment(-5, 40, cold_operating_cell_temp_c=-5)
    assert ElectricalEnvironment(-5, 40).cold_operating_cell_c == 20
    assert cold.cold_operating_cell_c == -5


def test_every_finding_follows_the_published_catalogue():
    from breos.electrical import FINDING_CODES

    scenarios = [
        check_topology([(TRINA_450, 22)], [HUAWEI_4KTL], [PVString(0, 0, 0, 11), PVString(0, 2, 0, 11)], PORTO),
        check_topology([(TRINA_450, 2)], [HUAWEI_4KTL], [PVString(0, 0, 0, 2)], PORTO),
        check_topology([(JINKO_440, 14)], [FRONIUS_GEN24_6], [PVString(0, 1, 0, 7, parallel=2)], PORTO),
        check_topology([(TRINA_450, 10)], [InverterSpec(nominal_power_w=4000)], [PVString(0, 0, 0, 10)], PORTO),
    ]
    seen = set()
    for check in scenarios:
        for finding in check.findings:
            kind = FINDING_CODES[finding.code]
            assert (finding.severity, finding.unit) == (kind.severity, kind.unit)
            for name in ("inverter", "mppt", "string", "array"):
                if name not in kind.located_by:
                    assert getattr(finding, name) is None or finding.code == "missing_value"
            seen.add(finding.code)
    assert len(seen) >= 8


def test_the_documented_catalogue_matches_the_code():
    from pathlib import Path

    from breos.electrical import FINDING_CODES

    text = (Path(__file__).parents[1] / "docs" / "getting-started" / "interpreting-results.md").read_text()
    rows = {}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 5 and cells[0].startswith("`") and cells[0].strip("`") in FINDING_CODES:
            rows[cells[0].strip("`")] = (cells[1], cells[2] or None)
    assert rows == {code: (kind.severity, kind.unit) for code, kind in FINDING_CODES.items()}
