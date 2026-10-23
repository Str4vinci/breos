"""ADR 0006 reference arithmetic and Q4–Q6, independent of dispatch defaults."""

from fractions import Fraction
from itertools import permutations

import numpy as np
import pandas as pd
import pytest

import breos
from breos.community.settlement import SETTLEMENT_COLUMNS, MeterBalances, settle_meter_balances
from tests.community_conservation import assert_settlement_closure

COEFFICIENTS = {"a": 0.5, "b": 0.3, "c": 0.2}
INDEX = pd.date_range("2026-01-01", periods=1, freq="15min", tz="Europe/Lisbon")


def reference(imports, *, exports=(0, 0, 0), injection=6000, allocation="proportional", pool=True):
    participants = {key: MeterBalances([c], [x]) for key, c, x in zip(COEFFICIENTS, imports, exports)}
    coefficients = COEFFICIENTS if allocation == "fixed" else None
    frame = settle_meter_balances(
        participants,
        {"roof": MeterBalances([0], [injection])},
        index=INDEX,
        allocation=allocation,
        coefficients=coefficients,
        pool_participant_export=pool,
    )
    assert_settlement_closure(frame, coefficients=coefficients, atol=1e-9)
    return frame.droplevel("interval")


def assert_participants(frame, **columns):
    for name, expected in columns.items():
        np.testing.assert_allclose(frame.xs("participant")[name], expected, rtol=0, atol=1e-9, err_msg=name)


def test_s1_proportional():
    frame = reference((3000, 1000, 4000))
    assert_participants(
        frame,
        Coefficient=(0.375, 0.125, 0.5),
        Shared_Allocated=(2250, 750, 3000),
        Shared_Used=(2250, 750, 3000),
        Participant_Surplus=(0, 0, 0),
        Supplier_Import=(750, 250, 1000),
    )
    assert frame.loc[("community", ""), "Community_Surplus"] == 0
    assert frame.loc[("asset", "roof"), "Attributed_Shared_Used"] == 6000


@pytest.mark.parametrize("allocation", ["fixed", "proportional"])
def test_s2_fixed_surplus_is_not_reallocated(allocation):
    frame = reference((1000, 3000, 4000), allocation=allocation)
    if allocation == "fixed":
        assert_participants(
            frame,
            Coefficient=(0.5, 0.3, 0.2),
            Shared_Allocated=(3000, 1800, 1200),
            Shared_Used=(1000, 1800, 1200),
            Participant_Surplus=(2000, 0, 0),
            Supplier_Import=(0, 1200, 2800),
        )
        assert frame.loc[("community", ""), "Community_Surplus"] == 2000
        assert frame.loc[("asset", "roof"), "Attributed_Surplus"] == 2000
    else:
        assert_participants(
            frame,
            Coefficient=(0.125, 0.375, 0.5),
            Shared_Allocated=(750, 2250, 3000),
            Shared_Used=(750, 2250, 3000),
            Participant_Surplus=(0, 0, 0),
            Supplier_Import=(250, 750, 1000),
        )
        assert frame.loc[("community", ""), "Community_Surplus"] == 0


def test_s3_exporting_participant_and_exact_rational_attribution():
    frame = reference((1000, 3000, 0), exports=(0, 0, 500), injection=5500, allocation="fixed")
    assert_participants(
        frame,
        Coefficient=(0.625, 0.375, 0),
        Shared_Allocated=(3750, 2250, 0),
        Shared_Used=(1000, 2250, 0),
        Participant_Surplus=(2750, 0, 0),
        Supplier_Import=(0, 750, 0),
    )
    assert frame.loc[("community", ""), "Energy_For_Sharing"] == 6000
    assert frame.loc[("community", ""), "Shared_Used_Total"] == pytest.approx(3250, rel=0, abs=1e-9)
    assert frame.loc[("community", ""), "Community_Surplus"] == pytest.approx(2750, rel=0, abs=1e-9)
    for row, share in ((("asset", "roof"), Fraction(11, 12)), (("participant", "c"), Fraction(1, 12))):
        np.testing.assert_allclose(
            frame.loc[row, ["Attributed_Shared_Used", "Attributed_Surplus"]],
            [float(3250 * share), float(2750 * share)],
            rtol=0,
            atol=1e-9,
        )


def test_s5_zero_denominator_despite_one_receiver():
    frame = reference((0, 0, 0), exports=(300, 200, 0), injection=2500)
    assert_participants(frame, Coefficient=(0, 0, 0), Shared_Allocated=(0, 0, 0), Shared_Used=(0, 0, 0))
    assert frame.loc[("community", ""), "Energy_For_Sharing"] == 3000
    assert frame.loc[("community", ""), "Community_Surplus"] == 3000
    assert_participants(frame, Attributed_Surplus=(300, 200, 0))
    assert frame.loc[("asset", "roof"), "Attributed_Surplus"] == 2500


@pytest.mark.parametrize("allocation", ["fixed", "proportional"])
def test_unpooled_export_still_excludes_exporter(allocation):
    frame = reference((1000, 3000, 0), exports=(0, 0, 500), injection=5500, allocation=allocation, pool=False)
    assert frame.loc[("participant", "c"), "Own_Export"] == 500
    assert (frame.loc[("participant", "c"), ["Coefficient", "Shared_Allocated", "Attributed_Shared_Used"]] == 0).all()
    assert frame.loc[("community", ""), "Energy_For_Sharing"] == 5500


def test_fixed_idle_receiver_keeps_its_share_as_surplus():
    frame = reference((0, 3000, 4000), allocation="fixed")
    assert_participants(frame, Shared_Allocated=(3000, 1800, 1200), Participant_Surplus=(3000, 0, 0))


@pytest.mark.parametrize("allocation", ["fixed", "proportional"])
def test_no_receiver_or_only_zero_coefficient_receivers(allocation):
    participants = {"a": MeterBalances([0, 0], [300, 300]), "b": MeterBalances([0, 100], [200, 0])}
    coefficients = {"a": 1.0, "b": 0.0} if allocation == "fixed" else None
    frame = settle_meter_balances(
        participants,
        {},
        index=pd.date_range(INDEX[0], periods=2, freq="15min"),
        allocation=allocation,
        coefficients=coefficients,
    )
    assert_settlement_closure(frame, coefficients=coefficients, atol=1e-9)
    community = frame.xs("community", level="site_type")
    assert community["Community_Surplus"].iloc[0] == 500
    assert community["Community_Surplus"].iloc[1] == (300 if allocation == "fixed" else 200)


def test_schema_units_asset_imports_and_public_export():
    imports = np.array([10.0, 0.0])
    frame = settle_meter_balances(
        {"a": MeterBalances([80, 80], [0, 0])},
        {"roof": MeterBalances(imports, [0, 40])},
        index=pd.date_range(INDEX[0], periods=2, freq="15min"),
        allocation="proportional",
    )
    assert_settlement_closure(frame)
    assert tuple(frame.columns) == SETTLEMENT_COLUMNS
    assert frame.index.names == ["interval", "site_type", "site_id"]
    assert frame.attrs["settlement_schema_version"] == "1.0"
    assert frame.attrs["energy_unit"] == "Wh"
    assert frame.attrs["interval_minutes"] == 15
    np.testing.assert_array_equal(frame.xs("roof", level="site_id")["Supplier_Import"], imports)
    np.testing.assert_array_equal(frame.xs("a", level="site_id")["Shared_Used"], [0, 40])
    np.testing.assert_array_equal(imports, [10, 0])
    assert "Community" in breos.__all__


@pytest.mark.parametrize("allocation", ["fixed", "proportional"])
@pytest.mark.parametrize("pool", [True, False])
def test_q4_reordering_is_bit_identical(allocation, pool):
    participants = {
        "z": MeterBalances([1e8, 1e-6, 0], [0, 0, 1e8]),
        "a": MeterBalances([1e-8, 1, 0], [0, 0, 1e-8]),
        "m": MeterBalances([1, 1e8, 0], [0, 0, 1]),
    }
    assets = {"b": MeterBalances([0, 2, 0], [1e-8, 0, 1e-8]), "y": MeterBalances([0, 0, 0], [1e8, 1, 1e8])}
    original = None
    for order in permutations(participants):
        for asset_order in permutations(assets):
            coefficients = {key: {"z": 0.5, "a": 0.3, "m": 0.2}[key] for key in reversed(order)}
            frame = settle_meter_balances(
                {key: participants[key] for key in order},
                {key: assets[key] for key in asset_order},
                index=pd.date_range(INDEX[0], periods=3, freq="15min"),
                allocation=allocation,
                coefficients=coefficients if allocation == "fixed" else None,
                pool_participant_export=pool,
            )
            if original is None:
                original = frame
            pd.testing.assert_frame_equal(frame, original, check_exact=True)
            # assert_frame_equal's numeric equality also accepts +/-0. Check
            # the uint64 representation to demand literal bit identity.
            np.testing.assert_array_equal(frame.to_numpy().view(np.uint64), original.to_numpy().view(np.uint64))


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("allocation", ["fixed", "proportional"])
@pytest.mark.parametrize("pool", [True, False])
def test_q5_q6_seeded_meter_properties(seed, allocation, pool):
    # Hypothesis is not a dev dependency. Exercise a full leap year plus a
    # second calendar year with reproducible random meters, idle receivers,
    # zero injection, empty assets, and fixed zero receiver denominators.
    rng = np.random.default_rng(seed)
    n = 366 * 96 + 2
    index = pd.date_range("2024-01-01", periods=n, freq="15min", tz="Europe/Lisbon")
    participants, assets = {}, {}
    count = int(rng.integers(1, 7))
    for kind, mapping, size in (("p", participants, count), ("roof", assets, seed % 3)):
        for number in range(size):
            status = rng.integers(0, 3, n)
            energy = rng.uniform(0, 5000, n)
            mapping[f"{kind}-{number}"] = MeterBalances(
                np.where(status == 1, energy, 0), np.where(status == 2, energy, 0)
            )
    coefficients = dict(zip(participants, rng.integers(0, 5, count).astype(float)))
    coefficients[next(iter(coefficients))] += 1
    total = sum(coefficients.values())
    coefficients = {key: value / total for key, value in coefficients.items()}
    frame = settle_meter_balances(
        participants,
        assets,
        index=index,
        allocation=allocation,
        coefficients=coefficients if allocation == "fixed" else None,
        pool_participant_export=pool,
    )
    assert_settlement_closure(frame, coefficients=coefficients if allocation == "fixed" else None, atol=1e-6)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"participants": {}}, "at least one"),
        ({"assets": {"a": MeterBalances([0], [0])}}, "unique"),
        ({"participants": {"Bad Id": MeterBalances([0], [0])}}, "ids"),
        ({"participants": {"a": MeterBalances([1], [1])}}, "simultaneous"),
        ({"participants": {"a": MeterBalances([-1], [0])}}, "nonnegative"),
        ({"participants": {"a": MeterBalances([np.nan], [0])}}, "finite"),
        ({"assets": {"roof": MeterBalances([0], [np.inf])}}, "finite"),
        ({"participants": {"a": MeterBalances([[1]], [0])}}, "shape"),
        ({"participants": {"a": MeterBalances([1, 2], [0, 0])}}, "shape"),
        ({"allocation": "dynamic"}, "allocation"),
        ({"coefficients": {"a": 1}}, "does not take"),
        ({"allocation": "fixed"}, "exactly"),
        ({"allocation": "fixed", "coefficients": {"b": 1}}, "exactly"),
        ({"allocation": "fixed", "coefficients": {"a": 0.9}}, "sum to 1"),
        ({"allocation": "fixed", "coefficients": {"a": -1}}, "nonnegative"),
        ({"allocation": "fixed", "coefficients": {"a": np.nan}}, "finite"),
        ({"allocation": "fixed", "coefficients": {"a": np.inf}}, "finite"),
        ({"pool_participant_export": "false"}, "bool"),
        ({"index": pd.DatetimeIndex([])}, "nonempty"),
        ({"index": pd.DatetimeIndex([pd.NaT])}, "NaT"),
        ({"index": pd.date_range(INDEX[0], periods=2, freq="h")}, "15-minute"),
        ({"index": pd.DatetimeIndex([INDEX[0], INDEX[0]])}, "15-minute"),
        ({"index": pd.date_range(INDEX[0], periods=2, freq="-15min")}, "15-minute"),
    ],
)
def test_invalid_inputs(changes, match):
    kwargs = dict(participants={"a": MeterBalances([1], [0])}, assets={}, index=INDEX, allocation="proportional")
    with pytest.raises(ValueError, match=match):
        settle_meter_balances(**(kwargs | changes))


@pytest.mark.parametrize("column", ["Shared_Allocated", "Shared_Used", "Community_Surplus", "Attributed_Surplus"])
def test_closure_helper_detects_corrupted_ledger(column):
    frame = settle_meter_balances(
        {"a": MeterBalances([10], [0])}, {"roof": MeterBalances([0], [20])}, index=INDEX, allocation="proportional"
    )
    row = (INDEX[0], "community", "") if column == "Community_Surplus" else (INDEX[0], "participant", "a")
    frame.loc[row, column] += 1
    with pytest.raises(AssertionError):
        assert_settlement_closure(frame)
