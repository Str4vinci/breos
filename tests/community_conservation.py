"""Independent ADR 0006 K6 closure checks, per interval and calendar year.

Only settlement identities (Closure 1–5) apply here. Dispatch and money
closure belong to the community runner and pricing PRs respectively.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from breos.community.settlement import SETTLEMENT_COLUMNS, SETTLEMENT_SCHEMA_VERSION


def assert_settlement_closure(
    frame: pd.DataFrame, *, coefficients: Mapping[str, float] | None = None, atol: float = 1e-7
) -> None:
    """Check closure, receiver rules, attribution and nonnegative finite values.

    Pass the original coefficients for fixed allocation so a wrong receiver
    denominator cannot conceal missing allocations. Annual identities sum the
    individual interval decisions: annual net meters cannot recover receivers.
    """
    assert frame.attrs["settlement_schema_version"] == SETTLEMENT_SCHEMA_VERSION
    assert tuple(frame.columns) == SETTLEMENT_COLUMNS
    values = frame.to_numpy()
    assert np.isfinite(values).all(), "non-finite settlement value"
    assert (values >= 0).all(), "negative settlement value"
    participants = frame.xs("participant", level="site_type")
    sites = frame[frame.index.get_level_values("site_type") != "community"]
    community = frame.xs("community", level="site_type").droplevel("site_id")

    def balanced(actual, expected, label):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=atol, err_msg=label)
        # Each column/site identity must also hold after summing each calendar
        # year. Support a partial year and multiple years, including leap years.
        groups = [actual.index.get_level_values("interval").year]
        groups.extend(actual.index.get_level_values(name) for name in actual.index.names if name != "interval")
        yearly_actual = actual.groupby(groups).sum()
        yearly_expected = expected.groupby(groups).sum()
        np.testing.assert_allclose(yearly_actual, yearly_expected, rtol=0, atol=atol, err_msg=f"annual {label}")

    totals = sites.groupby(level="interval", sort=True).sum()
    participant_totals = participants.groupby(level="interval", sort=True).sum()
    energy = community["Energy_For_Sharing"]
    used = participant_totals["Shared_Used"]
    allocated = participant_totals["Shared_Allocated"]
    balanced(energy, totals["Pooled_Injection"], "energy for sharing")
    balanced(community["Shared_Used_Total"], used, "used total")
    balanced(sites["Shared_Allocated"], sites["Shared_Used"] + sites["Participant_Surplus"], "allocation split")
    balanced(community["Community_Surplus"], energy - used, "community surplus")
    balanced(
        community["Community_Surplus"],
        participant_totals["Participant_Surplus"] + energy - allocated,
        "surplus including unallocated injection",
    )
    balanced(sites["Metered_Import"], sites["Shared_Used"] + sites["Supplier_Import"], "meter supply")
    balanced(totals["Attributed_Shared_Used"], used, "used attribution")
    balanced(totals["Attributed_Surplus"], community["Community_Surplus"], "surplus attribution")
    balanced(sites["Metered_Export"], sites["Pooled_Injection"] + sites["Own_Export"], "export split")
    assert (sites["Shared_Used"] <= np.minimum(sites["Shared_Allocated"], sites["Metered_Import"])).all()

    receivers = participants["Metered_Export"] == 0
    if frame.attrs["allocation"] == "fixed":
        assert coefficients is not None
        weights = pd.Series(
            [coefficients[key] for key in participants.index.get_level_values("site_id")], index=participants.index
        ).where(receivers, 0.0)
    else:
        weights = participants["Metered_Import"].where(receivers, 0.0)
    denominator = weights.groupby(level="interval").sum()
    expected_allocated = energy.where(denominator > 0, 0.0)
    balanced(allocated, expected_allocated, "allocation with positive receiver denominator")
    expected_coefficients = weights / denominator.reindex(weights.index, level="interval").replace(0.0, np.nan)
    balanced(participants["Coefficient"], expected_coefficients.fillna(0.0), "receiver coefficients")
    balanced(
        participants["Shared_Allocated"],
        participants["Coefficient"] * energy.reindex(participants.index, level="interval"),
        "allocation by coefficient",
    )
    for source, total in (("Attributed_Shared_Used", used), ("Attributed_Surplus", community["Community_Surplus"])):
        fraction = sites["Pooled_Injection"] / energy.reindex(sites.index, level="interval").replace(0.0, np.nan)
        balanced(sites[source], fraction.fillna(0.0) * total.reindex(sites.index, level="interval"), source)
    np.testing.assert_array_equal(sites[list(SETTLEMENT_COLUMNS[11:])], 0.0)
    np.testing.assert_array_equal(community[list(SETTLEMENT_COLUMNS[:11])], 0.0)
    if "asset" in frame.index.get_level_values("site_type"):
        rows = frame.xs("asset", level="site_type")
        np.testing.assert_array_equal(
            rows[["Coefficient", "Shared_Allocated", "Shared_Used", "Participant_Surplus", "Own_Export"]], 0.0
        )
