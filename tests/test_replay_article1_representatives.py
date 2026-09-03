import pandas as pd
import pytest

from tools.replay_article1_representatives import _verify_replayed_representatives


def _representative(**overrides) -> pd.DataFrame:
    row = {
        "Representative": "knee",
        "Modules": 9,
        "Battery_kWh": 7.0,
        "Tilt": 35.0,
        "Azimuth": 200.0,
        "Projected_Grid_Independence_%": 71.24,
        "Projected_NPV_Eur": 3271.49,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_representative_replay_accepts_matching_source_metrics():
    deltas = _verify_replayed_representatives(_representative(), _representative())

    assert deltas == {"Projected_Grid_Independence_%": 0.0, "Projected_NPV_Eur": 0.0}


def test_representative_replay_rejects_a_changed_design():
    with pytest.raises(ValueError, match="changed Modules"):
        _verify_replayed_representatives(_representative(), _representative(Modules=8))


def test_representative_replay_rejects_changed_metrics():
    with pytest.raises(ValueError, match="changed Projected_NPV_Eur"):
        _verify_replayed_representatives(_representative(), _representative(Projected_NPV_Eur=3271.48))
