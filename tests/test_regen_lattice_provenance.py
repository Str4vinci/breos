import pandas as pd
import pytest

from tools.revision.grid_eval import GI, NPV
from tools.revision.regen_lattice_provenance import (
    _validate_csv_archive,
    _validate_front,
    _validate_result_grid,
)


def _result_table():
    return pd.DataFrame(
        {
            "Design_ID": [0, 1],
            "Modules": [1, 1],
            "Battery_kWh": [0.0, 1.0],
            "Tilt": [20.0, 20.0],
            "Azimuth": [180.0, 180.0],
            GI: [20.0, 30.0],
            NPV: [100.0, 90.0],
        }
    )


def test_result_grid_requires_every_affordable_point_once():
    table = _result_table()

    assert _validate_result_grid(table, [(1, 0.0), (1, 1.0)], [20.0], [180.0]) == {0, 1}

    duplicate = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicates"):
        _validate_result_grid(duplicate, [(1, 0.0), (1, 1.0)], [20.0], [180.0])


def test_result_grid_rejects_non_integral_design_ids():
    table = _result_table()
    table["Design_ID"] = [0.0, 1.5]

    with pytest.raises(ValueError, match="non-integral Design_ID"):
        _validate_result_grid(table, [(1, 0.0), (1, 1.0)], [20.0], [180.0])


def test_result_grid_requires_the_generator_coordinate_to_id_mapping():
    table = _result_table()
    table["Design_ID"] = [1, 0]

    with pytest.raises(ValueError, match="mapping does not match grid_eval enumeration"):
        _validate_result_grid(table, [(1, 0.0), (1, 1.0)], [20.0], [180.0])


def test_front_must_equal_a_fresh_pareto_reduction():
    table = _result_table()
    front = table.copy()

    assert _validate_front(table, front) == 2

    with pytest.raises(ValueError, match="fresh Pareto reduction"):
        _validate_front(table, front.iloc[[0]])


def test_front_values_must_match_the_result_table():
    table = _result_table()
    front = table.copy()
    front.loc[0, NPV] += 1.0

    with pytest.raises(AssertionError):
        _validate_front(table, front)


def test_archive_requires_complete_design_year_coverage(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    pd.DataFrame({"Design_ID": [0, 0, 1, 1], "Year": [1, 2, 1, 2]}).to_csv(archive / "annual_00000.csv", index=False)

    verified = _validate_csv_archive(archive, {0, 1}, years_projection=2)

    assert verified["rows"] == 4
    assert verified["designs"] == 2
    assert len(verified["shards"]) == 1
    assert verified["shards"][0]["sha256"]

    pd.DataFrame({"Design_ID": [0, 0, 1], "Year": [1, 2, 1]}).to_csv(archive / "annual_00000.csv", index=False)
    with pytest.raises(ValueError, match="archive coverage is incomplete"):
        _validate_csv_archive(archive, {0, 1}, years_projection=2)


def test_archive_rejects_duplicate_headers(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "annual_00000.csv").write_text("Design_ID,Year,Export_kWh,Export_kWh\n0,1,2.0,2.0\n")

    with pytest.raises(ValueError, match=r"duplicate columns: \['Export_kWh'\]"):
        _validate_csv_archive(archive, {0}, years_projection=1)
