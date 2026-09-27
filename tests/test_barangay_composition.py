import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from str_suitability.config import PROJECTED_CRS
from str_suitability.rural.psa import RURAL, URBAN, URBAN_RURAL_COLUMN
from str_suitability.spatial.barangay_composition import (
    CELL_ID_COLUMN,
    COVERAGE_COLUMNS,
    GRID_AREA_COLUMN,
    PERCENTAGE_COLUMN,
    UNCLASSIFIED_LABEL,
    barangay_cell_overlay,
    cell_coverage,
)

CELL_SIZE_M = 1000.0


def make_grid(spans):
    frame = gpd.GeoDataFrame(
        [
            {CELL_ID_COLUMN: cell_id, "geometry": box(start, 0.0, start + width, CELL_SIZE_M)}
            for cell_id, start, width in spans
        ],
        geometry="geometry",
        crs=PROJECTED_CRS,
    )
    return frame.assign(projected_area_m2=frame.geometry.area)


def make_barangays(records):
    return gpd.GeoDataFrame(
        [
            {**attributes, "geometry": box(start, 0.0, end, CELL_SIZE_M)}
            for attributes, start, end in records
        ],
        geometry="geometry",
        crs=PROJECTED_CRS,
    )


def one_cell():
    return make_grid([("r0000c0000", 0.0, CELL_SIZE_M)])


def split_barangays():
    return make_barangays(
        [
            ({"psgc": "0400000001", "adm4_name": "Rural A", URBAN_RURAL_COLUMN: RURAL}, 0.0, 600.0),
            ({"psgc": "0400000002", "adm4_name": "Rural B", URBAN_RURAL_COLUMN: RURAL}, 600.0, 850.0),
            ({"psgc": "0400000003", "adm4_name": "Urban C", URBAN_RURAL_COLUMN: URBAN}, 850.0, 1000.0),
        ]
    )


def coverage_of(grid, barangays):
    overlay, _ = barangay_cell_overlay(grid, barangays)
    return overlay, cell_coverage(overlay, grid)


def test_percentages_are_area_shares_of_the_cell():
    overlay, _ = coverage_of(one_cell(), split_barangays())
    percentages = overlay.set_index("psgc")[PERCENTAGE_COLUMN].to_dict()
    assert percentages["0400000001"] == pytest.approx(60.0)
    assert percentages["0400000002"] == pytest.approx(25.0)
    assert percentages["0400000003"] == pytest.approx(15.0)


def test_coverage_comes_from_area_not_from_barangay_count():
    _, (coverage, _) = coverage_of(one_cell(), split_barangays())
    row = coverage.iloc[0]
    assert row["barangay_count"] == 3
    assert row["rural_coverage"] == pytest.approx(0.85)
    assert row["urban_coverage"] == pytest.approx(0.15)


def test_one_rural_and_one_urban_barangay_can_be_lopsided():
    barangays = make_barangays(
        [
            ({"psgc": "0400000001", "adm4_name": "Rural", URBAN_RURAL_COLUMN: RURAL}, 0.0, 850.0),
            ({"psgc": "0400000002", "adm4_name": "Urban", URBAN_RURAL_COLUMN: URBAN}, 850.0, 1000.0),
        ]
    )
    _, (coverage, _) = coverage_of(one_cell(), barangays)
    row = coverage.iloc[0]
    assert row["barangay_count"] == 2
    assert row["rural_coverage"] == pytest.approx(0.85)
    assert row["urban_coverage"] == pytest.approx(0.15)


def test_a_cell_keeps_every_barangay_it_overlaps():
    overlay, (coverage, _) = coverage_of(one_cell(), split_barangays())
    assert len(overlay) == 3
    assert coverage.iloc[0]["barangay_count"] == 3
    assert set(overlay["psgc"]) == {"0400000001", "0400000002", "0400000003"}


def test_the_dominant_barangay_is_a_diagnostic_not_the_assignment():
    _, (coverage, _) = coverage_of(one_cell(), split_barangays())
    row = coverage.iloc[0]
    assert row["dominant_barangay_psgc"] == "0400000001"
    assert row["dominant_barangay_share"] == pytest.approx(0.6)
    assert row["rural_coverage"] == pytest.approx(0.85)


def test_grid_area_and_covered_area_shares_differ_when_a_cell_is_partly_uncovered():
    barangays = make_barangays(
        [({"psgc": "0400000001", "adm4_name": "Rural", URBAN_RURAL_COLUMN: RURAL}, 0.0, 600.0)]
    )
    overlay, (coverage, _) = coverage_of(one_cell(), barangays)
    row = coverage.iloc[0]
    assert overlay.iloc[0][PERCENTAGE_COLUMN] == pytest.approx(60.0)
    assert overlay.iloc[0]["covered_area_share"] == pytest.approx(1.0)
    assert row["rural_coverage"] == pytest.approx(0.6)
    assert row["rural_share_of_covered"] == pytest.approx(1.0)
    assert row["uncovered_coverage"] == pytest.approx(0.4)


def test_a_cell_no_barangay_overlaps_is_wholly_uncovered():
    grid = make_grid([("r0000c0000", 0.0, CELL_SIZE_M), ("r0000c0001", CELL_SIZE_M, CELL_SIZE_M)])
    barangays = make_barangays(
        [({"psgc": "0400000001", "adm4_name": "Rural", URBAN_RURAL_COLUMN: RURAL}, 0.0, 500.0)]
    )
    overlay, (coverage, report) = coverage_of(grid, barangays)
    assert set(overlay[CELL_ID_COLUMN]) == {"r0000c0000"}
    uncovered = coverage.set_index(CELL_ID_COLUMN).loc["r0000c0001"]
    assert uncovered["barangay_count"] == 0
    assert uncovered["uncovered_coverage"] == pytest.approx(1.0)
    assert uncovered["grid_area_m2"] == pytest.approx(CELL_SIZE_M**2)
    assert pd.isna(uncovered["dominant_barangay_psgc"])
    assert report["cells_without_barangay"] == 1
    assert report["cells_with_barangay"] == 1


def test_coverage_shares_sum_to_one_even_when_a_cell_is_partly_uncovered():
    barangays = make_barangays(
        [
            ({"psgc": "0400000001", "adm4_name": "Rural", URBAN_RURAL_COLUMN: RURAL}, 0.0, 300.0),
            ({"psgc": "0400000002", "adm4_name": "Urban", URBAN_RURAL_COLUMN: URBAN}, 300.0, 500.0),
        ]
    )
    _, (coverage, _) = coverage_of(one_cell(), barangays)
    row = coverage.iloc[0]
    total = (
        row["rural_coverage"]
        + row["urban_coverage"]
        + row["unclassified_coverage"]
        + row["uncovered_coverage"]
    )
    assert total == pytest.approx(1.0)
    assert row["uncovered_coverage"] == pytest.approx(0.5)


def test_a_polygon_without_a_psa_record_is_unclassified_not_rural():
    barangays = make_barangays(
        [
            ({"psgc": "0400000001", "adm4_name": "Rural", URBAN_RURAL_COLUMN: RURAL}, 0.0, 700.0),
            ({"psgc": "0403418901", "adm4_name": "Forest Land", URBAN_RURAL_COLUMN: None}, 700.0, 1000.0),
        ]
    )
    overlay, (coverage, _) = coverage_of(one_cell(), barangays)
    labels = overlay.set_index("psgc")[URBAN_RURAL_COLUMN].to_dict()
    assert labels["0403418901"] == UNCLASSIFIED_LABEL
    row = coverage.iloc[0]
    assert row["rural_coverage"] == pytest.approx(0.7)
    assert row["unclassified_coverage"] == pytest.approx(0.3)
    assert row["urban_coverage"] == pytest.approx(0.0)
    assert row["rural_share_of_covered"] == pytest.approx(0.7)


def test_covered_area_shares_reproduce_the_rural_rule():
    grid = make_grid(
        [
            ("r0000c0000", 0.0, CELL_SIZE_M),
            ("r0000c0001", CELL_SIZE_M, CELL_SIZE_M),
            ("r0000c0002", 2 * CELL_SIZE_M, CELL_SIZE_M),
            ("r0000c0003", 3 * CELL_SIZE_M, CELL_SIZE_M),
        ]
    )
    barangays = make_barangays(
        [
            ({"psgc": "0400000001", "adm4_name": "Rural A", URBAN_RURAL_COLUMN: RURAL}, 0.0, 1000.0),
            ({"psgc": "0400000002", "adm4_name": "Urban A", URBAN_RURAL_COLUMN: URBAN}, 1000.0, 1600.0),
            ({"psgc": "0400000003", "adm4_name": "Rural B", URBAN_RURAL_COLUMN: RURAL}, 1600.0, 2500.0),
            ({"psgc": "0400000004", "adm4_name": "Urban B", URBAN_RURAL_COLUMN: URBAN}, 2500.0, 3000.0),
        ]
    )
    _, (coverage, _) = coverage_of(grid, barangays)
    shares = coverage.set_index(CELL_ID_COLUMN)
    assert shares.loc["r0000c0001", "rural_share_of_covered"] == pytest.approx(0.4)
    assert shares.loc["r0000c0001", "urban_share_of_covered"] == pytest.approx(0.6)
    assert shares.loc["r0000c0002", "rural_share_of_covered"] == pytest.approx(0.5)
    assert shares.loc["r0000c0000", "rural_share_of_covered"] == pytest.approx(1.0)
    assert shares.loc["r0000c0003", "uncovered_coverage"] == pytest.approx(1.0)


def test_overlapping_barangay_polygons_are_rejected():
    barangays = make_barangays(
        [
            ({"psgc": "0400000001", "adm4_name": "Whole", URBAN_RURAL_COLUMN: RURAL}, 0.0, 1000.0),
            ({"psgc": "0400000002", "adm4_name": "Overlap", URBAN_RURAL_COLUMN: URBAN}, 400.0, 900.0),
        ]
    )
    with pytest.raises(AssertionError, match="overlap each other"):
        barangay_cell_overlay(one_cell(), barangays)


def test_the_grid_cell_ids_geometry_and_columns_are_preserved():
    grid = make_grid([("r0000c0000", 0.0, CELL_SIZE_M), ("r0000c0001", CELL_SIZE_M, 400.0)])
    before = grid.geometry.to_wkb()
    overlay, (coverage, _) = coverage_of(grid, split_barangays())
    assert coverage[CELL_ID_COLUMN].tolist() == grid[CELL_ID_COLUMN].tolist()
    assert coverage.columns.tolist() == list(COVERAGE_COLUMNS)
    assert grid.geometry.to_wkb().equals(before)
    assert len(grid) == 2
    assert overlay[GRID_AREA_COLUMN].isin(grid["projected_area_m2"]).all()


def test_percentages_of_a_fully_covered_cell_sum_to_one_hundred():
    overlay, _ = coverage_of(one_cell(), split_barangays())
    assert overlay[PERCENTAGE_COLUMN].sum() == pytest.approx(100.0)
    assert overlay[PERCENTAGE_COLUMN].between(0.0, 100.0).all()
