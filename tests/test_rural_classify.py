import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from str_suitability.config import PROJECTED_CRS
from str_suitability.rural.classify import (
    CELL_CLASS_COLUMN,
    CELL_CLASSIFICATION_COLUMNS,
    CELL_ID_COLUMN,
    RURAL_CELL,
    UNCLASSIFIED_CELL,
    URBAN_CELL,
    attach_listing_classification,
    classify_grid_cells,
    load_listing_classification,
    rural_population_levels,
    select_cells_with_class,
)
from str_suitability.rural.psa import RURAL, URBAN, URBAN_RURAL_COLUMN


def mixed_barangays() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "psgc": ["0400000001", "0400000002", "0400000003", "0400000004"],
            "adm4_name": ["Rural A", "Urban A", "Rural B", "Urban B"],
            URBAN_RURAL_COLUMN: [RURAL, URBAN, RURAL, URBAN],
            "geometry": [
                box(0, 0, 1000, 1000),
                box(1000, 0, 1600, 1000),
                box(1600, 0, 2500, 1000),
                box(2500, 0, 3000, 1000),
            ],
        },
        geometry="geometry",
        crs=PROJECTED_CRS,
    )


def test_cells_are_classified_by_majority_rural_area(make_grid):
    classification, report = classify_grid_cells(make_grid(4), mixed_barangays())
    classes = classification.set_index(CELL_ID_COLUMN)[CELL_CLASS_COLUMN].to_dict()
    assert classes == {
        "r0000c0000": RURAL_CELL,
        "r0000c0001": URBAN_CELL,
        "r0000c0002": RURAL_CELL,
        "r0000c0003": UNCLASSIFIED_CELL,
    }
    assert report["rural_cells"] == 2
    assert report["urban_cells"] == 1
    assert report["unclassified_cells"] == 1


def test_area_shares_are_over_the_barangay_covered_area(make_grid):
    classification, _ = classify_grid_cells(make_grid(4), mixed_barangays())
    shares = classification.set_index(CELL_ID_COLUMN)
    assert shares.loc["r0000c0001", "rural_area_share"] == pytest.approx(0.4)
    assert shares.loc["r0000c0001", "urban_area_share"] == pytest.approx(0.6)
    assert shares.loc["r0000c0002", "rural_area_share"] == pytest.approx(0.5)
    assert shares.loc["r0000c0000", "rural_area_share"] == pytest.approx(1.0)
    assert shares.loc["r0000c0000", "urban_area_share"] == pytest.approx(0.0)


def test_exactly_half_rural_area_is_rural(make_grid):
    classification, _ = classify_grid_cells(make_grid(4), mixed_barangays())
    assert (
        classification.set_index(CELL_ID_COLUMN).loc["r0000c0002", CELL_CLASS_COLUMN] == RURAL_CELL
    )


def test_a_cell_below_the_threshold_is_urban(make_grid):
    classification, _ = classify_grid_cells(make_grid(4), mixed_barangays())
    assert (
        classification.set_index(CELL_ID_COLUMN).loc["r0000c0001", CELL_CLASS_COLUMN] == URBAN_CELL
    )


def test_a_cell_no_barangay_overlaps_is_unclassified_without_shares(make_grid):
    classification, _ = classify_grid_cells(make_grid(4), mixed_barangays())
    row = classification.set_index(CELL_ID_COLUMN).loc["r0000c0003"]
    assert row[CELL_CLASS_COLUMN] == UNCLASSIFIED_CELL
    assert pd.isna(row["rural_area_share"])
    assert pd.isna(row["urban_area_share"])


def test_dominant_barangay_is_the_largest_overlap(make_grid):
    classification, _ = classify_grid_cells(make_grid(4), mixed_barangays())
    row = classification.set_index(CELL_ID_COLUMN).loc["r0000c0001"]
    assert row["dominant_barangay_psgc"] == "0400000002"
    assert row["dominant_barangay"] == "Urban A"
    assert row["dominant_barangay_urban_rural"] == URBAN


def test_classification_keeps_the_grid_cell_ids_and_geometry(make_grid):
    grid = make_grid(4)
    before = grid.geometry.to_wkb()
    classification, _ = classify_grid_cells(grid, mixed_barangays())
    assert classification.columns.tolist() == CELL_CLASSIFICATION_COLUMNS
    assert classification[CELL_ID_COLUMN].tolist() == grid[CELL_ID_COLUMN].tolist()
    assert grid.geometry.to_wkb().equals(before)
    assert len(grid) == 4


def test_shares_stay_within_the_unit_interval_and_sum_within_one(make_grid):
    classification, _ = classify_grid_cells(make_grid(4), mixed_barangays())
    covered = classification["rural_area_share"].notna()
    assert classification.loc[covered, "rural_area_share"].between(0.0, 1.0).all()
    assert classification.loc[covered, "urban_area_share"].between(0.0, 1.0).all()
    assert (
        classification.loc[covered, "rural_area_share"]
        + classification.loc[covered, "urban_area_share"]
    ).le(1.0 + 1e-9).all()


def test_threshold_is_configurable(make_grid):
    classification, _ = classify_grid_cells(make_grid(4), mixed_barangays(), threshold=0.75)
    classes = classification.set_index(CELL_ID_COLUMN)[CELL_CLASS_COLUMN].to_dict()
    assert classes["r0000c0002"] == URBAN_CELL


def test_select_cells_with_class_returns_that_class_only(make_grid):
    grid = make_grid(4)
    classification, _ = classify_grid_cells(grid, mixed_barangays())
    rural = select_cells_with_class(grid, classification, RURAL_CELL)
    assert set(rural[CELL_ID_COLUMN]) == {"r0000c0000", "r0000c0002"}
    assert len(rural) == 2


def write_classification_csv(tmp_path, rows):
    path = tmp_path / "airoi_rural_urban.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


LISTING_ROWS = [
    {
        "listing_id": "a",
        "matched_barangay_psgc": "0400000001",
        "matched_barangay_name": "Rural A",
        "matched_municipality": "Bay",
        URBAN_RURAL_COLUMN: RURAL,
        "match_status": "matched_rural",
    },
    {
        "listing_id": "b",
        "matched_barangay_psgc": "0400000002",
        "matched_barangay_name": "Urban A",
        "matched_municipality": "Bay",
        URBAN_RURAL_COLUMN: URBAN,
        "match_status": "matched_urban",
    },
    {
        "listing_id": "c",
        "matched_barangay_psgc": None,
        "matched_barangay_name": None,
        "matched_municipality": "Pakil",
        URBAN_RURAL_COLUMN: None,
        "match_status": "unmatched",
    },
]


def test_listing_classification_reports_rural_urban_and_unmatched(tmp_path):
    frame, report = load_listing_classification(write_classification_csv(tmp_path, LISTING_ROWS))
    assert report == {
        "listings": 3,
        "rural_listings": 1,
        "urban_listings": 1,
        "unmatched_listings": 1,
    }
    assert frame["listing_id"].tolist() == ["a", "b", "c"]


def test_listing_classification_rejects_an_unknown_label(tmp_path):
    rows = [dict(LISTING_ROWS[0], **{URBAN_RURAL_COLUMN: "Mixed"})]
    with pytest.raises(AssertionError, match="must be"):
        load_listing_classification(write_classification_csv(tmp_path, rows))


def test_listing_classification_rejects_duplicate_listing_ids(tmp_path):
    rows = [LISTING_ROWS[0], dict(LISTING_ROWS[0])]
    with pytest.raises(AssertionError, match="repeats a listing_id"):
        load_listing_classification(write_classification_csv(tmp_path, rows))


def test_attach_requires_the_classification_to_cover_the_listings(tmp_path):
    classification, _ = load_listing_classification(
        write_classification_csv(tmp_path, LISTING_ROWS[:2])
    )
    targets = pd.DataFrame({"listing_id": ["a", "b", "c"]})
    with pytest.raises(AssertionError, match="does not cover exactly"):
        attach_listing_classification(targets, classification)


def test_attach_never_overwrites_an_existing_label(tmp_path):
    classification, _ = load_listing_classification(
        write_classification_csv(tmp_path, LISTING_ROWS[:2])
    )
    targets = pd.DataFrame({"listing_id": ["a", "b"], URBAN_RURAL_COLUMN: [RURAL, RURAL]})
    with pytest.raises(AssertionError, match="must remain the only source"):
        attach_listing_classification(targets, classification)


def test_attach_adds_the_psa_label_when_absent(tmp_path):
    classification, _ = load_listing_classification(
        write_classification_csv(tmp_path, LISTING_ROWS[:2])
    )
    targets = pd.DataFrame({"listing_id": ["a", "b"]})
    attached = attach_listing_classification(targets, classification)
    assert attached[URBAN_RURAL_COLUMN].tolist() == [RURAL, URBAN]


def make_targets(rows):
    return pd.DataFrame(rows)


def test_population_levels_narrow_from_all_to_active_to_fitted():
    targets = make_targets(
        [
            {"listing_id": "a", "municipality": "Bay", URBAN_RURAL_COLUMN: RURAL, "in_training_population": True},
            {"listing_id": "b", "municipality": "Bay", URBAN_RURAL_COLUMN: RURAL, "in_training_population": True},
            {"listing_id": "c", "municipality": "Cavinti", URBAN_RURAL_COLUMN: RURAL, "in_training_population": False},
            {"listing_id": "d", "municipality": "Bay", URBAN_RURAL_COLUMN: URBAN, "in_training_population": True},
        ]
    )
    observations = pd.DataFrame(
        {CELL_ID_COLUMN: ["cell_1", "cell_2"], "listing_id": ["a", "b"]}
    )
    levels = rural_population_levels(targets, observations)
    assert levels["all_rural"]["listings"] == 3
    assert levels["active_rural"]["listings"] == 2
    assert levels["fitted_rural"]["listings"] == 2
    assert levels["fitted_rural"]["cells"] == 2
    assert levels["all_rural"]["municipalities"] == 2
    assert levels["active_rural"]["municipalities"] == 1


def test_population_levels_drop_rural_listings_without_a_cell():
    targets = make_targets(
        [
            {"listing_id": "a", "municipality": "Bay", URBAN_RURAL_COLUMN: RURAL, "in_training_population": True},
            {"listing_id": "b", "municipality": "Bay", URBAN_RURAL_COLUMN: RURAL, "in_training_population": True},
        ]
    )
    observations = pd.DataFrame({CELL_ID_COLUMN: ["cell_1"], "listing_id": ["a"]})
    levels = rural_population_levels(targets, observations)
    assert levels["active_rural"]["listings"] == 2
    assert levels["fitted_rural"]["listings"] == 1
