import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from str_suitability.config import GEOGRAPHIC_CRS, PROJECTED_CRS
from str_suitability.rural.classify import (
    CELL_CLASS_COLUMN,
    CELL_ID_COLUMN,
    RURAL_CELL,
    URBAN_CELL,
    classify_grid_cells,
)
from str_suitability.rural.psa import RURAL, URBAN, URBAN_RURAL_COLUMN
from str_suitability.rural.training import (
    build_rural_training_frame,
    rural_training_cell_classes,
    select_rural_targets,
)

CLASSIFICATION_COLUMNS = [
    "listing_id",
    "matched_barangay_psgc",
    "matched_barangay_name",
    "matched_municipality",
    URBAN_RURAL_COLUMN,
    "match_status",
]


def listing_classification(rows):
    return pd.DataFrame(rows, columns=CLASSIFICATION_COLUMNS)


def cell_point(grid, index):
    row = grid.loc[grid[CELL_ID_COLUMN] == f"r0000c{index:04d}"].iloc[0]
    return float(row["longitude"]), float(row["latitude"])


def outside_point(grid):
    min_x, min_y, max_x, max_y = grid.total_bounds
    far = gpd.GeoSeries(
        [box(max_x + 5000, min_y, max_x + 6000, min_y + 1000).centroid],
        crs=PROJECTED_CRS,
    ).to_crs(GEOGRAPHIC_CRS)
    return float(far.x.iloc[0]), float(far.y.iloc[0])


def listing_row(listing_id, grid, index, municipality, revenue, outside=False):
    longitude, latitude = outside_point(grid) if outside else cell_point(grid, index)
    return {
        "listing_id": listing_id,
        "municipality": municipality,
        "longitude": longitude,
        "latitude": latitude,
        "in_training_population": True,
        "ttm_revenue": revenue,
        "ttm_occupancy": 0.1,
    }


def classification_row(listing_id, municipality, urban_rural):
    is_rural = urban_rural == RURAL
    return {
        "listing_id": listing_id,
        "matched_barangay_psgc": "0400000001" if is_rural else "0400000002",
        "matched_barangay_name": "Rural One" if is_rural else "Urban One",
        "matched_municipality": municipality,
        URBAN_RURAL_COLUMN: urban_rural,
        "match_status": "matched_rural" if is_rural else "matched_urban",
    }


def scenario(
    make_grid,
    make_barangays,
    rural_indices=(0, 1, 2, 3, 4, 5, 6, 7),
    rural_boundary=None,
    urban_cells=4,
    outside_positions=(),
):
    rural_boundary = len(rural_indices) if rural_boundary is None else rural_boundary
    total = rural_boundary + urban_cells
    assert max(rural_indices) < total, "a rural listing sits outside the grid"
    grid = make_grid(total)

    targets, classification, observations = [], [], []
    for position, index in enumerate(rural_indices):
        listing_id = f"rural_{position}"
        municipality = f"Municipality {position % 3}"
        outside = position in outside_positions
        targets.append(
            listing_row(listing_id, grid, index, municipality, 100000.0 + position, outside=outside)
        )
        classification.append(classification_row(listing_id, municipality, RURAL))
        if not outside:
            observations.append({CELL_ID_COLUMN: f"r0000c{index:04d}", "listing_id": listing_id})

    for offset in range(urban_cells):
        index = rural_boundary + offset
        listing_id = f"urban_{offset}"
        targets.append(listing_row(listing_id, grid, index, "Urban Municipality", 300000.0))
        classification.append(classification_row(listing_id, "Urban Municipality", URBAN))
        observations.append({CELL_ID_COLUMN: f"r0000c{index:04d}", "listing_id": listing_id})

    cell_classification, _ = classify_grid_cells(grid, make_barangays(rural_boundary, urban_cells))
    return (
        grid,
        pd.DataFrame(targets),
        listing_classification(classification),
        pd.DataFrame(observations),
        cell_classification,
    )


def test_rural_training_frame_is_the_rural_subset_of_the_fitted_population(
    make_grid, make_barangays
):
    grid, targets, classification, observations, _ = scenario(make_grid, make_barangays)
    merged, report = build_rural_training_frame(targets, grid, classification, observations)
    assert len(merged) == 8
    assert set(merged["listing_id"]) == {f"rural_{index}" for index in range(8)}
    assert report["all_rural_listings"] == 8
    assert report["active_rural_listings"] == 8
    assert report["fitted_rural_listings"] == 8


def test_no_urban_listing_enters_the_rural_training_set(make_grid, make_barangays):
    grid, targets, classification, observations, _ = scenario(make_grid, make_barangays)
    merged, _ = build_rural_training_frame(targets, grid, classification, observations)
    assert not merged["listing_id"].str.startswith("urban_").any()
    assert set(merged[URBAN_RURAL_COLUMN]) == {RURAL}


def test_a_rural_listing_outside_the_retained_grid_is_not_fitted(make_grid, make_barangays):
    grid, targets, classification, observations, _ = scenario(
        make_grid, make_barangays, outside_positions=(3,)
    )
    merged, report = build_rural_training_frame(targets, grid, classification, observations)
    assert "rural_3" not in set(merged["listing_id"])
    assert report["all_rural_listings"] == 8
    assert report["active_rural_listings"] == 8
    assert report["fitted_rural_listings"] == 7
    assert report["unassigned_to_cell"] == 1


def test_training_frame_mismatch_is_rejected(make_grid, make_barangays):
    grid, targets, classification, observations, _ = scenario(make_grid, make_barangays)
    trimmed = observations.loc[observations["listing_id"] != "rural_0"]
    with pytest.raises(AssertionError, match="not exactly the rural subset"):
        build_rural_training_frame(targets, grid, classification, trimmed)


def test_rural_concentration_reports_spread(make_grid, make_barangays):
    grid, targets, classification, observations, _ = scenario(make_grid, make_barangays)
    _, report = build_rural_training_frame(targets, grid, classification, observations)
    assert report["cells_with_training_listings"] == 8
    assert report["cells_holding_one_listing"] == 8
    assert report["max_listings_in_a_cell"] == 1
    assert report["municipalities_with_training_listings"] == 3
    assert sum(report["listings_per_municipality"].values()) == 8


def test_a_rural_listing_inside_a_majority_urban_cell_keeps_its_rural_label(
    make_grid, make_barangays
):
    grid, targets, classification, observations, cell_classification = scenario(
        make_grid, make_barangays, rural_indices=(0, 1, 2, 3, 4, 5, 6, 8), rural_boundary=8
    )
    cell_labels = cell_classification.set_index(CELL_ID_COLUMN)[CELL_CLASS_COLUMN]
    assert cell_labels["r0000c0008"] == URBAN_CELL

    merged, _ = build_rural_training_frame(targets, grid, classification, observations)
    assert "rural_7" in set(merged["listing_id"])

    report = rural_training_cell_classes(merged, cell_classification, observations)
    assert report["listings_in_rural_cells"] == 7
    assert report["listings_in_urban_cells"] == 1
    assert report["rural_cells_with_fitted_rural_listings"] == 7


def test_rural_training_cell_classes_counts_unobserved_rural_cells(make_grid, make_barangays):
    grid, targets, classification, observations, cell_classification = scenario(
        make_grid, make_barangays, rural_indices=(0, 1, 2, 3), rural_boundary=8, urban_cells=4
    )
    merged, _ = build_rural_training_frame(targets, grid, classification, observations)
    report = rural_training_cell_classes(merged, cell_classification, observations)
    assert report["listings_in_urban_cells"] == 0
    assert report["rural_cells_with_any_training_listing"] == 4
    assert report["rural_cells_without_any_training_listing"] == 4


def test_select_rural_targets_drops_urban_and_unmatched(make_grid, make_barangays):
    grid, targets, classification, observations, _ = scenario(make_grid, make_barangays)
    unmatched = pd.DataFrame(
        [
            {
                "listing_id": "unmatched_0",
                "matched_barangay_psgc": None,
                "matched_barangay_name": None,
                "matched_municipality": "Pakil",
                URBAN_RURAL_COLUMN: None,
                "match_status": "unmatched",
            }
        ],
        columns=CLASSIFICATION_COLUMNS,
    )
    extra = pd.DataFrame(
        [
            {
                "listing_id": "unmatched_0",
                "municipality": "Pakil",
                "longitude": 121.0,
                "latitude": 14.0,
                "in_training_population": True,
                "ttm_revenue": 5000.0,
                "ttm_occupancy": 0.05,
            }
        ]
    )
    rural = select_rural_targets(
        pd.concat([targets, extra], ignore_index=True),
        pd.concat([classification, unmatched], ignore_index=True),
    )
    assert "unmatched_0" not in set(rural["listing_id"])
    assert "urban_0" not in set(rural["listing_id"])
    assert set(rural[URBAN_RURAL_COLUMN]) == {RURAL}
    assert len(rural) == 8
