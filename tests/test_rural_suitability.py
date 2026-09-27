import numpy as np
import pandas as pd
import pytest

from str_suitability.rural.classify import (
    CELL_CLASS_COLUMN,
    CELL_ID_COLUMN,
    RURAL_CELL,
    UNCLASSIFIED_CELL,
    URBAN_CELL,
    classify_grid_cells,
    select_cells_with_class,
)
from str_suitability.rural.suitability import (
    IN_RURAL_ANALYSIS_COLUMN,
    OUTSIDE_RURAL_ANALYSIS_LABELS,
    RURAL_CLASS_COLUMN,
    RURAL_CLASS_LABELS,
    RURAL_SCORE_COLUMN,
    assemble_rural_output,
    compute_rural_suitability,
    rural_class_lookup,
)
from str_suitability.suitability.classify import CLASS_LABELS

REVENUE = [10.0, 20.0, 30.0, 45.0, 50.0, 60.0, 75.0, 80.0, 90.0, 100.0, 110.0, 120.0]
OCCUPANCY = [0.5, 0.1, 0.8, 0.2, 0.6, 0.3, 0.9, 0.05, 0.7, 0.4, 0.25, 0.6]
POI = [0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 5.0, 0.0, 1.0, 0.0, 3.0, 0.0]
ATTRACTION = [5.0, 1.0, 9.0, 2.0, 7.0, 3.0, 10.0, 0.5, 8.0, 4.0, 6.0, 1.5]
TRANSPORT = [3.0, 1.0, 8.0, 2.0, 6.0, 4.0, 9.0, 0.2, 7.0, 5.0, 2.5, 1.1]


def scoring_grid(make_grid):
    grid = make_grid(12)
    grid["predicted_revenue"] = REVENUE
    grid["predicted_occupancy"] = OCCUPANCY
    grid["poi_density_total"] = POI
    grid["distance_to_nearest_tourist_attraction"] = ATTRACTION
    grid["distance_to_nearest_transportation_facility"] = TRANSPORT
    return grid


def domain(make_grid, make_barangays, rural_cells=8, urban_cells=4):
    grid = scoring_grid(make_grid)
    classification, _ = classify_grid_cells(grid, make_barangays(rural_cells, urban_cells))
    rural = select_cells_with_class(grid, classification, RURAL_CELL)
    return grid, classification, rural


def test_rural_labels_are_the_five_ordered_bands():
    assert list(RURAL_CLASS_LABELS) == [
        "Rural — Very Low",
        "Rural — Low",
        "Rural — Moderate",
        "Rural — High",
        "Rural — Very High",
    ]
    lookup = rural_class_lookup()
    assert list(lookup) == list(CLASS_LABELS)
    assert lookup["Very High Suitability"] == "Rural — Very High"
    assert lookup["Very Low Suitability"] == "Rural — Very Low"
    assert lookup["Moderate Suitability"] == "Rural — Moderate"


def test_the_highest_rural_score_takes_the_highest_rural_class(make_grid, make_barangays):
    _, _, rural = domain(make_grid, make_barangays)
    scored, _ = compute_rural_suitability(rural)
    ranked = scored.sort_values(RURAL_SCORE_COLUMN)
    assert ranked[RURAL_CLASS_COLUMN].iloc[0] == "Rural — Very Low"
    assert ranked[RURAL_CLASS_COLUMN].iloc[-1] == "Rural — Very High"


def test_rural_suitability_scores_only_the_rural_cells(make_grid, make_barangays):
    _, _, rural = domain(make_grid, make_barangays)
    scored, report = compute_rural_suitability(rural)
    assert len(scored) == len(rural)
    assert set(scored[CELL_ID_COLUMN]) == set(rural[CELL_ID_COLUMN])
    assert scored[RURAL_SCORE_COLUMN].notna().all()
    assert set(scored[RURAL_CLASS_COLUMN]) <= set(RURAL_CLASS_LABELS)
    assert report["rural_cells_scored"] == len(rural)


def test_rural_weights_sum_to_one_over_the_five_indicators(make_grid, make_barangays):
    _, _, rural = domain(make_grid, make_barangays)
    _, report = compute_rural_suitability(rural)
    assert set(report["weights"]) == {
        "predicted_revenue",
        "predicted_occupancy",
        "poi_density_total",
        "distance_to_nearest_tourist_attraction",
        "distance_to_nearest_transportation_facility",
    }
    assert sum(report["weights"].values()) == pytest.approx(1.0)
    assert report["weight_sum"] == pytest.approx(1.0)


def test_rural_weights_differ_from_weights_taken_over_every_cell(make_grid, make_barangays):
    grid, _, rural = domain(make_grid, make_barangays)
    _, rural_report = compute_rural_suitability(rural)
    _, all_report = compute_rural_suitability(grid)
    rural_weights = pd.Series(rural_report["weights"])
    all_weights = pd.Series(all_report["weights"])
    assert not np.allclose(rural_weights.to_numpy(), all_weights.to_numpy())


def test_assembly_scores_rural_cells_and_masks_the_rest(make_grid, make_barangays):
    grid, classification, rural = domain(make_grid, make_barangays, rural_cells=8, urban_cells=3)
    scored, _ = compute_rural_suitability(rural)
    output, report = assemble_rural_output(grid, classification, scored)

    assert len(output) == len(grid)
    assert output[CELL_ID_COLUMN].tolist() == grid[CELL_ID_COLUMN].tolist()
    assert report["cells"] == len(grid)
    assert report["cells_urban"] == 3
    assert report["cells_unclassified"] == 1

    rural_rows = output[output[CELL_CLASS_COLUMN] == RURAL_CELL]
    urban_rows = output[output[CELL_CLASS_COLUMN] == URBAN_CELL]
    unclassified_rows = output[output[CELL_CLASS_COLUMN] == UNCLASSIFIED_CELL]

    assert rural_rows[RURAL_SCORE_COLUMN].notna().all()
    assert rural_rows[RURAL_CLASS_COLUMN].isin(RURAL_CLASS_LABELS).all()
    assert urban_rows[RURAL_SCORE_COLUMN].isna().all()
    assert unclassified_rows[RURAL_SCORE_COLUMN].isna().all()
    assert set(urban_rows[RURAL_CLASS_COLUMN]) == {OUTSIDE_RURAL_ANALYSIS_LABELS[URBAN_CELL]}
    assert set(unclassified_rows[RURAL_CLASS_COLUMN]) == {
        OUTSIDE_RURAL_ANALYSIS_LABELS[UNCLASSIFIED_CELL]
    }


def test_urban_cells_are_never_given_a_rural_class(make_grid, make_barangays):
    grid, classification, rural = domain(make_grid, make_barangays)
    scored, _ = compute_rural_suitability(rural)
    output, report = assemble_rural_output(grid, classification, scored)
    outside = output[output[CELL_CLASS_COLUMN] != RURAL_CELL]
    assert not outside[RURAL_CLASS_COLUMN].isin(RURAL_CLASS_LABELS).any()
    assert report["class_counts"][OUTSIDE_RURAL_ANALYSIS_LABELS[URBAN_CELL]] == 4
    assert report["cells_in_rural_analysis"] == 8
    assert report["cells_outside_rural_analysis"] == 4


def test_assembly_marks_the_rural_analytical_domain(make_grid, make_barangays):
    grid, classification, rural = domain(make_grid, make_barangays)
    scored, _ = compute_rural_suitability(rural)
    output, _ = assemble_rural_output(grid, classification, scored)
    assert output[IN_RURAL_ANALYSIS_COLUMN].sum() == 8
    assert output.loc[output[IN_RURAL_ANALYSIS_COLUMN], RURAL_SCORE_COLUMN].notna().all()


def test_assembly_output_keeps_geometry_and_crs(make_grid, make_barangays):
    grid, classification, rural = domain(make_grid, make_barangays)
    scored, _ = compute_rural_suitability(rural)
    output, _ = assemble_rural_output(grid, classification, scored)
    assert output.crs == grid.crs
    assert output.geometry.notna().all()
    assert output.geometry.to_wkb().equals(grid.geometry.to_wkb())


def test_rural_scores_are_bounded_by_the_weights(make_grid, make_barangays):
    _, _, rural = domain(make_grid, make_barangays)
    scored, report = compute_rural_suitability(rural)
    assert scored[RURAL_SCORE_COLUMN].between(0.0, 1.0).all()
    assert report["score_min"] >= 0.0
    assert report["score_max"] <= 1.0
