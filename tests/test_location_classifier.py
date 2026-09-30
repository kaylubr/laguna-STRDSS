import inspect

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from str_suitability import config
from str_suitability.config import GEOGRAPHIC_CRS, PERFORMANCE_CLASS_LABELS
from str_suitability.modeling.experiments import _columns, run_location_experiments
from str_suitability.modeling.location_classifier import (
    CELL_OCCUPANCY,
    CELL_REVENUE,
    CLASSIFIER_FEATURES,
    CLASS_COLUMN,
    MEDIAN_MARKET_FEATURES,
    PERFORMANCE_CLASS,
    PERFORMANCE_SCORE,
    PROBABILITY_COLUMNS,
    SURROUNDING_MEDIAN_REVENUE,
    SURROUNDING_OCCUPANCY,
    SURROUNDING_REVENUE,
    _group_splits,
    _holdout_split,
    label_performance,
    market_features,
    score_cells,
    screen_grade,
    train_location_classifier,
)
from str_suitability.rural import pipeline as rural_pipeline


def _cell(cell_id: str, longitude: float, latitude: float, municipality: str) -> dict:
    return {
        "cell_id": cell_id,
        "municipality": municipality,
        "longitude": longitude,
        "latitude": latitude,
        "geometry": box(longitude - 0.004, latitude - 0.004, longitude + 0.004, latitude + 0.004),
    }


def test_surrounding_means_exclude_the_cell_and_stay_missing_when_empty():
    grid = gpd.GeoDataFrame(
        [
            _cell("isolated", 121.20, 14.20, "A"),
            _cell("host", 121.50, 14.20, "B"),
            _cell("neighbor", 121.52, 14.20, "C"),
            _cell("empty", 121.80, 14.20, "D"),
        ],
        geometry="geometry",
        crs=GEOGRAPHIC_CRS,
    )
    listings = pd.DataFrame(
        [
            {"longitude": 121.20, "latitude": 14.20, "ttm_revenue": 900_000.0, "ttm_occupancy": 0.9},
            {"longitude": 121.50, "latitude": 14.20, "ttm_revenue": 800_000.0, "ttm_occupancy": 0.8},
            {"longitude": 121.52, "latitude": 14.20, "ttm_revenue": 40_000.0, "ttm_occupancy": 0.1},
        ]
    )
    places = pd.DataFrame(
        [
            {"longitude": 121.20, "latitude": 14.20, "name": "Falls"},
            {"longitude": 121.50, "latitude": 14.20, "name": "Lake"},
        ]
    )

    features = market_features(grid, listings, places, radius_km=5.0).set_index("cell_id")

    assert pd.isna(features.loc["isolated", SURROUNDING_REVENUE])
    assert pd.isna(features.loc["isolated", SURROUNDING_OCCUPANCY])
    assert features.loc["isolated", "surrounding_listing_count"] == 0
    assert features.loc["host", SURROUNDING_REVENUE] == pytest.approx(40_000.0)
    assert features.loc["host", CELL_REVENUE] == pytest.approx(800_000.0)
    assert set(CLASSIFIER_FEATURES).isdisjoint({CELL_REVENUE, CELL_OCCUPANCY, PERFORMANCE_SCORE})

    labeled, report = label_performance(features.reset_index())
    assert "empty" not in set(labeled["cell_id"])
    assert report["cells_without_a_label"] == 1
    assert set(report["class_counts"]) == set(PERFORMANCE_CLASS_LABELS)


def test_performance_classes_follow_the_percentile_cuts():
    frame = pd.DataFrame(
        {
            "cell_id": [f"c{index}" for index in range(8)],
            "listings_in_cell": [1] * 8,
            CELL_REVENUE: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            CELL_OCCUPANCY: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "municipality": [f"m{index}" for index in range(8)],
        }
    )
    labeled, report = label_performance(frame)
    classes = labeled.set_index("cell_id")[PERFORMANCE_CLASS]
    assert list(PERFORMANCE_CLASS_LABELS) == ["Low", "Moderate", "High"]
    assert classes["c0"] == 0
    assert classes["c7"] == 2
    assert classes["c3"] == 1
    assert set(classes) == {0, 1, 2}
    assert report["low_cutoff"] < report["high_cutoff"]
    assert "min-max" in report["normalization"]


def test_predicted_class_is_the_highest_of_three_probabilities(monkeypatch):
    monkeypatch.setattr(
        config,
        "CLASSIFIER_PARAMS",
        {
            "n_estimators": 20,
            "max_depth": 4,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "min_samples_split": 2,
        },
    )
    cells, listings = [], []
    for index in range(12):
        longitude = 121.2 + index * 0.08
        cells.append(_cell(f"c{index}", longitude, 14.2, f"m{index % 4}"))
        listings.append(
            {
                "longitude": longitude,
                "latitude": 14.2,
                "ttm_revenue": 10_000.0 * (index + 1),
                "ttm_occupancy": 0.05 * (index + 1),
            }
        )
    grid = gpd.GeoDataFrame(cells, geometry="geometry", crs=GEOGRAPHIC_CRS)
    places = pd.DataFrame([{"longitude": 121.2, "latitude": 14.2, "name": "Falls"}])
    features = market_features(grid, pd.DataFrame(listings), places, radius_km=5.0)
    labeled, _ = label_performance(features)
    fitted = train_location_classifier(labeled)
    assert list(fitted["model"].classes_) == [0, 1, 2]
    scored = score_cells(fitted["model"], features)
    probabilities = scored[list(PROBABILITY_COLUMNS)].to_numpy()
    assert probabilities.shape[1] == 3
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    expected = [PERFORMANCE_CLASS_LABELS[int(index)] for index in probabilities.argmax(axis=1)]
    assert scored[CLASS_COLUMN].tolist() == expected
    train_groups = set(labeled.iloc[: fitted["train_cells"]]["municipality"])
    assert fitted["scheme"].startswith("stratified_group_kfold") or fitted["scheme"] == "group_shuffle"


def test_median_market_differs_from_the_mean_and_stays_missing_when_empty():
    grid = gpd.GeoDataFrame(
        [
            _cell("host", 121.50, 14.20, "B"),
            _cell("empty", 121.80, 14.20, "D"),
        ],
        geometry="geometry",
        crs=GEOGRAPHIC_CRS,
    )
    listings = pd.DataFrame(
        [
            {"longitude": 121.50, "latitude": 14.20, "ttm_revenue": 100_000.0, "ttm_occupancy": 0.4},
            {"longitude": 121.51, "latitude": 14.20, "ttm_revenue": 10_000.0, "ttm_occupancy": 0.1},
            {"longitude": 121.52, "latitude": 14.20, "ttm_revenue": 20_000.0, "ttm_occupancy": 0.2},
            {"longitude": 121.53, "latitude": 14.20, "ttm_revenue": 1_000_000.0, "ttm_occupancy": 0.9},
        ]
    )
    places = pd.DataFrame([{"longitude": 121.50, "latitude": 14.20, "name": "Lake"}])
    features = market_features(grid, listings, places, radius_km=5.0).set_index("cell_id")

    assert features.loc["host", SURROUNDING_REVENUE] != pytest.approx(
        features.loc["host", SURROUNDING_MEDIAN_REVENUE]
    )
    assert pd.isna(features.loc["empty", SURROUNDING_MEDIAN_REVENUE])
    assert pd.isna(features.loc["empty", "surrounding_median_occupancy"])
    assert set(_columns("all_signals", "mean")).isdisjoint(MEDIAN_MARKET_FEATURES)
    assert set(_columns("all_signals", "median")).isdisjoint(
        {SURROUNDING_REVENUE, SURROUNDING_OCCUPANCY}
    )
    assert CELL_REVENUE not in _columns("all_signals", "median")


def test_grouped_validation_uses_every_fold():
    frame = pd.DataFrame({"feature": range(15)})
    target = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    groups = pd.Series([f"m{index}" for index in range(15)])
    folds, scheme = _group_splits(frame, target, groups)
    assert scheme == "stratified_group_kfold_5"
    assert len(folds) == 5
    test_rows = []
    for train_index, test_index in folds:
        assert set(groups.iloc[train_index]).isdisjoint(set(groups.iloc[test_index]))
        test_rows.extend(test_index.tolist())
    assert sorted(test_rows) == list(range(15))


def test_experiments_keep_the_prespecified_model(monkeypatch):
    monkeypatch.setattr(
        config,
        "CLASSIFIER_PARAMS",
        {
            "n_estimators": 8,
            "max_depth": 3,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "min_samples_split": 2,
        },
    )
    cells, listings = [], []
    for index in range(15):
        longitude = 121.1 + index * 0.05
        cells.append(_cell(f"c{index}", longitude, 14.2, f"m{index}"))
        listings.append(
            {
                "longitude": longitude,
                "latitude": 14.2,
                "ttm_revenue": 20_000.0 * (index + 1),
                "ttm_occupancy": 0.04 * (index + 1),
            }
        )
    grid = gpd.GeoDataFrame(cells, geometry="geometry", crs=GEOGRAPHIC_CRS)
    places = pd.DataFrame([{"longitude": 121.1, "latitude": 14.2, "name": "Falls"}])
    features = market_features(grid, pd.DataFrame(listings), places, radius_km=5.0)
    labeled = features.copy()
    labeled[PERFORMANCE_CLASS] = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    fitted = train_location_classifier(labeled)
    experiments = run_location_experiments(
        grid, pd.DataFrame(listings), places, labeled, fitted["test_metrics"]
    )

    assert experiments["n_folds"] == 5
    assert experiments["prespecified_model"]["replaced_by_experiments"] is False
    assert experiments["prespecified_model"]["radius_km"] == 5.0
    assert experiments["prespecified_model"]["market_statistic"] == "mean"
    assert experiments["prespecified_model"]["signals"] == "all_signals"
    assert experiments["prespecified_model"]["min_samples_leaf"] == 1
    assert [row["name"] for row in experiments["signal_groups"]] == [
        "tourist",
        "market_competition",
        "all_signals",
    ]
    assert [row["radius_km"] for row in experiments["radii"]] == [1.0, 3.0, 5.0]
    mean_features = next(row["features"] for row in experiments["market_statistics"] if row["name"] == "mean")
    median_features = next(
        row["features"] for row in experiments["market_statistics"] if row["name"] == "median"
    )
    assert "surrounding_mean_revenue" in mean_features
    assert "surrounding_median_revenue" not in mean_features
    assert "surrounding_median_revenue" in median_features
    assert "surrounding_mean_revenue" not in median_features
    assert experiments["leaf_sizes"]["outer_test_used_for_selection"] is False
    assert experiments["leaf_sizes"]["final_model_changed"] is False
    assert "macro_f1" in experiments["tourist_minus_market_competition"]["difference"]
    for row in (*experiments["signal_groups"], *experiments["radii"], *experiments["market_statistics"]):
        assert CELL_REVENUE not in row["features"]
        assert CELL_OCCUPANCY not in row["features"]
    assert fitted["test_metrics"]["n_folds"] == 5
    assert "std" in fitted["test_metrics"]["cv"]["macro_f1"]
    assert fitted["importance"]
    assert "importance_std" in fitted["importance"][0]


def test_screen_grade_gives_half_credit_to_a_neighbor_and_none_to_an_opposite_end():
    forest = [
        [14, 38, 20],
        [28, 65, 51],
        [21, 26, 25],
    ]
    guess = [
        [22, 27, 23],
        [25, 85, 34],
        [21, 37, 14],
    ]
    graded = screen_grade(forest)
    frequency = screen_grade(guess)
    assert graded["exact"] == 104
    assert graded["adjacent"] == 143
    assert graded["reversals"] == 41
    assert graded["low_called_high"] == 20
    assert graded["high_called_low"] == 21
    assert graded["screen_credit"] == pytest.approx(175.5 / 288)
    assert graded["strong_kept"] == 51
    assert graded["true_high"] == 72
    assert frequency["reversals"] == 44
    assert frequency["screen_credit"] == pytest.approx(182.5 / 288)
    assert frequency["strong_kept"] == 51


def test_holdout_keeps_each_municipality_on_one_side():
    frame = pd.DataFrame({"feature": range(12)})
    target = pd.Series([0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2])
    groups = pd.Series([f"m{index}" for index in range(12)])
    train_index, test_index, scheme = _holdout_split(frame, target, groups)
    assert scheme.startswith("stratified_group_kfold")
    assert set(groups.iloc[train_index]).isdisjoint(set(groups.iloc[test_index]))


def test_final_pipeline_does_not_call_entropy_or_five_class_jenks():
    source = inspect.getsource(rural_pipeline.run_rural_pipeline)
    assert "classify_probabilities" not in source
    assert "FisherJenks" not in source
    assert "compute_suitability" not in source
    assert "label_performance" in source
