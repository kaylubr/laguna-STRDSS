import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from str_suitability import config
from str_suitability.rural.classify import (
    CELL_CLASS_COLUMN,
    CELL_ID_COLUMN,
    RURAL_CELL,
    URBAN_CELL,
    classify_grid_cells,
    select_cells_with_class,
)
from str_suitability.rural.pipeline import run_rural_pipeline
from str_suitability.rural.psa import RURAL, URBAN, URBAN_RURAL_COLUMN
from str_suitability.modeling.location_classifier import CLASS_COLUMN, PROBABILITY_COLUMNS
from str_suitability.rural.suitability import IN_RURAL_ANALYSIS_COLUMN
from str_suitability.suitability.stage import compute_suitability

SMALL_PARAM_GRID = {"n_estimators": [5], "max_depth": [3]}
RURAL_CELLS = 10
URBAN_CELLS = 3


def write_inputs(tmp_path, make_grid, make_barangays):
    grid = make_grid(RURAL_CELLS + URBAN_CELLS)
    barangays = make_barangays(RURAL_CELLS, URBAN_CELLS)

    processed = tmp_path / "processed"
    processed.mkdir()
    grid.to_parquet(processed / "grid_features.parquet", index=False)

    targets, observations = [], []
    for index in range(RURAL_CELLS):
        row = grid.iloc[index]
        listing_id = f"rural_{index}"
        targets.append(
            {
                "listing_id": listing_id,
                "municipality": f"Municipality {index % 3}",
                "longitude": row["longitude"],
                "latitude": row["latitude"],
                "in_training_population": True,
                "ttm_revenue": 100000.0 + 5000.0 * index,
                "ttm_occupancy": 0.05 + 0.01 * index,
            }
        )
        observations.append({CELL_ID_COLUMN: row[CELL_ID_COLUMN], "listing_id": listing_id})

    for offset in range(URBAN_CELLS):
        row = grid.iloc[RURAL_CELLS + offset]
        listing_id = f"urban_{offset}"
        targets.append(
            {
                "listing_id": listing_id,
                "municipality": "Urban Municipality",
                "longitude": row["longitude"],
                "latitude": row["latitude"],
                "in_training_population": True,
                "ttm_revenue": 400000.0,
                "ttm_occupancy": 0.3,
            }
        )
        observations.append({CELL_ID_COLUMN: row[CELL_ID_COLUMN], "listing_id": listing_id})

    pd.DataFrame(observations).to_parquet(processed / "training_observations.parquet", index=False)

    interim = tmp_path / "interim"
    interim.mkdir()
    pd.DataFrame(targets).to_parquet(interim / "airroi_targets.parquet", index=False)

    classification_rows = []
    for index in range(RURAL_CELLS):
        classification_rows.append(
            {
                "listing_id": f"rural_{index}",
                "matched_barangay_psgc": "0400000001",
                "matched_barangay_name": "Rural One",
                "matched_municipality": f"Municipality {index % 3}",
                URBAN_RURAL_COLUMN: RURAL,
                "match_status": "matched_rural",
            }
        )
    for offset in range(URBAN_CELLS):
        classification_rows.append(
            {
                "listing_id": f"urban_{offset}",
                "matched_barangay_psgc": "0400000002",
                "matched_barangay_name": "Urban One",
                "matched_municipality": "Urban Municipality",
                URBAN_RURAL_COLUMN: URBAN,
                "match_status": "matched_urban",
            }
        )
    classification_path = tmp_path / "airoi_rural_urban.csv"
    pd.DataFrame(classification_rows).to_csv(classification_path, index=False)

    polygons = barangays.copy()
    polygons["adm4_pcode"] = "PH" + polygons["psgc"]
    polygons_path = tmp_path / "laguna.geojson"
    polygons[["adm4_pcode", "adm4_name", "geometry"]].to_crs("EPSG:4326").to_file(
        polygons_path, driver="GeoJSON"
    )

    classification_json = tmp_path / "barangay_with_classification.json"
    classification_json.write_text(
        json.dumps(
            {
                "count": len(barangays),
                "results": [
                    {
                        "code": row["psgc"],
                        "area_name": row["adm4_name"],
                        "geographic_level": "Bgy",
                        URBAN_RURAL_COLUMN: row[URBAN_RURAL_COLUMN],
                    }
                    for _, row in barangays.iterrows()
                ],
            }
        )
    )

    return processed, interim, tmp_path / "rural", polygons_path, classification_json, classification_path


@pytest.fixture
def configured(tmp_path, monkeypatch, make_grid, make_barangays):
    processed, interim, output, polygons, classification_json, classification_csv = write_inputs(
        tmp_path, make_grid, make_barangays
    )
    monkeypatch.setattr(config, "BARANGAY_POLYGONS_PATH", polygons)
    monkeypatch.setattr(config, "BARANGAY_CLASSIFICATION_PATH", classification_json)
    monkeypatch.setattr(config, "RURAL_URBAN_LISTINGS_PATH", classification_csv)
    monkeypatch.setattr(config, "PARAM_GRID", SMALL_PARAM_GRID)
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
    monkeypatch.setattr(config, "PERMUTATION_REPEATS", 2)

    report = run_rural_pipeline(processed_dir=processed, output_dir=output, interim_dir=interim)
    return report, output, processed


def test_pipeline_partitions_the_grid(configured):
    report, _, _ = configured
    grid = report["grid"]
    assert grid["cells"] == RURAL_CELLS + URBAN_CELLS
    assert grid["rural_cells"] == RURAL_CELLS
    assert grid["urban_cells"] == URBAN_CELLS
    assert grid["unclassified_cells"] == 0


def test_pipeline_trains_on_rural_listings_only(configured):
    report, _, _ = configured
    training = report["rural_training"]
    assert training["all_rural_listings"] == RURAL_CELLS
    assert training["active_rural_listings"] == RURAL_CELLS
    assert training["fitted_rural_listings"] == RURAL_CELLS
    assert training["listings_in_urban_cells"] == 0
    assert training["rural_cells_with_fitted_rural_listings"] == RURAL_CELLS


def test_pipeline_scores_every_cell_from_the_classifier(configured):
    report, _, _ = configured
    domain = report["prediction_domain"]
    assert domain["cells_scored"] == RURAL_CELLS + URBAN_CELLS
    assert domain["labeled_cells"] == RURAL_CELLS + URBAN_CELLS
    assert "macro_f1" in report["test_metrics"]
    assert "balanced_accuracy" in report["baseline_metrics"]
    assert "weights" not in report


def test_pipeline_writes_every_rural_output(configured):
    _, output, _ = configured
    for name in (
        "grid_rural_classification.parquet",
        "training_observations.parquet",
        "location_training_cells.parquet",
        "grid_suitability.parquet",
        "model_summary.json",
        "suitability_summary.json",
    ):
        assert (output / name).exists(), f"{name} was not written"


def test_pipeline_scores_urban_cells_too(configured):
    _, output, _ = configured
    scored = gpd.read_parquet(output / "grid_suitability.parquet")
    urban = scored[scored[CELL_CLASS_COLUMN] == URBAN_CELL]
    rural = scored[scored[CELL_CLASS_COLUMN] == RURAL_CELL]

    assert len(scored) == RURAL_CELLS + URBAN_CELLS
    assert set(scored[CLASS_COLUMN]) <= set(config.PERFORMANCE_CLASS_LABELS)
    assert np.allclose(scored[list(PROBABILITY_COLUMNS)].sum(axis=1), 1.0)
    highest = scored[list(PROBABILITY_COLUMNS)].to_numpy().argmax(axis=1)
    expected = [config.PERFORMANCE_CLASS_LABELS[int(index)] for index in highest]
    assert scored[CLASS_COLUMN].tolist() == expected
    assert scored[IN_RURAL_ANALYSIS_COLUMN].sum() == RURAL_CELLS


def test_pipeline_reports_the_location_classifier(configured):
    _, output, _ = configured
    summary = json.loads((output / "model_summary.json").read_text())
    suitability = json.loads((output / "suitability_summary.json").read_text())
    model = summary["models"]["location_success"]
    assert suitability["weighting_method"] == "random_forest_classifier"
    assert "entropy" not in suitability["weighting_method"]
    assert {
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "confusion_matrix",
    } <= set(model["test_metrics"])
    assert "fisher_jenks" not in suitability
    assert suitability["predicted_class"] == "rf_predicted_class"
    assert model["baseline_metrics"]["n"] == model["test_metrics"]["n"]
    assert model["importance"]
    assert "importance_std" in model["importance"][0]
    assert model["test_metrics"]["n_folds"] == model["baseline_metrics"]["n_folds"]
    assert "cv" in model["test_metrics"]
    experiments = model["experiments"]
    assert experiments["prespecified_model"]["replaced_by_experiments"] is False
    assert experiments["prespecified_model"]["radius_km"] == 5.0
    assert experiments["prespecified_model"]["market_statistic"] == "mean"
    assert experiments["leaf_sizes"]["final_model_changed"] is False
    training = pd.read_parquet(output / "location_training_cells.parquet")
    assert "cell_mean_revenue" in training.columns
    assert "cell_mean_revenue" not in model["features"]
    assert "surrounding_mean_revenue" in model["features"]


def test_suitability_runs_over_exactly_the_rural_cells(make_grid, make_barangays):
    grid = make_grid(RURAL_CELLS + URBAN_CELLS)
    step = np.arange(len(grid), dtype=float)
    grid["predicted_revenue"] = 1000.0 + step * 100.0
    grid["predicted_occupancy"] = 0.1 + step * 0.01
    classification, _ = classify_grid_cells(grid, make_barangays(RURAL_CELLS, URBAN_CELLS))
    rural = select_cells_with_class(grid, classification, RURAL_CELL)
    scored, _ = compute_suitability(rural)
    assert len(scored) == RURAL_CELLS
    assert set(scored[CELL_ID_COLUMN]) == set(rural[CELL_ID_COLUMN])
    assert set(scored[CELL_ID_COLUMN]).isdisjoint(
        set(grid[CELL_ID_COLUMN]) - set(rural[CELL_ID_COLUMN])
    )
