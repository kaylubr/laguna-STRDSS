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
from str_suitability.rural.suitability import (
    IN_RURAL_ANALYSIS_COLUMN,
    OUTSIDE_RURAL_ANALYSIS_LABELS,
    RURAL_CLASS_COLUMN,
    RURAL_SCORE_COLUMN,
)
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
    from str_suitability.modeling import train

    processed, interim, output, polygons, classification_json, classification_csv = write_inputs(
        tmp_path, make_grid, make_barangays
    )
    monkeypatch.setattr(config, "BARANGAY_POLYGONS_PATH", polygons)
    monkeypatch.setattr(config, "BARANGAY_CLASSIFICATION_PATH", classification_json)
    monkeypatch.setattr(config, "RURAL_URBAN_LISTINGS_PATH", classification_csv)
    monkeypatch.setattr(train, "PARAM_GRID", SMALL_PARAM_GRID)

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


def test_pipeline_predicts_the_rural_domain_only(configured):
    report, _, _ = configured
    domain = report["prediction_domain"]
    assert domain["rural_cells_predicted"] == RURAL_CELLS
    assert domain["urban_cells_excluded"] == URBAN_CELLS
    assert domain["unclassified_cells_excluded"] == 0


def test_pipeline_weights_sum_to_one(configured):
    report, _, _ = configured
    weights = report["weights"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["distance_to_listed_tourist_place"] == pytest.approx(1.0 / 3.0)
    assert weights["nearby_mean_revenue"] == pytest.approx(1.0 / 6.0)
    assert weights["nearby_mean_occupancy"] == pytest.approx(1.0 / 6.0)
    assert weights["competition_listing_count"] == pytest.approx(1.0 / 3.0)
    assert set(weights).isdisjoint(config.POI_BLOC_INDICATORS)


def test_pipeline_writes_every_rural_output(configured):
    _, output, _ = configured
    for name in (
        "grid_rural_classification.parquet",
        "training_observations.parquet",
        "predicted_revenue.parquet",
        "predicted_occupancy.parquet",
        "grid_suitability.parquet",
        "model_summary.json",
        "suitability_summary.json",
    ):
        assert (output / name).exists(), f"{name} was not written"


def test_pipeline_output_masks_urban_cells(configured):
    _, output, _ = configured
    scored = gpd.read_parquet(output / "grid_suitability.parquet")
    urban = scored[scored[CELL_CLASS_COLUMN] == URBAN_CELL]
    rural = scored[scored[CELL_CLASS_COLUMN] == RURAL_CELL]

    assert len(scored) == RURAL_CELLS + URBAN_CELLS
    assert urban[RURAL_SCORE_COLUMN].isna().all()
    assert set(urban[RURAL_CLASS_COLUMN]) == {OUTSIDE_RURAL_ANALYSIS_LABELS[URBAN_CELL]}
    assert rural[RURAL_SCORE_COLUMN].notna().all()
    assert scored[IN_RURAL_ANALYSIS_COLUMN].sum() == RURAL_CELLS


def test_pipeline_reports_both_models(configured):
    _, output, _ = configured
    summary = json.loads((output / "model_summary.json").read_text())
    assert set(summary["models"]) == {"revenue", "occupancy"}
    for name in ("revenue", "occupancy"):
        metrics = summary["models"][name]
        assert {"mae", "rmse", "r2", "n"} <= set(metrics["test_metrics"])
        assert metrics["best_params"]
        assert metrics["importance"]
    assert summary["rural_training"]["fitted_rural_listings"] == RURAL_CELLS


def test_saved_predictions_cover_only_rural_cells(configured):
    _, output, _ = configured
    classification = pd.read_parquet(output / "grid_rural_classification.parquet")
    rural_ids = set(
        classification.loc[classification[CELL_CLASS_COLUMN] == RURAL_CELL, CELL_ID_COLUMN]
    )
    outside_ids = set(
        classification.loc[classification[CELL_CLASS_COLUMN] != RURAL_CELL, CELL_ID_COLUMN]
    )
    assert rural_ids.isdisjoint(outside_ids)

    for name in ("predicted_revenue", "predicted_occupancy"):
        frame = pd.read_parquet(output / f"{name}.parquet")
        assert set(frame[CELL_ID_COLUMN]) == rural_ids
        assert len(frame) == RURAL_CELLS


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
