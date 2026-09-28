import json

import geopandas as gpd
import numpy as np
import pandas as pd

from str_suitability import config
from str_suitability.modeling.experiments import run_location_experiments
from str_suitability.modeling.location_classifier import (
    CLASS_COLUMN,
    PROBABILITY_COLUMNS,
    label_performance,
    market_features,
    score_cells,
    train_location_classifier,
)
from str_suitability.rural.diagrams import DIAGRAM_PATH_NAME, write_cell_diagrams
from str_suitability.rural.classify import (
    CELL_CLASS_COLUMN,
    CELL_ID_COLUMN,
    RURAL_CELL,
    attach_listing_classification,
    classify_grid_cells,
    load_listing_classification,
    rural_population_levels,
)
from str_suitability.rural.psa import (
    classify_barangay_polygons,
    load_barangay_classification,
    load_barangay_polygons,
    report_barangay_classification,
)
from str_suitability.rural.site_score import load_listed_tourist_places
from str_suitability.rural.suitability import IN_RURAL_ANALYSIS_COLUMN, attach_training_support
from str_suitability.rural.training import build_rural_training_frame, rural_training_cell_classes

GRID_FILENAME = "grid_features.parquet"
TARGETS_FILENAME = "airroi_targets.parquet"
OBSERVATIONS_FILENAME = "training_observations.parquet"
CELL_CLASSIFICATION_FILENAME = "grid_rural_classification.parquet"
SUITABILITY_FILENAME = "grid_suitability.parquet"
MODEL_SUMMARY_FILENAME = "model_summary.json"
SUITABILITY_SUMMARY_FILENAME = "suitability_summary.json"

LOCATION_TRAINING_FILENAME = "location_training_cells.parquet"


def run_rural_pipeline(
    processed_dir=None, output_dir=None, interim_dir=None
) -> dict[str, object]:
    processed_dir = config.PROCESSED_DIR if processed_dir is None else processed_dir
    output_dir = config.RURAL_PROCESSED_DIR if output_dir is None else output_dir
    interim_dir = config.INTERIM_DIR if interim_dir is None else interim_dir

    grid = gpd.read_parquet(processed_dir / GRID_FILENAME)
    targets = pd.read_parquet(interim_dir / TARGETS_FILENAME)
    observations = pd.read_parquet(processed_dir / OBSERVATIONS_FILENAME)

    classification = load_barangay_classification(config.BARANGAY_CLASSIFICATION_PATH)
    barangays = classify_barangay_polygons(
        load_barangay_polygons(config.BARANGAY_POLYGONS_PATH), classification
    )
    barangay_report = report_barangay_classification(barangays, classification)

    cell_classification, cell_report = classify_grid_cells(grid, barangays)
    listing_classification, listing_report = load_listing_classification(
        config.RURAL_URBAN_LISTINGS_PATH
    )
    levels = rural_population_levels(
        attach_listing_classification(targets, listing_classification), observations
    )

    merged, training_report = build_rural_training_frame(
        targets, grid, listing_classification, observations
    )
    training_report.update(
        rural_training_cell_classes(merged, cell_classification, observations)
    )
    active_listings = targets.loc[targets["in_training_population"]].copy()
    places = load_listed_tourist_places(config.LISTED_TOURIST_PLACES_PATH)
    features = market_features(grid, active_listings, places)
    labeled, label_report = label_performance(features)
    fitted = train_location_classifier(labeled)
    experiments = run_location_experiments(
        grid, active_listings, places, labeled, fitted["test_metrics"]
    )
    scored = score_cells(fitted["model"], features)
    output = _assemble_scored_grid(grid, cell_classification, scored)
    output, support_report = attach_training_support(output, active_listings)
    write_cell_diagrams(output, fitted["model"], output_dir / DIAGRAM_PATH_NAME)
    print(
        "location classifier: "
        f"{label_report['class_counts']} of {label_report['labeled_cells']} labeled cells; "
        f"mean macro F1 {fitted['test_metrics']['macro_f1']} "
        f"across {fitted['test_metrics']['n_folds']} municipality folds"
    )

    prediction_report = {
        "cells_scored": int(len(output)),
        "labeled_cells": int(label_report["labeled_cells"]),
        "cells_without_a_label": int(label_report["cells_without_a_label"]),
        "cells_with_no_surrounding_listings": int((features["surrounding_listing_count"] == 0).sum()),
    }
    assert prediction_report["cells_scored"] == cell_report["cells"], (
        "the suitability model did not score every grid cell"
    )
    predicted_counts = {
        label: int((output[CLASS_COLUMN] == label).sum()) for label in config.PERFORMANCE_CLASS_LABELS
    }
    suitability_report = {
        "weighting_method": "random_forest_classifier",
        "predicted_class": CLASS_COLUMN,
        "probabilities": list(PROBABILITY_COLUMNS),
        "neighborhood_radius_km": float(config.NEIGHBORHOOD_RADIUS_KM),
        **label_report,
        "predicted_class_counts": predicted_counts,
        "macro_f1_above_majority_baseline": fitted["macro_f1_above_majority_baseline"],
        "test_metrics": fitted["test_metrics"],
        "baseline_metrics": fitted["baseline_metrics"],
        "stratified_baseline_metrics": fitted["stratified_baseline_metrics"],
        "cells_outside_training_feature_range": fitted["cells_outside_training_feature_range"],
    }
    output_report = {
        "cells": int(len(output)),
        "class_counts": predicted_counts,
        **support_report,
    }
    model_report = {
        key: value for key, value in fitted.items() if key not in {"model", "evaluation_model"}
    }
    model_report["experiments"] = experiments

    _write_outputs(
        output_dir,
        cell_classification,
        merged,
        labeled,
        output,
        {
            "study_area": config.PROVINCE_NAME,
            "grid_cells": cell_report["cells"],
            "barangay_classification": barangay_report,
            "listing_classification": listing_report,
            "grid_classification": cell_report,
            "rural_population_levels": levels,
            "rural_training": training_report,
            "prediction_domain": prediction_report,
            "models": {"location_success": model_report},
            "suitability": suitability_report,
            "output": output_report,
        },
        suitability_report | output_report,
    )
    return _summary(cell_report, training_report, prediction_report, suitability_report, output_report)


def _assemble_scored_grid(
    grid: gpd.GeoDataFrame,
    cell_classification: pd.DataFrame,
    scored: pd.DataFrame,
) -> gpd.GeoDataFrame:
    overlap = [column for column in scored.columns if column != CELL_ID_COLUMN and column in grid.columns]
    output = grid.drop(columns=overlap).merge(
        cell_classification[[CELL_ID_COLUMN, CELL_CLASS_COLUMN]],
        on=CELL_ID_COLUMN,
        how="left",
        validate="one_to_one",
    ).merge(scored, on=CELL_ID_COLUMN, how="left", validate="one_to_one")
    output[IN_RURAL_ANALYSIS_COLUMN] = output[CELL_CLASS_COLUMN] == RURAL_CELL
    assert output[CLASS_COLUMN].isin(config.PERFORMANCE_CLASS_LABELS).all(), (
        "a grid cell is outside Low, Moderate, and High"
    )
    probability_total = output[list(PROBABILITY_COLUMNS)].sum(axis=1)
    assert np.allclose(probability_total, 1.0), "a cell's class probabilities do not sum to 1"
    return gpd.GeoDataFrame(output, geometry="geometry", crs=grid.crs)


def _write_outputs(
    output_dir,
    cell_classification: pd.DataFrame,
    merged: pd.DataFrame,
    labeled: pd.DataFrame,
    output: gpd.GeoDataFrame,
    model_summary: dict[str, object],
    suitability_summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_classification.to_parquet(output_dir / CELL_CLASSIFICATION_FILENAME, index=False)
    merged.to_parquet(output_dir / OBSERVATIONS_FILENAME, index=False)
    labeled.to_parquet(output_dir / LOCATION_TRAINING_FILENAME, index=False)
    output.to_parquet(output_dir / SUITABILITY_FILENAME, index=False)
    (output_dir / MODEL_SUMMARY_FILENAME).write_text(
        json.dumps(model_summary, indent=2, default=str)
    )
    (output_dir / SUITABILITY_SUMMARY_FILENAME).write_text(
        json.dumps(suitability_summary, indent=2, default=str)
    )


def _summary(
    cell_report: dict[str, object],
    training_report: dict[str, object],
    prediction_report: dict[str, object],
    suitability_report: dict[str, object],
    output_report: dict[str, object],
) -> dict[str, object]:
    return {
        "grid": cell_report,
        "rural_training": training_report,
        "prediction_domain": prediction_report,
        "class_counts": output_report["class_counts"],
        "macro_f1_above_majority_baseline": suitability_report["macro_f1_above_majority_baseline"],
        "test_metrics": suitability_report["test_metrics"],
        "baseline_metrics": suitability_report["baseline_metrics"],
    }
