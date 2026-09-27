import json

import geopandas as gpd
import pandas as pd

from str_suitability import config
from str_suitability.pipeline import FEATURE_COLUMNS, run_model
from str_suitability.rural.diagrams import (
    DIAGRAM_PATH_NAME,
    example_cells,
    format_example,
    tree_path,
    write_cell_diagrams,
)
from str_suitability.rural.classify import (
    CELL_CLASS_COLUMN,
    CELL_ID_COLUMN,
    RURAL_CELL,
    attach_listing_classification,
    classify_grid_cells,
    load_listing_classification,
    rural_population_levels,
    select_cells_with_class,
)
from str_suitability.rural.prediction import predict_rural_cells
from str_suitability.rural.psa import (
    classify_barangay_polygons,
    load_barangay_classification,
    load_barangay_polygons,
    report_barangay_classification,
)
from str_suitability.rural.site_score import compute_site_score, load_listed_tourist_places
from str_suitability.rural.suitability import (
    RURAL_CLASS_COLUMN,
    assemble_rural_output,
    attach_training_support,
)
from str_suitability.rural.training import build_rural_training_frame, rural_training_cell_classes

GRID_FILENAME = "grid_features.parquet"
TARGETS_FILENAME = "airroi_targets.parquet"
OBSERVATIONS_FILENAME = "training_observations.parquet"
CELL_CLASSIFICATION_FILENAME = "grid_rural_classification.parquet"
SUITABILITY_FILENAME = "grid_suitability.parquet"
MODEL_SUMMARY_FILENAME = "model_summary.json"
SUITABILITY_SUMMARY_FILENAME = "suitability_summary.json"

MODEL_TARGETS = {
    "revenue": config.REVENUE_TARGET,
    "occupancy": config.OCCUPANCY_TARGET,
}


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
    results = {name: run_model(merged, target) for name, target in MODEL_TARGETS.items()}

    rural_cells = select_cells_with_class(grid, cell_classification, RURAL_CELL)
    predictions = predict_rural_cells(
        rural_cells, {name: result["model"] for name, result in results.items()}, FEATURE_COLUMNS
    )
    active_listings = targets.loc[targets["in_training_population"]].copy()
    scored, suitability_report = compute_site_score(
        rural_cells,
        active_listings,
        load_listed_tourist_places(config.LISTED_TOURIST_PLACES_PATH),
    )
    for column, frame in predictions.items():
        scored = scored.merge(
            frame.rename(columns={column: f"rural_{column}"}),
            on=CELL_ID_COLUMN,
            how="left",
            validate="one_to_one",
        )
    already_on_grid = [
        column for column in scored.columns if column != CELL_ID_COLUMN and column in grid.columns
    ]
    scored = scored.drop(columns=already_on_grid)
    output, output_report = assemble_rural_output(grid, cell_classification, scored)
    output, support_report = attach_training_support(output, merged)
    output_report.update(support_report)
    diagram_path = output_dir / DIAGRAM_PATH_NAME
    write_cell_diagrams(output, results["revenue"]["model"], diagram_path)
    print(
        "rural training: "
        f"{training_report['fitted_rural_listings']} fitted listings in "
        f"{training_report['rural_cells_with_fitted_rural_listings']} cells"
    )
    print("example cells:")
    chosen = example_cells(output)
    for row in chosen:
        print(format_example(row, results["revenue"]["model"]))
        print()
    high_rows = [row for row in chosen if row[RURAL_CLASS_COLUMN] == "High"]
    if high_rows:
        high_values = [float(high_rows[0][column]) for column in FEATURE_COLUMNS]
        print("occupancy tree for the high cell:")
        print(
            "\n".join(
                f"  {step}"
                for step in tree_path(
                    results["occupancy"]["model"],
                    high_values,
                    "trailing-twelve-month occupancy",
                )
            )
        )
        print()

    prediction_report = {
        "rural_cells_predicted": int(len(rural_cells)),
        "urban_cells_excluded": cell_report["urban_cells"],
        "unclassified_cells_excluded": cell_report["unclassified_cells"],
    }
    classified_rural = set(
        cell_classification.loc[cell_classification[CELL_CLASS_COLUMN] == RURAL_CELL, CELL_ID_COLUMN]
    )
    assert set(rural_cells[CELL_ID_COLUMN]) == classified_rural, (
        "the rural prediction domain must be exactly the cells classified rural"
    )
    assert (
        prediction_report["rural_cells_predicted"]
        + prediction_report["urban_cells_excluded"]
        + prediction_report["unclassified_cells_excluded"]
        == cell_report["cells"]
    ), "the rural domain and the excluded cells must partition the grid"

    _write_outputs(
        output_dir,
        cell_classification,
        merged,
        predictions,
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
            "models": {
                name: {key: value for key, value in result.items() if key != "model"}
                for name, result in results.items()
            },
            "suitability": suitability_report,
            "rural_output": output_report,
        },
        suitability_report | output_report,
    )
    return _summary(cell_report, training_report, prediction_report, suitability_report, output_report)


def _write_outputs(
    output_dir,
    cell_classification: pd.DataFrame,
    merged: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    output: gpd.GeoDataFrame,
    model_summary: dict[str, object],
    suitability_summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_classification.to_parquet(output_dir / CELL_CLASSIFICATION_FILENAME, index=False)
    merged.to_parquet(output_dir / OBSERVATIONS_FILENAME, index=False)
    for name, frame in predictions.items():
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
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
        "weights": suitability_report["weights"],
        "class_counts": output_report["class_counts"],
    }
