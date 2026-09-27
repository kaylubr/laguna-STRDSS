import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from str_suitability import config
from str_suitability.features.accessibility import (
    add_accessibility_distances,
    add_tourism_features,
)
from str_suitability.features.context import (
    LAKE_DISTANCE_COLUMN,
    OTHER_WATER_DISTANCE_COLUMN,
    POBLACION_DISTANCE_COLUMN,
    add_poblacion_distance,
    add_water_features,
)
from str_suitability.features.poi import add_poi_density
from str_suitability.ingest.load_osm import load_boundaries
from str_suitability.modeling.evaluate import describe_target, evaluate_on_test, evaluate_on_train
from str_suitability.modeling.importance import permutation_importance_table
from str_suitability.modeling.train import train_model
from str_suitability.preprocess.clean_psa import attach_psgc_codes, load_psa_population
from str_suitability.spatial.grid import assert_one_row_per_cell, build_grid, derive_land_boundary
from str_suitability.spatial.join import (
    assign_listings_to_cells,
    build_municipality_table,
    join_demographics,
)

FEATURE_DIR = config.PROJECT_ROOT / "assets" / "osm"

POI_DENSITY_COLUMNS = [
    "poi_density_restaurants",
    "poi_density_commercial",
    "poi_density_transportation",
    "poi_density_recreation",
    "poi_density_tourist_attraction",
    "poi_density_other_facilities",
    "poi_density_total",
]

FEATURE_COLUMNS = [
    *POI_DENSITY_COLUMNS,
    "tourist_attraction_count",
    "tourist_attraction_density",
    "distance_to_nearest_transportation_facility",
    "distance_to_nearest_tourist_attraction",
    LAKE_DISTANCE_COLUMN,
    OTHER_WATER_DISTANCE_COLUMN,
    POBLACION_DISTANCE_COLUMN,
    "population_density_per_km2",
]


def load_water(feature_dir: Path) -> gpd.GeoDataFrame:
    water_path = feature_dir / "laguna_water.geojson"
    if not water_path.exists():
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=config.GEOGRAPHIC_CRS)
    return gpd.read_file(water_path)


def load_pois(feature_dir: Path) -> gpd.GeoDataFrame:
    return gpd.read_parquet(feature_dir / "laguna_pois.parquet")


def build_grid_frame() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    province, municipalities = load_boundaries(config.BOUNDARY_DIR)
    water = load_water(FEATURE_DIR)
    land_polygon = derive_land_boundary(province, water)
    grid = build_grid(land_polygon, config.PRIMARY_CELL_SIZE_M)
    return grid, municipalities


def build_feature_frame(
    grid: gpd.GeoDataFrame,
    municipalities: gpd.GeoDataFrame,
    population: pd.DataFrame,
    pois: gpd.GeoDataFrame,
    water: gpd.GeoDataFrame,
    barangays: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, dict]:
    assert_one_row_per_cell(grid)

    municipality_table = build_municipality_table(municipalities, population)
    assert municipality_table["psgc_ref"].notna().all(), "every municipality must have a PSGC code"
    assert municipality_table["psgc_ref"].is_unique, "PSGC codes must be unique"
    assert municipality_table["population"].notna().all(), "every municipality must have a population"

    grid, demographic_report = join_demographics(grid, municipality_table)
    assert_one_row_per_cell(grid)

    transport_facilities = pois[pois["is_transport_facility"]].reset_index(drop=True)
    attractions = pois[pois["is_tourist_attraction"]].reset_index(drop=True)

    grid = add_poi_density(grid, pois)
    assert_one_row_per_cell(grid)
    grid = add_accessibility_distances(grid, transport_facilities, attractions)
    assert_one_row_per_cell(grid)
    grid = add_tourism_features(grid, attractions)
    assert_one_row_per_cell(grid)
    grid = add_water_features(grid, water)
    assert_one_row_per_cell(grid)
    grid, poblacion_report = add_poblacion_distance(grid, barangays)
    assert_one_row_per_cell(grid)

    report = {
        **demographic_report,
        "municipalities_with_demographics": int(municipality_table["population"].notna().sum()),
        "transport_facilities": int(len(transport_facilities)),
        "tourist_attractions": int(len(attractions)),
        "poblacion": poblacion_report,
        "cells_missing_population": int(grid["population_density_per_km2"].isna().sum()),
        "cells_missing_attraction_distance": int(
            grid["distance_to_nearest_tourist_attraction"].isna().sum()
        ),
        "cells_missing_transport_distance": int(
            grid["distance_to_nearest_transportation_facility"].isna().sum()
        ),
        "cells_missing_water_distance": int(
            grid[[LAKE_DISTANCE_COLUMN, OTHER_WATER_DISTANCE_COLUMN]].isna().any(axis=1).sum()
        ),
        "cells_missing_poblacion_distance": int(grid[POBLACION_DISTANCE_COLUMN].isna().sum()),
    }
    return grid, report


def build_training_frame(
    targets: pd.DataFrame, grid: gpd.GeoDataFrame
) -> tuple[pd.DataFrame, dict]:
    training = targets[targets["in_training_population"]].copy()
    assigned, join_report = assign_listings_to_cells(training, grid)

    feature_columns = ["cell_id", *FEATURE_COLUMNS]
    merged = assigned.merge(
        pd.DataFrame(grid[feature_columns]), on="cell_id", how="left"
    )
    merged = merged.dropna(subset=["cell_id", *FEATURE_COLUMNS]).reset_index(drop=True)

    report = {
        **join_report,
        "training_listings_with_features": int(len(merged)),
        "dropped_missing_features": int(len(assigned) - len(merged)),
        "distinct_cells_with_training_listings": int(merged["cell_id"].nunique()),
        "feature_nulls": int(merged[FEATURE_COLUMNS].isna().sum().sum()),
    }
    return merged, report


def run_model(merged: pd.DataFrame, target_column: str) -> dict:
    features = merged[FEATURE_COLUMNS]
    target = merged[target_column]
    groups = merged["cell_id"]

    primary = train_model(features, target, groups=groups)
    comparison = train_model(features, target)
    model = primary["model"]
    test_metrics = evaluate_on_test(model, primary["features_test"], primary["target_test"])
    random_metrics = evaluate_on_test(
        comparison["model"], comparison["features_test"], comparison["target_test"]
    )

    return {
        "target": target_column,
        "cv_scheme": primary["scheme"],
        "best_params": primary["best_params"],
        "best_cv_rmse": primary["best_cv_rmse"],
        "test_metrics": test_metrics,
        "test_metrics_random_split": random_metrics,
        "test_r2_change_grouped_minus_random": float(test_metrics["r2"] - random_metrics["r2"]),
        "train_metrics": evaluate_on_train(model, primary["features_train"], primary["target_train"]),
        "target_distribution": describe_target(target),
        "importance": permutation_importance_table(
            model, primary["features_test"], primary["target_test"]
        ).to_dict(orient="records"),
        "model": model,
    }


def run_pipeline() -> dict:
    targets = pd.read_parquet(config.INTERIM_DIR / "airroi_targets.parquet")
    population = load_psa_population(config.PSA_PATH)
    _, municipalities = load_boundaries(config.BOUNDARY_DIR)
    population, psgc_report = attach_psgc_codes(population, municipalities)

    grid, _ = build_grid_frame()
    pois = load_pois(FEATURE_DIR)
    water = load_water(FEATURE_DIR)
    barangays = gpd.read_file(config.BARANGAY_POLYGONS_PATH)
    features, feature_report = build_feature_frame(
        grid, municipalities, population, pois, water, barangays
    )
    merged, join_report = build_training_frame(targets, features)

    revenue = run_model(merged, config.REVENUE_TARGET)
    occupancy = run_model(merged, config.OCCUPANCY_TARGET)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(config.PROCESSED_DIR / "grid_features.parquet", index=False)
    merged.to_parquet(config.PROCESSED_DIR / "training_observations.parquet", index=False)

    for name, result in (("revenue", revenue), ("occupancy", occupancy)):
        features[f"predicted_{name}"] = result["model"].predict(features[FEATURE_COLUMNS])
        features[["cell_id", f"predicted_{name}"]].to_parquet(
            config.PROCESSED_DIR / f"predicted_{name}.parquet", index=False
        )

    summary = {
        "psgc_codes_attached": psgc_report,
        "grid_cells": int(len(features)),
        "unique_cell_ids": int(features["cell_id"].nunique()),
        "active_listings_retained": int(len(targets[targets["in_training_population"]])),
        "listings_assigned_to_cell": int(join_report["assigned_to_cell"]),
        "listings_used_for_training": int(len(merged)),
        "cells_with_demographics": int(features["population_density_per_km2"].notna().sum()),
        "cells_with_training_listings": int(join_report["distinct_cells_with_training_listings"]),
        "feature_report": feature_report,
        "join_report": join_report,
        "revenue": {key: value for key, value in revenue.items() if key != "model"},
        "occupancy": {key: value for key, value in occupancy.items() if key != "model"},
    }
    (config.PROCESSED_DIR / "model_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return summary
