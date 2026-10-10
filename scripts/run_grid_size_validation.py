"""Held-out scores for the same forest on 500 m, 1 km, 1.5 km, and 2 km grids.

Writes only under results/grid_size_validation/. Does not replace the 1 km
grid or the existing site-candidate files.
"""

from __future__ import annotations

import traceback

import geopandas as gpd
import pandas as pd

from str_suitability import config
from str_suitability.config import GEOGRAPHIC_CRS
from str_suitability.features.context import add_poblacion_distance
from str_suitability.features.road_distance import nearest_mapped_road_distance
from str_suitability.ingest.load_osm import load_boundaries
from str_suitability.modeling.site_candidate import (
    CELL_ID_COLUMN,
    LAGUNA_ROADS_PATH,
    build_feature_frame,
    build_training_sample,
    evaluate_forest,
    load_places,
)
from str_suitability.pipeline import load_water
from str_suitability.rural.classify import CELL_CLASS_COLUMN, RURAL_CELL, classify_grid_cells
from str_suitability.rural.psa import (
    classify_barangay_polygons,
    load_barangay_classification,
    load_barangay_polygons,
)
from str_suitability.spatial.grid import build_grid, derive_land_boundary
from str_suitability.spatial.join import assign_listings_to_cells

SIZES_M = (500, 1000, 1500, 2000)
LOCKED_1KM_ROC = 0.7274258287458627
OUT = config.PROJECT_ROOT / "results" / "grid_size_validation"


def _with_barangay_centers(barangays: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    points = barangays.geometry.representative_point().to_crs(GEOGRAPHIC_CRS)
    enriched = barangays.copy()
    enriched["center_lon"] = points.x.to_numpy()
    enriched["center_lat"] = points.y.to_numpy()
    return enriched


def _municipality(grid: gpd.GeoDataFrame, municipalities: gpd.GeoDataFrame) -> pd.Series:
    munis = municipalities.to_crs(grid.crs)
    name_column = "name" if "name" in munis.columns else munis.columns[0]
    overlay = gpd.overlay(
        grid[[CELL_ID_COLUMN, "geometry"]],
        munis[[name_column, "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    overlay["overlap_area_m2"] = overlay.geometry.area
    dominant = (
        overlay.sort_values([CELL_ID_COLUMN, "overlap_area_m2"], ascending=[True, False])
        .drop_duplicates(CELL_ID_COLUMN)
        .set_index(CELL_ID_COLUMN)[name_column]
    )
    return grid[CELL_ID_COLUMN].astype(str).map(dominant)


def _listing_counts(grid: gpd.GeoDataFrame, listings: pd.DataFrame) -> pd.Series:
    joined, _report = assign_listings_to_cells(listings, grid)
    active = joined.loc[joined["in_training_population"].astype(bool)].dropna(subset=[CELL_ID_COLUMN])
    active[CELL_ID_COLUMN] = active[CELL_ID_COLUMN].astype(str)
    counts = active.groupby(CELL_ID_COLUMN).size()
    return grid[CELL_ID_COLUMN].astype(str).map(counts).fillna(0).astype(int)


def _evaluate_size(
    cell_size_m: int,
    land,
    barangays: gpd.GeoDataFrame,
    municipalities: gpd.GeoDataFrame,
    listings: pd.DataFrame,
    places: pd.DataFrame,
    roads: gpd.GeoDataFrame,
) -> dict[str, object]:
    print(f"building {cell_size_m} m grid", flush=True)
    grid = build_grid(land, cell_size_m)
    grid[CELL_ID_COLUMN] = grid[CELL_ID_COLUMN].astype(str)
    classification, _report = classify_grid_cells(grid, barangays)
    grid = grid.merge(
        classification[[CELL_ID_COLUMN, CELL_CLASS_COLUMN]],
        on=CELL_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    grid["municipality"] = _municipality(grid, municipalities).to_numpy()
    grid["listings_in_cell"] = _listing_counts(grid, listings).to_numpy()
    grid, _poblacion = add_poblacion_distance(grid, barangays)
    distances = nearest_mapped_road_distance(grid, roads)
    distances[CELL_ID_COLUMN] = distances[CELL_ID_COLUMN].astype(str)
    grid = grid.merge(distances[["cell_id", "road_distance_km"]], on=CELL_ID_COLUMN, how="left", validate="one_to_one")
    featured = build_feature_frame(grid, places, roads)
    rural = featured["cell_class"].eq(RURAL_CELL)
    listed = int((rural & featured["listings_in_cell"].ge(1) & featured["municipality"].notna()).sum())
    empty = int((rural & featured["listings_in_cell"].eq(0) & featured["municipality"].notna()).sum())
    print(f"  cells {len(grid)} rural {int(rural.sum())} listed {listed} empty {empty}", flush=True)
    sample = build_training_sample(featured, negative_seed=42)
    _oof, report = evaluate_forest(sample)
    folds = [float(row["roc_auc"]) for row in report["folds"]]
    return {
        "cell_size_m": cell_size_m,
        "grid_cells": int(len(grid)),
        "rural_cells": int(rural.sum()),
        "listed_cells": listed,
        "empty_cells": empty,
        "training_rows": int(report["training_rows"]),
        "roc_auc_mean": float(report["metrics"]["roc_auc"]["mean"]),
        "roc_auc_min": float(min(folds)),
        "roc_auc_max": float(max(folds)),
        "pr_auc_mean": float(report["metrics"]["pr_auc"]["mean"]),
        "macro_f1_mean": float(report["metrics"]["macro_f1"]["mean"]),
        "roc_auc_folds": folds,
        "error": "",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    province, municipalities = load_boundaries(config.BOUNDARY_DIR)
    land = derive_land_boundary(province, load_water(config.PROJECT_ROOT / "assets" / "osm"))
    barangays = _with_barangay_centers(
        classify_barangay_polygons(
            load_barangay_polygons(config.BARANGAY_POLYGONS_PATH),
            load_barangay_classification(config.BARANGAY_CLASSIFICATION_PATH),
        )
    )
    listings = pd.read_parquet(config.INTERIM_DIR / "airroi_targets.parquet")
    places, _place_report = load_places(config.LISTED_TOURIST_PLACES_PATH)
    roads = gpd.read_file(LAGUNA_ROADS_PATH)
    rows = []
    for cell_size_m in SIZES_M:
        try:
            rows.append(
                _evaluate_size(cell_size_m, land, barangays, municipalities, listings, places, roads)
            )
        except Exception as error:
            traceback.print_exc()
            rows.append(
                {
                    "cell_size_m": cell_size_m,
                    "grid_cells": None,
                    "rural_cells": None,
                    "listed_cells": None,
                    "empty_cells": None,
                    "training_rows": None,
                    "roc_auc_mean": None,
                    "roc_auc_min": None,
                    "roc_auc_max": None,
                    "pr_auc_mean": None,
                    "macro_f1_mean": None,
                    "roc_auc_folds": [],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"  failed {cell_size_m}: {error}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "summary.csv", index=False)
    (OUT / "summary.json").write_text(table.to_json(orient="records", indent=2), encoding="utf-8")
    print(table.to_string(index=False), flush=True)
    print(f"locked 1 km ROC-AUC {LOCKED_1KM_ROC}", flush=True)


if __name__ == "__main__":
    main()
