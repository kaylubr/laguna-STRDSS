"""Interactive 500 m map in the same style as the 1 km site-candidate map.

Writes results/grid_size_validation/map_500m.html. Does not replace the 1 km map.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_visualisation as viz
from run_grid_size_validation import (
    _listing_counts,
    _municipality,
    _with_barangay_centers,
)
from str_suitability import config
from str_suitability.features.context import add_poblacion_distance
from str_suitability.features.road_distance import nearest_mapped_road_distance
from str_suitability.ingest.load_osm import load_boundaries
from str_suitability.modeling.site_candidate import (
    CELL_ID_COLUMN,
    FEATURES,
    LAGUNA_ROADS_PATH,
    build_feature_frame,
    build_training_sample,
    load_places,
    score_rural_cells,
)
from str_suitability.pipeline import load_water
from str_suitability.rural.classify import (
    CELL_CLASS_COLUMN,
    RURAL_CELL,
    UNCLASSIFIED_CELL,
    URBAN_CELL,
    classify_grid_cells,
)
from str_suitability.rural.psa import (
    classify_barangay_polygons,
    load_barangay_classification,
    load_barangay_polygons,
)
from str_suitability.spatial.grid import build_grid, derive_land_boundary

OUT = config.PROJECT_ROOT / "results" / "grid_size_validation" / "map_500m.html"
MODEL_DIR = config.PROJECT_ROOT / "results" / "grid_500m"
CELL_SIZE_M = 500
HELD_OUT_ROC_AUC = 0.7756046551039055


def _featured(land, barangays, municipalities, listings, places, roads) -> gpd.GeoDataFrame:
    print("building 500 m grid", flush=True)
    grid = build_grid(land, CELL_SIZE_M)
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
    grid = grid.merge(
        distances[[CELL_ID_COLUMN, "road_distance_km"]],
        on=CELL_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    return build_feature_frame(grid, places, roads)


def _payload(grid: gpd.GeoDataFrame, scores: pd.DataFrame) -> dict[str, object]:
    geographic = grid.to_crs(config.GEOGRAPHIC_CRS)
    scores = scores.copy()
    scores[CELL_ID_COLUMN] = scores[CELL_ID_COLUMN].astype(str)
    geographic[CELL_ID_COLUMN] = geographic[CELL_ID_COLUMN].astype(str)
    cells = geographic.merge(
        scores[[CELL_ID_COLUMN, "candidate_pattern_score", "is_empty", "display_band"]],
        on=CELL_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    records = []
    for row in cells.itertuples(index=False):
        is_rural = row.cell_class == RURAL_CELL
        score = row.candidate_pattern_score
        is_empty = bool(row.listings_in_cell == 0) if is_rural else False
        band = viz._band_index(row.display_band) if is_rural and pd.notna(score) and is_empty else None
        records.append(
            {
                "geometry": viz.geometry_parts(row.geometry),
                "id": str(row.cell_id),
                "municipality": None if pd.isna(row.municipality) else str(row.municipality),
                "class": viz.CELL_CLASSES[row.cell_class],
                "empty": is_empty,
                "score": None if pd.isna(score) else round(float(score), 5),
                "band": band,
                "listings": int(row.listings_in_cell) if is_rural else None,
                "features": {
                    feature: None if pd.isna(getattr(row, feature)) else round(float(getattr(row, feature)), 4)
                    for feature in FEATURES
                }
                if is_rural and pd.notna(score)
                else None,
            }
        )
    empty_scores = scores.loc[scores["is_empty"]]
    band_counts = [int((empty_scores["display_band"] == band).sum()) for band in viz.DISPLAY_BANDS]
    return {
        "cells": records,
        "bounds": [
            [float(geographic.total_bounds[1]), float(geographic.total_bounds[0])],
            [float(geographic.total_bounds[3]), float(geographic.total_bounds[2])],
        ],
        "bands": list(viz.DISPLAY_BANDS),
        "colors": list(viz.DISPLAY_COLORS),
        "featureLabels": [viz.FEATURE_LABELS[feature] for feature in FEATURES],
        "scoreMean": float(empty_scores["candidate_pattern_score"].mean()),
        "scoreCount": int(len(empty_scores)),
        "bandCounts": band_counts,
        "auc": HELD_OUT_ROC_AUC,
        "folds": 5,
        "trainingRows": 244,
        "trainingPositives": 122,
        "trainingNegatives": 122,
        "ruralCells": int(cells["cell_class"].eq(RURAL_CELL).sum()),
        "urbanCells": int(cells["cell_class"].eq(URBAN_CELL).sum()),
        "unclassifiedCells": int(cells["cell_class"].eq(UNCLASSIFIED_CELL).sum()),
    }


def main() -> None:
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
    featured = _featured(land, barangays, municipalities, listings, places, roads)
    rural = featured["cell_class"].eq(RURAL_CELL) & featured["municipality"].notna()
    listed = int((rural & featured["listings_in_cell"].ge(1)).sum())
    empty = int((rural & featured["listings_in_cell"].eq(0)).sum())
    if listed != 122 or empty != 4000:
        raise RuntimeError(f"500 m grid has {listed} listing cells and {empty} empty cells")
    sample = build_training_sample(featured, negative_seed=42)
    scores = score_rural_cells(sample, featured)
    payload = _payload(featured, scores)
    html = viz.render_html(payload)
    html = html.replace(
        "<title>Laguna rural site-candidate scores</title>",
        "<title>Laguna 500 m rural site-candidate scores</title>",
    )
    html = html.replace(
        "<h1>Rural site-candidate scores</h1>",
        "<h1>500 m rural site-candidate scores</h1>",
    )
    html = html.replace(
        "Color shows RF scores for empty rural cells only.",
        "500 m grid. Color shows RF scores for empty rural cells only.",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    scored = featured.merge(
        scores[[CELL_ID_COLUMN, "candidate_pattern_score", "is_empty", "display_band", "in_training_sample"]],
        on=CELL_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    scored.to_parquet(MODEL_DIR / "cells.parquet")
    sample.drop(columns=["geometry"], errors="ignore").to_parquet(MODEL_DIR / "training_sample.parquet", index=False)
    print(
        f"wrote {OUT} empty {payload['scoreCount']} bands {payload['bandCounts']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
