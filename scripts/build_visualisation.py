import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from str_suitability import config
from str_suitability.modeling.site_candidate import (
    FEATURES,
    REPORT_FILENAME,
    SCORES_FILENAME,
)
from str_suitability.rural.classify import RURAL_CELL, UNCLASSIFIED_CELL, URBAN_CELL

OUTPUT_DIR = config.PROCESSED_DIR / "frontend"
GRID_PATH = config.RURAL_PROCESSED_DIR / "grid_suitability.parquet"
SCORES_PATH = config.RURAL_PROCESSED_DIR / SCORES_FILENAME
REPORT_PATH = config.RURAL_PROCESSED_DIR / REPORT_FILENAME
TEMPLATE_PATH = Path(__file__).with_name("site_candidate_map_template.html")

FEATURE_LABELS = {
    "listed_places_within_radius": "Landscape places within 5 km",
    "distance_to_listed_tourist_place": "Distance to nearest landscape place (km)",
    "road_distance_km": "Distance to nearest mapped road (km)",
    "distance_to_nearest_town_center_km": "Distance to Poblacion (km)",
    "distance_to_major_road_km": "Distance to major road (km)",
    "attraction_count_within_5km": "Falls and mountains within 5 km",
}
DISPLAY_BANDS = ("<0.40", "0.40 to <0.60", ">=0.60")
DISPLAY_COLORS = ("#c7523c", "#e2b44a", "#39836f")
CELL_CLASSES = {RURAL_CELL: 0, URBAN_CELL: 1, UNCLASSIFIED_CELL: 2}


def geometry_parts(geometry) -> list[list[list[list[float]]]]:
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    result = []
    for polygon in polygons:
        rings = [[[round(lat, 6), round(lon, 6)] for lon, lat in polygon.exterior.coords]]
        for interior in polygon.interiors:
            rings.append([[round(lat, 6), round(lon, 6)] for lon, lat in interior.coords])
        result.append(rings)
    return result


def _band_index(label: str) -> int:
    return DISPLAY_BANDS.index(label)


def build_payload() -> dict[str, object]:
    grid = gpd.read_parquet(GRID_PATH).to_crs(config.GEOGRAPHIC_CRS)
    scores = pd.read_parquet(SCORES_PATH)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert set(scores["model_name"]) == {"random_forest"}
    scores["cell_id"] = scores["cell_id"].astype(str)
    grid["cell_id"] = grid["cell_id"].astype(str)
    score_columns = [
        "cell_id",
        "candidate_pattern_score",
        "is_empty",
        "in_training_sample",
        "display_band",
        *FEATURES,
    ]
    cells = grid.merge(
        scores[score_columns],
        on="cell_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_candidate"),
    )
    rural = cells["cell_class"] == RURAL_CELL
    assert cells.loc[rural, "candidate_pattern_score"].notna().sum() == 1041

    records = []
    for row in cells.itertuples(index=False):
        cell_class = row.cell_class
        score = getattr(row, "candidate_pattern_score")
        is_rural = cell_class == RURAL_CELL
        is_empty = bool(getattr(row, "listings_in_cell", 0) == 0) if is_rural else False
        band = _band_index(row.display_band) if is_rural and pd.notna(score) else None
        records.append(
            {
                "geometry": geometry_parts(row.geometry),
                "id": str(row.cell_id),
                "municipality": None if pd.isna(row.municipality) else str(row.municipality),
                "class": CELL_CLASSES[cell_class],
                "empty": is_empty,
                "score": None if pd.isna(score) else round(float(score), 5),
                "band": band,
                "listings": int(getattr(row, "listings_in_cell", 0)) if is_rural else None,
                "features": {
                    feature: None
                    if pd.isna(getattr(row, f"{feature}_candidate", getattr(row, feature, None)))
                    else round(float(getattr(row, f"{feature}_candidate", getattr(row, feature, None))), 4)
                    for feature in FEATURES
                } if is_rural else None,
            }
        )

    empty_scores = scores.loc[scores["is_empty"], "candidate_pattern_score"]
    band_counts = [int((scores.loc[scores["is_empty"], "display_band"] == band).sum()) for band in DISPLAY_BANDS]
    metrics = report["metrics"]
    return {
        "cells": records,
        "bounds": [
            [float(grid.total_bounds[1]), float(grid.total_bounds[0])],
            [float(grid.total_bounds[3]), float(grid.total_bounds[2])],
        ],
        "bands": list(DISPLAY_BANDS),
        "colors": list(DISPLAY_COLORS),
        "featureLabels": [FEATURE_LABELS[feature] for feature in FEATURES],
        "scoreMean": float(empty_scores.mean()),
        "scoreCount": int(len(empty_scores)),
        "bandCounts": band_counts,
        "auc": float(metrics["roc_auc"]["mean"]),
        "folds": len(report["folds"]),
        "trainingRows": report["training_rows"],
        "trainingPositives": report["training_positives"],
        "trainingNegatives": report["training_negatives"],
        "ruralCells": int(rural.sum()),
        "urbanCells": int((cells["cell_class"] == URBAN_CELL).sum()),
        "unclassifiedCells": int((cells["cell_class"] == UNCLASSIFIED_CELL).sum()),
    }


def render_html(payload: dict[str, object]) -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    (OUTPUT_DIR / "index.html").write_text(render_html(payload), encoding="utf-8")
    print(
        f"Rendered RF site-candidate map: {payload['scoreCount']} empty rural cells; "
        f"mean held-out ROC-AUC {payload['auc']:.3f}"
    )


if __name__ == "__main__":
    main()