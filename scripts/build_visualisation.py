import hashlib
import json
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd

from str_suitability import config
from str_suitability.pipeline import FEATURE_COLUMNS
from str_suitability.rural.classify import RURAL_CELL, UNCLASSIFIED_CELL, URBAN_CELL
from str_suitability.rural.site_score import (
    COMPETITION_COLUMN,
    COMPETITION_CONTRIBUTION,
    DISTANCE_COLUMN,
    LISTINGS_IN_CELL_COLUMN,
    LISTINGS_NEARBY_COLUMN,
    MARKET_CONTRIBUTION,
    NEAREST_PLACE_COLUMN,
    NEARBY_OCCUPANCY_COLUMN,
    NEARBY_REVENUE_COLUMN,
    TOURIST_ACCESS_CONTRIBUTION,
    load_listed_tourist_places,
)
from str_suitability.rural.suitability import OUTSIDE_RURAL_ANALYSIS_LABELS
from str_suitability.suitability.classify import TRAFFIC_CLASS_LABELS

OUTPUT_DIR = config.PROCESSED_DIR / "frontend"
GRID_PATH = config.PROCESSED_DIR / "grid_features.parquet"
RURAL_GRID_PATH = config.RURAL_PROCESSED_DIR / "grid_suitability.parquet"
TARGETS_PATH = config.INTERIM_DIR / "airroi_targets.parquet"

CELL_CLASS_KEYS = {RURAL_CELL: 0, URBAN_CELL: 1, UNCLASSIFIED_CELL: 2}
RURAL_OUTSIDE_LABELS = (
    OUTSIDE_RURAL_ANALYSIS_LABELS[URBAN_CELL],
    OUTSIDE_RURAL_ANALYSIS_LABELS[UNCLASSIFIED_CELL],
)

RURAL_COLUMNS = (
    "cell_id",
    "cell_class",
    "in_rural_analysis",
    "rural_composite_suitability_score",
    "rural_suitability_class",
    DISTANCE_COLUMN,
    NEAREST_PLACE_COLUMN,
    NEARBY_REVENUE_COLUMN,
    NEARBY_OCCUPANCY_COLUMN,
    LISTINGS_IN_CELL_COLUMN,
    LISTINGS_NEARBY_COLUMN,
    COMPETITION_COLUMN,
    TOURIST_ACCESS_CONTRIBUTION,
    MARKET_CONTRIBUTION,
    COMPETITION_CONTRIBUTION,
    "rural_predicted_revenue",
    "rural_predicted_occupancy",
)

FEATURE_LABELS = {
    "distance_to_listed_tourist_place": "Distance to nearest place in poi_laguna.json",
    "listed_places_within_radius": "Listed tourist places within 5 km",
    "active_listings_within_radius": "Active Airbnb listings within 5 km",
}

RURAL_CLASSES = (*TRAFFIC_CLASS_LABELS, *RURAL_OUTSIDE_LABELS)
RURAL_CLASS_COLORS = ("#d73027", "#fee08b", "#1a9850", "#7f7f7f", "#bdbdbd")
RURAL_CLASS_INDEX = {label: index for index, label in enumerate(RURAL_CLASSES)}

NUMERIC_FIELDS = (
    ("rural_composite_suitability_score", "Rural site score"),
    ("distance_to_listed_tourist_place", "Distance to listed tourist place"),
    ("nearby_mean_revenue", "Nearby mean revenue"),
    ("nearby_mean_occupancy", "Nearby mean occupancy"),
    ("competition_listing_count", "Listings in the cell and neighborhood"),
    ("rural_predicted_revenue", "Rural-model predicted revenue"),
    ("rural_predicted_occupancy", "Rural-model predicted occupancy"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attach_rural_classification(cells: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rural = gpd.read_parquet(RURAL_GRID_PATH)
    missing = [column for column in RURAL_COLUMNS if column not in rural.columns]
    assert not missing, (
        "the rural grid is missing site-score columns "
        f"{missing}; re-run the rural pipeline before building the viewer"
    )
    rural = rural[list(RURAL_COLUMNS)]
    cell_count = len(cells)
    merged = cells.merge(rural, on="cell_id", how="left", validate="one_to_one")
    assert len(merged) == cell_count, "the rural grid changed the cell count"
    assert set(merged["cell_class"]) == set(CELL_CLASS_KEYS), (
        "a grid cell carries an unknown rural cell class"
    )
    assert merged["rural_suitability_class"].notna().all(), (
        "a grid cell carries no rural suitability label"
    )
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=cells.crs)


def geometry_parts(geometry) -> list[list[list[list[float]]]]:
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    parts = []
    for polygon in polygons:
        rings = [[[round(lat, 6), round(lon, 6)] for lon, lat in polygon.exterior.coords]]
        for interior in polygon.interiors:
            rings.append([[round(lat, 6), round(lon, 6)] for lon, lat in interior.coords])
        parts.append(rings)
    return parts


def _optional_number(value, digits: int):
    if pd.isna(value):
        return None
    return round(float(value), digits)


def _feature_row(row) -> list[float] | None:
    if any(pd.isna(getattr(row, column)) for column in FEATURE_COLUMNS):
        return None
    return [round(float(getattr(row, column)), 4) for column in FEATURE_COLUMNS]


def _importance_rows(summary: dict, model_name: str) -> list[dict]:
    return [
        {
            "feature": row["feature"],
            "importance": round(float(row["importance_mean"]), 6),
        }
        for row in summary["models"][model_name]["importance"]
    ]


def _with_features(cells: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    missing = [column for column in FEATURE_COLUMNS if column not in cells.columns]
    if not missing:
        return cells
    features = pd.read_parquet(config.PROCESSED_DIR / "grid_features.parquet")
    merged = cells.merge(features[["cell_id", *missing]], on="cell_id", how="left", validate="one_to_one")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=cells.crs)


def _listing_dots() -> list[list]:
    targets = pd.read_parquet(TARGETS_PATH)
    active = targets.loc[targets["in_training_population"].astype(bool)]
    return [
        [
            round(float(row.longitude), 6),
            round(float(row.latitude), 6),
            str(row.listing_id),
        ]
        for row in active.itertuples(index=False)
    ]


def _place_dots() -> list[list]:
    places = load_listed_tourist_places(config.LISTED_TOURIST_PLACES_PATH)
    return [
        [
            round(float(row.longitude), 6),
            round(float(row.latitude), 6),
            str(row.name),
        ]
        for row in places.itertuples(index=False)
    ]


def build_payload() -> tuple[dict, dict]:
    cells = _with_features(
        attach_rural_classification(gpd.read_parquet(GRID_PATH).to_crs(config.GEOGRAPHIC_CRS))
    )
    places = _place_dots()
    listings = _listing_dots()
    rural_summary = json.loads(
        (config.RURAL_PROCESSED_DIR / "suitability_summary.json").read_text(encoding="utf-8")
    )
    rural_models = json.loads(
        (config.RURAL_PROCESSED_DIR / "model_summary.json").read_text(encoding="utf-8")
    )

    cell_records = []
    multipart_cells = 0
    for row in cells.itertuples():
        parts = geometry_parts(row.geometry)
        if len(parts) > 1:
            multipart_cells += 1
        cell_records.append(
            {
                "g": parts,
                "id": row.cell_id,
                "m": None if pd.isna(row.municipality) else str(row.municipality),
                "k": CELL_CLASS_KEYS[row.cell_class],
                "inr": bool(row.in_rural_analysis),
                "rc": RURAL_CLASS_INDEX[row.rural_suitability_class],
                "rs": _optional_number(row.rural_composite_suitability_score, 6),
                "ta": _optional_number(row.distance_to_listed_tourist_place, 3),
                "tn": None if pd.isna(row.nearest_listed_tourist_place) else str(row.nearest_listed_tourist_place),
                "mr": _optional_number(row.nearby_mean_revenue, 2),
                "mo": _optional_number(row.nearby_mean_occupancy, 5),
                "cc": None if pd.isna(row.competition_listing_count) else int(row.competition_listing_count),
                "lc": None if pd.isna(row.listings_in_cell) else int(row.listings_in_cell),
                "ln": None if pd.isna(row.listings_nearby) else int(row.listings_nearby),
                "rr": _optional_number(row.rural_predicted_revenue, 2),
                "ro": _optional_number(row.rural_predicted_occupancy, 5),
                "tac": _optional_number(row.tourist_access_contribution, 4),
                "nmc": _optional_number(row.nearby_market_contribution, 4),
                "lcc": _optional_number(row.local_competition_contribution, 4),
                "f": _feature_row(row),
            }
        )

    payload = {
        "ruralClasses": list(RURAL_CLASSES),
        "ruralClassColors": list(RURAL_CLASS_COLORS),
        "numericFields": [{"key": key, "label": label} for key, label in NUMERIC_FIELDS],
        "bounds": [
            [float(cells.total_bounds[1]), float(cells.total_bounds[0])],
            [float(cells.total_bounds[3]), float(cells.total_bounds[2])],
        ],
        "cells": cell_records,
        "places": places,
        "listings": listings,
        "weights": rural_summary["weights"],
        "weightingMethod": rural_summary["weighting_method"],
        "neighborhoodRadiusKm": rural_summary["neighborhood_radius_km"],
        "forest": {
            "features": [
                {"key": column, "label": FEATURE_LABELS[column]} for column in FEATURE_COLUMNS
            ],
            "revenueImportance": _importance_rows(rural_models, "revenue"),
            "occupancyImportance": _importance_rows(rural_models, "occupancy"),
        },
        "totalCells": int(len(cells)),
        "ruralCells": int((cells["cell_class"] == RURAL_CELL).sum()),
        "urbanCells": int((cells["cell_class"] == URBAN_CELL).sum()),
        "unclassifiedCells": int((cells["cell_class"] == UNCLASSIFIED_CELL).sum()),
        "outsideRuralCells": int((~cells["in_rural_analysis"].astype(bool)).sum()),
    }
    return payload, {
        "cells": len(cells),
        "multipart_cells": multipart_cells,
        "places": len(places),
        "listings": len(listings),
        "rural_cells": int((cells["cell_class"] == RURAL_CELL).sum()),
        "urban_cells": int((cells["cell_class"] == URBAN_CELL).sum()),
        "unclassified_cells": int((cells["cell_class"] == UNCLASSIFIED_CELL).sum()),
        "outside_rural_cells": int((~cells["in_rural_analysis"].astype(bool)).sum()),
    }


def mapbox_token() -> str:
    from_env = os.environ.get("MAPBOX_TOKEN", "").strip()
    if from_env:
        return from_env
    path = config.PROJECT_ROOT / ".mapbox_token"
    assert path.is_file(), (
        "a Mapbox public token is required in .mapbox_token or the MAPBOX_TOKEN environment variable"
    )
    token = path.read_text(encoding="utf-8").strip()
    assert token.startswith("pk."), "the Mapbox token must be a public token"
    return token


def render_html(payload: dict) -> str:
    template = (Path(__file__).parent / "visualisation_template.html").read_text()
    rendered = template.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    return rendered.replace("__MAPBOX_TOKEN__", mapbox_token())


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    before = sha256(GRID_PATH)
    rural_before = sha256(RURAL_GRID_PATH)

    payload, report = build_payload()
    (OUTPUT_DIR / "index.html").write_text(render_html(payload))

    after = sha256(GRID_PATH)
    rural_after = sha256(RURAL_GRID_PATH)
    size_kb = (OUTPUT_DIR / "index.html").stat().st_size / 1024

    print(json.dumps({**report, "grid_hash_before": before[:16], "grid_hash_after": after[:16],
                      "grid_unchanged": before == after,
                      "rural_grid_unchanged": rural_before == rural_after,
                      "html_kb": round(size_kb, 1)}, indent=2))
    print("rural class counts:", {label: sum(1 for c in payload["cells"] if c["rc"] == i)
                                  for i, label in enumerate(RURAL_CLASSES)})


if __name__ == "__main__":
    main()
