import hashlib
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from str_suitability import config
from str_suitability.features.road_distance import (
    GRID_ROAD_DISTANCE_PATH,
    LAGUNA_ROADS_PATH,
    ROAD_DISTANCE_KM,
)
from str_suitability.modeling.location_classifier import (
    DISTANCE_COLUMN,
    LISTED_PLACES_WITHIN_RADIUS,
    SURROUNDING_LISTINGS,
    SURROUNDING_OCCUPANCY,
    NEAREST_PLACE_COLUMN,
)
from str_suitability.modeling.presence import (
    PRESENCE_CLASSES,
    PRESENCE_FEATURES,
    listed_places_for_training,
    presence_forest,
)
from str_suitability.rural.classify import RURAL_CELL, UNCLASSIFIED_CELL, URBAN_CELL
from str_suitability.rural.diagrams import classifier_tree_trace

OUTPUT_DIR = config.PROCESSED_DIR / "frontend"
GRID_PATH = config.PROCESSED_DIR / "grid_features.parquet"
RURAL_GRID_PATH = config.RURAL_PROCESSED_DIR / "grid_suitability.parquet"
TARGETS_PATH = config.INTERIM_DIR / "airroi_targets.parquet"

CELL_CLASS_KEYS = {RURAL_CELL: 0, URBAN_CELL: 1, UNCLASSIFIED_CELL: 2}

MAP_COLUMNS = (
    "cell_id",
    "cell_class",
    "in_rural_analysis",
    SURROUNDING_OCCUPANCY,
    SURROUNDING_LISTINGS,
)

FEATURE_LABELS = {
    LISTED_PLACES_WITHIN_RADIUS: "Listed landscape places within 5 km",
    DISTANCE_COLUMN: "Distance to nearest listed landscape place",
    ROAD_DISTANCE_KM: "Distance to nearest mapped road",
}

DISPLAY_CLASSES = ("Not the pattern", "Uncertain", "Looks listed")
DISPLAY_COLORS = ("#d73027", "#fee08b", "#1a9850")
CANDIDATE_CLASSES = (
    "Lower candidate-pattern score",
    "Middle candidate-pattern score",
    "Higher candidate-pattern score",
)
CANDIDATE_NOTE = (
    "A site-pattern screening score based on observable location characteristics. "
    "It is not a forecast of revenue, occupancy, investment return, or regulatory suitability."
)
LOW_BAND = 0.40
HIGH_BAND = 0.60

NUMERIC_FIELDS = (
    ("presence_probability", "P(looks listed)"),
    (SURROUNDING_OCCUPANCY, "Nearby occupancy, flag only"),
    (SURROUNDING_LISTINGS, "Nearby listings, flag only"),
    (DISTANCE_COLUMN, "Distance to listed tourist place"),
    (LISTED_PLACES_WITHIN_RADIUS, "Listed places within 5 km"),
    (ROAD_DISTANCE_KM, "Distance to nearest mapped road (km)"),
)


def _display_band(probability: float) -> int:
    """Red below 0.40, yellow from 0.40 up to 0.60, green at 0.60 and above."""
    if probability < LOW_BAND:
        return 0
    if probability < HIGH_BAND:
        return 1
    return 2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attach_rural_classification(cells: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rural = gpd.read_parquet(RURAL_GRID_PATH)
    missing = [column for column in MAP_COLUMNS if column not in rural.columns]
    assert not missing, (
        "the suitability grid is missing random-forest columns "
        f"{missing}; re-run scripts/run_rural.py before building the viewer"
    )
    rural = rural[list(MAP_COLUMNS)]
    overlap = [column for column in rural.columns if column != "cell_id" and column in cells.columns]
    cell_count = len(cells)
    merged = cells.drop(columns=overlap).merge(rural, on="cell_id", how="left", validate="one_to_one")
    assert len(merged) == cell_count, "the suitability grid changed the cell count"
    assert set(merged["cell_class"]) == set(CELL_CLASS_KEYS), (
        "a grid cell carries an unknown rural cell class"
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


def _tree_values(row) -> list[float]:
    values = []
    for column in PRESENCE_FEATURES:
        measurement = getattr(row, column)
        values.append(np.nan if pd.isna(measurement) else float(measurement))
    return values


def _feature_row(row) -> list[float | None]:
    values = []
    for column in PRESENCE_FEATURES:
        measurement = getattr(row, column)
        values.append(None if pd.isna(measurement) else round(float(measurement), 4))
    return values


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


def _road_lines() -> list[list[list[float]]]:
    """Mapped roads inside the grid, simplified in metres. Not a model input."""
    roads = gpd.read_file(LAGUNA_ROADS_PATH).to_crs(config.PROJECTED_CRS)
    grid = gpd.read_parquet(GRID_PATH).to_crs(config.PROJECTED_CRS)
    min_x, min_y, max_x, max_y = grid.total_bounds
    pad = 500
    roads = roads.cx[min_x - pad : max_x + pad, min_y - pad : max_y + pad]
    roads = roads.loc[~roads.geometry.is_empty & roads.geometry.notna()].copy()
    roads["geometry"] = roads.geometry.simplify(25)
    geographic = roads.to_crs(config.GEOGRAPHIC_CRS)
    lines = []
    for geometry in geographic.geometry:
        if geometry.geom_type == "LineString":
            parts = [geometry]
        elif geometry.geom_type == "MultiLineString":
            parts = list(geometry.geoms)
        else:
            continue
        for part in parts:
            if part.is_empty:
                continue
            coords = [[round(lat, 5), round(lon, 5)] for lon, lat in part.coords]
            if len(coords) >= 2:
                lines.append(coords)
    return lines


def _place_dots() -> list[list]:
    places, _report = listed_places_for_training(config.LISTED_TOURIST_PLACES_PATH)
    return [
        [
            round(float(row.longitude), 6),
            round(float(row.latitude), 6),
            str(row.name),
        ]
        for row in places.itertuples(index=False)
    ]


def _map_forest():
    """The presence forest saved for the map. A mismatch means the tree path is not that forest."""
    labeled = pd.read_parquet(config.RURAL_PROCESSED_DIR / "presence_training.parquet")
    model = presence_forest()
    model.fit(labeled[list(PRESENCE_FEATURES)], labeled["presence"].astype(int))
    return model


def _assert_same_forest(model, cells: pd.DataFrame) -> None:
    fresh = model.predict_proba(cells[list(PRESENCE_FEATURES)])
    column = list(model.classes_).index(1)
    assert np.allclose(cells["presence_probability"].to_numpy(), fresh[:, column]), (
        "the refit forest does not match the probabilities on the map"
    )


def build_payload() -> tuple[dict, dict]:
    cells = attach_rural_classification(gpd.read_parquet(GRID_PATH).to_crs(config.GEOGRAPHIC_CRS))
    presence = pd.read_parquet(config.RURAL_PROCESSED_DIR / "presence_scores.parquet")
    presence["cell_id"] = presence["cell_id"].astype(str)
    cells["cell_id"] = cells["cell_id"].astype(str)
    overlap = [column for column in presence.columns if column != "cell_id" and column in cells.columns]
    cells = cells.drop(columns=overlap).merge(presence, on="cell_id", how="left", validate="one_to_one")
    assert cells["presence_class"].isin(PRESENCE_CLASSES).all(), "a square has no presence class"
    assert cells["presence_probability"].notna().all(), "a square has no presence probability"
    distances = pd.read_parquet(GRID_ROAD_DISTANCE_PATH)
    distances["cell_id"] = distances["cell_id"].astype(str)
    cells = cells.merge(distances, on="cell_id", how="left", validate="one_to_one")
    assert cells[ROAD_DISTANCE_KM].notna().all(), "a map cell has no distance to a mapped road"
    candidate = pd.read_parquet(config.RURAL_PROCESSED_DIR / "site_candidate_v2_scores.parquet")
    candidate["cell_id"] = candidate["cell_id"].astype(str)
    assert set(candidate["model_name"]) == {"v2_logistic_regression"}
    cells = cells.merge(
        candidate[["cell_id", "candidate_pattern_score"]],
        on="cell_id",
        how="left",
        validate="one_to_one",
    )
    rural_mask = cells["cell_class"] == RURAL_CELL
    assert cells.loc[rural_mask, "candidate_pattern_score"].notna().all(), (
        "a rural square has no candidate-pattern score"
    )
    forest = _map_forest()
    _assert_same_forest(forest, cells)
    places = _place_dots()
    listings = _listing_dots()
    roads = _road_lines()
    presence_report = json.loads(
        (config.RURAL_PROCESSED_DIR / "presence_screen.json").read_text(encoding="utf-8")
    )
    matrix = presence_report["forest"]["confusion_matrix"]
    listed_kept = int(matrix[1][1])
    listed_true = int(sum(matrix[1]))

    cell_records = []
    multipart_cells = 0
    for row in cells.itertuples():
        parts = geometry_parts(row.geometry)
        if len(parts) > 1:
            multipart_cells += 1
        probability = float(row.presence_probability)
        cell_records.append(
            {
                "g": parts,
                "id": row.cell_id,
                "m": None if pd.isna(row.municipality) else str(row.municipality),
                "k": CELL_CLASS_KEYS[row.cell_class],
                "inr": bool(row.in_rural_analysis),
                "rc": _display_band(probability),
                "fc": 0 if probability < 0.5 else 1,
                "pl": round(1.0 - probability, 4),
                "pp": round(probability, 4),
                "ta": _optional_number(row.distance_to_listed_tourist_place, 3),
                "tn": None if pd.isna(row.nearest_listed_tourist_place) else str(row.nearest_listed_tourist_place),
                "so": _optional_number(row.surrounding_mean_occupancy, 5),
                "sl": None if pd.isna(row.surrounding_listing_count) else int(row.surrounding_listing_count),
                "pc": None if pd.isna(row.listed_places_within_radius) else int(row.listed_places_within_radius),
                "rd": _optional_number(row.road_distance_km, 3),
                "cs": _optional_number(row.candidate_pattern_score, 4),
                "cc": None if pd.isna(row.candidate_pattern_score) else _display_band(float(row.candidate_pattern_score)),
                "tp": classifier_tree_trace(forest, _tree_values(row)),
                "f": _feature_row(row),
            }
        )

    payload = {
        "ruralClasses": list(DISPLAY_CLASSES),
        "ruralClassColors": list(DISPLAY_COLORS),
        "candidateClasses": list(CANDIDATE_CLASSES),
        "candidateClassColors": list(DISPLAY_COLORS),
        "candidateNote": CANDIDATE_NOTE,
        "candidateAuc": json.loads(
            (config.RURAL_PROCESSED_DIR / "site_candidate_v2_report.json").read_text(encoding="utf-8")
        )["metrics"]["v2_logistic_regression"]["roc_auc"]["mean"],
        "voteClasses": list(PRESENCE_CLASSES),
        "voteColors": ["#d73027", "#1a9850"],
        "treeLabels": ["Places", "Place km", "Road km"],
        "numericFields": [{"key": key, "label": label} for key, label in NUMERIC_FIELDS],
        "bounds": [
            [float(cells.total_bounds[1]), float(cells.total_bounds[0])],
            [float(cells.total_bounds[3]), float(cells.total_bounds[2])],
        ],
        "cells": cell_records,
        "places": places,
        "listings": listings,
        "roads": roads,
        "neighborhoodRadiusKm": presence_report.get("places", {}).get("places_used"),
        "presenceAuc": presence_report["forest"]["roc_auc"]["mean"],
        "guessAuc": presence_report["stratified_baseline"]["roc_auc"]["mean"],
        "listedKept": listed_kept,
        "listedTrue": listed_true,
        "placesUsed": presence_report["places"]["places_used"],
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
        "roads": len(roads),
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
    rural = [cell for cell in payload["cells"] if cell["k"] == 0]
    print("rural class counts:", {label: sum(1 for cell in rural if cell["rc"] == i)
                                  for i, label in enumerate(DISPLAY_CLASSES)})
    print("candidate pattern counts:", {label: sum(1 for cell in rural if cell["cc"] == i)
                                        for i, label in enumerate(CANDIDATE_CLASSES)})
    print("rural market evidence:", {
        "existing": sum(1 for cell in rural if (cell["sl"] or 0) > 0),
        "none": sum(1 for cell in rural if not (cell["sl"] or 0) > 0),
    })
    print("listed kept:", payload["listedKept"], "of", payload["listedTrue"])


if __name__ == "__main__":
    main()
