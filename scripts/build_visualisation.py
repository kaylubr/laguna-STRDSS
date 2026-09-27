import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from str_suitability import config
from str_suitability.rural.classify import RURAL_CELL, UNCLASSIFIED_CELL, URBAN_CELL
from str_suitability.rural.suitability import OUTSIDE_RURAL_ANALYSIS_LABELS, RURAL_CLASS_LABELS
from str_suitability.suitability.classify import CLASS_LABELS

FEATURE_DIR = config.PROJECT_ROOT / "assets" / "osm"
OUTPUT_DIR = config.PROCESSED_DIR / "frontend"
GRID_PATH = config.PROCESSED_DIR / "grid_suitability.parquet"
RURAL_GRID_PATH = config.RURAL_PROCESSED_DIR / "grid_suitability.parquet"
POI_PATH = FEATURE_DIR / "laguna_pois.parquet"
OBSERVATIONS_PATH = config.PROCESSED_DIR / "training_observations.parquet"

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
)

POI_CATEGORIES = (
    "commercial",
    "other_facilities",
    "recreation",
    "restaurants",
    "transportation",
    "tourist_attraction",
)

CLASS_COLORS = ("#a50026", "#f46d43", "#fee08b", "#a6d96a", "#1a9850")

RURAL_CLASSES = (*RURAL_CLASS_LABELS, *RURAL_OUTSIDE_LABELS)
RURAL_CLASS_COLORS = (*CLASS_COLORS, "#7f7f7f", "#bdbdbd")
RURAL_CLASS_INDEX = {label: index for index, label in enumerate(RURAL_CLASSES)}

CATEGORY_COLORS = {
    "commercial": "#1f77b4",
    "other_facilities": "#7f7f7f",
    "recreation": "#2ca02c",
    "restaurants": "#ff7f0e",
    "transportation": "#9467bd",
    "tourist_attraction": "#d62728",
}

NUMERIC_FIELDS = (
    ("composite_suitability_score", "Composite suitability score"),
    ("predicted_revenue", "Predicted annual revenue"),
    ("predicted_occupancy", "Predicted occupancy"),
    ("poi_density_total", "POI density"),
    ("distance_to_nearest_tourist_attraction", "Distance to nearest tourist attraction"),
    ("distance_to_nearest_transportation_facility", "Distance to nearest transportation facility"),
    ("poi_count_total", "POI count in cell"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attach_rural_classification(cells: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rural = gpd.read_parquet(RURAL_GRID_PATH)[list(RURAL_COLUMNS)]
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


def mark_contributors(cells: gpd.GeoDataFrame, pois: gpd.GeoDataFrame) -> pd.Series:
    inside = gpd.sjoin(
        pois.to_crs(cells.crs), cells[["cell_id", "geometry"]], predicate="within", how="inner"
    )
    return pois.index.isin(inside.index)


def geometry_parts(geometry) -> list[list[list[list[float]]]]:
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    parts = []
    for polygon in polygons:
        rings = [[[round(lat, 6), round(lon, 6)] for lon, lat in polygon.exterior.coords]]
        for interior in polygon.interiors:
            rings.append([[round(lat, 6), round(lon, 6)] for lon, lat in interior.coords])
        parts.append(rings)
    return parts


def build_payload() -> tuple[dict, dict]:
    cells = attach_rural_classification(
        gpd.read_parquet(GRID_PATH).to_crs(config.GEOGRAPHIC_CRS)
    )
    pois = gpd.read_parquet(POI_PATH)
    observations = pd.read_parquet(OBSERVATIONS_PATH)

    residential_counts = observations.groupby("cell_id").size()
    contributing = mark_contributors(cells, pois)

    class_index = {label: index for index, label in enumerate(CLASS_LABELS)}
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
                "c": class_index[row.suitability_class],
                "s": round(float(row.composite_suitability_score), 6),
                "r": round(float(row.predicted_revenue), 2),
                "o": round(float(row.predicted_occupancy), 5),
                "p": round(float(row.poi_density_total), 4),
                "a": round(float(row.distance_to_nearest_tourist_attraction), 4),
                "t": round(float(row.distance_to_nearest_transportation_facility), 4),
                "n": int(row.poi_count_total),
                "l": int(residential_counts.get(row.cell_id, 0)),
                "m": row.municipality,
                "k": CELL_CLASS_KEYS[row.cell_class],
                "inr": bool(row.in_rural_analysis),
                "rc": RURAL_CLASS_INDEX[row.rural_suitability_class],
                "rs": (
                    None
                    if pd.isna(row.rural_composite_suitability_score)
                    else round(float(row.rural_composite_suitability_score), 6)
                ),
            }
        )

    poi_records = []
    for row, is_contributing in zip(pois.itertuples(), contributing, strict=True):
        poi_records.append(
            [
                round(float(row.longitude), 6),
                round(float(row.latitude), 6),
                POI_CATEGORIES.index(str(row.category)),
                (str(row.name)[:70] if pd.notna(row.name) else ""),
                bool(row.is_tourist_attraction),
                bool(row.is_transport_facility),
                bool(is_contributing),
                str(row.osm_type),
                int(row.osm_id),
            ]
        )

    payload = {
        "classes": list(CLASS_LABELS),
        "classColors": list(CLASS_COLORS),
        "ruralClasses": list(RURAL_CLASSES),
        "ruralClassColors": list(RURAL_CLASS_COLORS),
        "categories": list(POI_CATEGORIES),
        "categoryColors": [CATEGORY_COLORS[name] for name in POI_CATEGORIES],
        "numericFields": [{"key": key, "label": label} for key, label in NUMERIC_FIELDS],
        "bounds": [
            [float(cells.total_bounds[1]), float(cells.total_bounds[0])],
            [float(cells.total_bounds[3]), float(cells.total_bounds[2])],
        ],
        "cells": cell_records,
        "pois": poi_records,
        "weights": json.loads(
            (config.PROCESSED_DIR / "suitability_summary.json").read_text()
        )["weights"],
        "cellsWithListings": int((residential_counts > 0).sum()),
        "totalCells": int(len(cells)),
        "cellCounts": [
            int(sum(1 for record in cell_records if record["c"] == index))
            for index in range(len(CLASS_LABELS))
        ],
        "poiTotal": int(len(pois)),
        "poiContributing": int(contributing.sum()),
        "ruralCells": int((cells["cell_class"] == RURAL_CELL).sum()),
        "urbanCells": int((cells["cell_class"] == URBAN_CELL).sum()),
        "unclassifiedCells": int((cells["cell_class"] == UNCLASSIFIED_CELL).sum()),
        "outsideRuralCells": int((~cells["in_rural_analysis"].astype(bool)).sum()),
    }
    return payload, {
        "cells": len(cells),
        "multipart_cells": multipart_cells,
        "pois_total": len(pois),
        "pois_contributing": int(contributing.sum()),
        "pois_excluded": int(len(pois) - contributing.sum()),
        "cells_with_listings": int((residential_counts > 0).sum()),
        "rural_cells": int((cells["cell_class"] == RURAL_CELL).sum()),
        "urban_cells": int((cells["cell_class"] == URBAN_CELL).sum()),
        "unclassified_cells": int((cells["cell_class"] == UNCLASSIFIED_CELL).sum()),
        "outside_rural_cells": int((~cells["in_rural_analysis"].astype(bool)).sum()),
    }


def render_html(payload: dict) -> str:
    template = (Path(__file__).parent / "visualisation_template.html").read_text()
    return template.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))


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
    print("\nclass counts:", {label: sum(1 for c in payload["cells"] if c["c"] == i)
                              for i, label in enumerate(CLASS_LABELS)})
    print("rural class counts:", {label: sum(1 for c in payload["cells"] if c["rc"] == i)
                                  for i, label in enumerate(RURAL_CLASSES)})


if __name__ == "__main__":
    main()
