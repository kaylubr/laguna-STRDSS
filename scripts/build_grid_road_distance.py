"""Distance from each existing grid centroid to the nearest mapped Laguna road.

The five-feature Random Forest is not fit or changed here.
"""

import json

import geopandas as gpd

from str_suitability.config import PROCESSED_DIR
from str_suitability.features.road_distance import (
    DISTANCE_CRS,
    DISTANCE_CRS_REASON,
    GRID_ROAD_DISTANCE_PATH,
    LAGUNA_ROADS_PATH,
    nearest_mapped_road_distance,
    validate_road_distances,
)

GRID_PATH = PROCESSED_DIR / "grid_features.parquet"


def main() -> None:
    grid = gpd.read_parquet(GRID_PATH)
    roads = gpd.read_file(LAGUNA_ROADS_PATH)
    distances = nearest_mapped_road_distance(grid, roads)
    summary = validate_road_distances(distances, expected_cells=len(grid))
    GRID_ROAD_DISTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    distances.to_parquet(GRID_ROAD_DISTANCE_PATH, index=False)
    report = {
        "cells": summary["cells"],
        "roads": int(len(roads)),
        "road_classes_included": sorted(roads["road_class"].dropna().astype(str).unique().tolist())
        if "road_class" in roads.columns
        else [],
        "highway_types_included": sorted(roads["highway_type"].dropna().astype(str).unique().tolist())
        if "highway_type" in roads.columns
        else [],
        "filtered_by_road_class": False,
        "distance_crs": DISTANCE_CRS,
        "distance_crs_reason": DISTANCE_CRS_REASON,
        "definition": (
            "Straight-line distance from the geometric centroid of each existing "
            "grid polygon to the nearest line in the mapped Laguna road network."
        ),
        "source": (
            "data/laguna_roads.geojson, an OpenStreetMap/Geofabrik extract of "
            "mapped Laguna roads. Not DPWH data and not a legal road classification."
        ),
        **summary,
    }
    (PROCESSED_DIR / "grid_road_distance_summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
