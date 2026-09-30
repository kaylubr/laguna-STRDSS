"""Straight-line distance from each grid centroid to the nearest mapped road.

The roads are the project's OpenStreetMap/Geofabrik Laguna extract. They are
not a DPWH inventory and not a legal road classification. Every mapped road is
eligible. Distance is measured in EPSG:32651 metres, then stored in kilometres.
It is not a driving distance.
"""

import geopandas as gpd
import numpy as np
import pandas as pd

from str_suitability.config import DATA_DIR, PROCESSED_DIR, PROJECTED_CRS
from str_suitability.rural.classify import CELL_ID_COLUMN

ROAD_DISTANCE_M = "road_distance_m"
ROAD_DISTANCE_KM = "road_distance_km"
ROAD_DISTANCE_COLUMNS = (CELL_ID_COLUMN, ROAD_DISTANCE_M, ROAD_DISTANCE_KM)
LAGUNA_ROADS_PATH = DATA_DIR / "laguna_roads.geojson"
GRID_ROAD_DISTANCE_PATH = PROCESSED_DIR / "grid_road_distance.parquet"
DISTANCE_CRS = PROJECTED_CRS
DISTANCE_CRS_REASON = (
    "EPSG:32651 is WGS 84 / UTM zone 51N. Laguna lies between 120E and 126E, "
    "so this zone's axes are metres. The existing 1 km grid is already stored "
    "in this CRS. Longitude and latitude degrees are not used as distances."
)


def nearest_mapped_road_distance(
    grid: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """One row per cell: metres and kilometres from the cell centroid to the nearest mapped road."""
    assert grid.crs is not None, "the grid has no CRS"
    assert roads.crs is not None, "the road network has no CRS"
    assert CELL_ID_COLUMN in grid.columns, "the grid has no cell_id"
    assert grid[CELL_ID_COLUMN].is_unique, "the grid repeats a cell_id"
    assert len(roads) > 0, "the mapped road network is empty"

    projected_grid = grid.to_crs(DISTANCE_CRS)
    projected_roads = _all_mapped_roads(roads).to_crs(DISTANCE_CRS)
    _assert_metre_crs(projected_grid)
    _assert_metre_crs(projected_roads)
    centroids = gpd.GeoDataFrame(
        {CELL_ID_COLUMN: projected_grid[CELL_ID_COLUMN].astype(str).to_numpy()},
        geometry=projected_grid.geometry.centroid,
        crs=DISTANCE_CRS,
    )
    joined = centroids.sjoin_nearest(
        projected_roads[["geometry"]],
        how="left",
        distance_col=ROAD_DISTANCE_M,
    )
    joined = joined.sort_values(ROAD_DISTANCE_M).drop_duplicates(CELL_ID_COLUMN, keep="first")
    distances = centroids[[CELL_ID_COLUMN]].merge(
        joined[[CELL_ID_COLUMN, ROAD_DISTANCE_M]],
        on=CELL_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    distances[ROAD_DISTANCE_KM] = distances[ROAD_DISTANCE_M] / 1000.0
    distances = distances[list(ROAD_DISTANCE_COLUMNS)]
    validate_road_distances(distances, expected_cells=len(grid))
    return distances.reset_index(drop=True)


def validate_road_distances(
    distances: pd.DataFrame,
    expected_cells: int | None = None,
) -> dict[str, float | int]:
    """Check uniqueness, coverage, units, and the kilometre conversion."""
    missing = [column for column in ROAD_DISTANCE_COLUMNS if column not in distances.columns]
    assert not missing, f"road distances are missing columns: {missing}"
    assert distances[CELL_ID_COLUMN].is_unique, "road distances repeat a cell_id"
    assert distances[ROAD_DISTANCE_M].notna().all(), "a cell has no road distance in metres"
    assert distances[ROAD_DISTANCE_KM].notna().all(), "a cell has no road distance in kilometres"
    assert (distances[ROAD_DISTANCE_M] >= 0).all(), "a road distance is negative"
    assert np.allclose(
        distances[ROAD_DISTANCE_KM].to_numpy(dtype=float),
        distances[ROAD_DISTANCE_M].to_numpy(dtype=float) / 1000.0,
    ), "road_distance_km is not road_distance_m / 1000"
    if expected_cells is not None:
        assert len(distances) == expected_cells, (
            f"road distances cover {len(distances)} cells, expected {expected_cells}"
        )
    metres = distances[ROAD_DISTANCE_M].to_numpy(dtype=float)
    return {
        "cells": int(len(distances)),
        "min_m": float(metres.min()),
        "median_m": float(np.median(metres)),
        "mean_m": float(metres.mean()),
        "max_m": float(metres.max()),
        "min_km": float(metres.min() / 1000.0),
        "median_km": float(np.median(metres) / 1000.0),
        "mean_km": float(metres.mean() / 1000.0),
        "max_km": float(metres.max() / 1000.0),
    }


def _all_mapped_roads(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep every mapped road. road_class is not a filter."""
    usable = roads.loc[~roads.geometry.is_empty & roads.geometry.notna()].copy()
    assert len(usable) > 0, "every mapped road has an empty geometry"
    return usable


def _assert_metre_crs(frame: gpd.GeoDataFrame) -> None:
    assert frame.crs is not None and frame.crs.to_epsg() == 32651, (
        f"road distance was not calculated in EPSG:32651, got {frame.crs}"
    )
    unit = frame.crs.axis_info[0].unit_name.lower()
    assert unit in {"metre", "meter"}, f"the distance CRS unit is {unit}, not metres"
