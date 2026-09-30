import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, box

from str_suitability.config import PROCESSED_DIR, PROJECTED_CRS
from str_suitability.features.road_distance import (
    GRID_ROAD_DISTANCE_PATH,
    ROAD_DISTANCE_COLUMNS,
    ROAD_DISTANCE_KM,
    ROAD_DISTANCE_M,
    nearest_mapped_road_distance,
    validate_road_distances,
)


def _cell(cell_id: str, x: float, y: float) -> dict:
    return {"cell_id": cell_id, "geometry": box(x - 10, y - 10, x + 10, y + 10)}


def test_nearest_mapped_road_uses_metres_and_keeps_local_roads():
    x, y = 316_000.0, 1_550_000.0
    grid = gpd.GeoDataFrame(
        [_cell("near", x, y), _cell("far", x + 1000, y)],
        geometry="geometry",
        crs=PROJECTED_CRS,
    )
    roads = gpd.GeoDataFrame(
        {
            "road_class": ["major", "local"],
            "highway_type": ["primary", "residential"],
            "geometry": [
                LineString([(x - 50, y + 500), (x + 50, y + 500)]),
                LineString([(x - 50, y + 100), (x + 2000, y + 100)]),
            ],
        },
        geometry="geometry",
        crs=PROJECTED_CRS,
    )

    first = nearest_mapped_road_distance(grid, roads)
    second = nearest_mapped_road_distance(grid, roads)

    assert list(first.columns) == list(ROAD_DISTANCE_COLUMNS)
    assert first["cell_id"].tolist() == ["near", "far"]
    # The local road is 100 m away. The major road is 500 m away.
    # A major-road-only filter would not return 100 m.
    assert first.loc[first["cell_id"] == "near", ROAD_DISTANCE_M].iloc[0] == pytest.approx(100)
    assert first.loc[first["cell_id"] == "far", ROAD_DISTANCE_M].iloc[0] == pytest.approx(100)
    assert first.loc[first["cell_id"] == "near", ROAD_DISTANCE_KM].iloc[0] == pytest.approx(0.1)
    assert first[ROAD_DISTANCE_M].tolist() == pytest.approx(second[ROAD_DISTANCE_M].tolist())
    summary = validate_road_distances(first, expected_cells=2)
    assert summary["min_m"] >= 0
    assert summary["cells"] == 2


def test_saved_grid_road_distances_cover_the_existing_cells():
    grid = gpd.read_parquet(PROCESSED_DIR / "grid_features.parquet")
    distances = pd.read_parquet(GRID_ROAD_DISTANCE_PATH)
    summary = validate_road_distances(distances, expected_cells=len(grid))
    assert summary["cells"] == 1764
    assert set(distances["cell_id"]) == set(grid["cell_id"].astype(str))
    assert grid.crs.to_epsg() == 32651
