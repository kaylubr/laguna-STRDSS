import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from str_suitability.config import GEOGRAPHIC_CRS, PROJECTED_CRS
from str_suitability.pipeline import FEATURE_COLUMNS
from str_suitability.rural.classify import CELL_ID_COLUMN
from str_suitability.rural.psa import RURAL, URBAN, URBAN_RURAL_COLUMN

CELL_SIZE_M = 1000
FEATURE_OFFSET = 10.0
LEGACY_GRID_COLUMNS = (
    "poi_density_total",
    "distance_to_nearest_tourist_attraction",
    "distance_to_nearest_transportation_facility",
    "population_density_per_km2",
)


def make_cells(count: int, cell_size_m: int = CELL_SIZE_M) -> gpd.GeoDataFrame:
    records = []
    for index in range(count):
        x = index * cell_size_m
        records.append(
            {
                CELL_ID_COLUMN: f"r0000c{index:04d}",
                "row": 0,
                "column": index,
                "geometry": box(x, 0.0, x + cell_size_m, cell_size_m),
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=PROJECTED_CRS)


def add_grid_features(frame: gpd.GeoDataFrame, offset: float = 0.0) -> gpd.GeoDataFrame:
    centroids = gpd.GeoSeries(frame.geometry.centroid, crs=PROJECTED_CRS)
    geographic = centroids.to_crs(GEOGRAPHIC_CRS)
    frame = frame.assign(longitude=geographic.x.to_numpy(), latitude=geographic.y.to_numpy())
    step = np.arange(len(frame), dtype=float)
    for position, name in enumerate([*FEATURE_COLUMNS, *LEGACY_GRID_COLUMNS]):
        frame[name] = FEATURE_OFFSET + offset + step * (position + 1.0) / len(FEATURE_COLUMNS)
    return gpd.GeoDataFrame(frame, geometry="geometry", crs=PROJECTED_CRS)


@pytest.fixture
def make_grid():
    def build(count: int = 12, offset: float = 0.0) -> gpd.GeoDataFrame:
        return add_grid_features(make_cells(count), offset)

    return build


@pytest.fixture
def make_barangays():
    def build(rural_cells: int = 8, urban_cells: int = 4) -> gpd.GeoDataFrame:
        records = []
        if rural_cells:
            records.append(
                {
                    "psgc": "0400000001",
                    "adm4_name": "Rural One",
                    URBAN_RURAL_COLUMN: RURAL,
                    "geometry": box(0.0, 0.0, rural_cells * CELL_SIZE_M, CELL_SIZE_M),
                }
            )
        if urban_cells:
            records.append(
                {
                    "psgc": "0400000002",
                    "adm4_name": "Urban One",
                    URBAN_RURAL_COLUMN: URBAN,
                    "geometry": box(
                        rural_cells * CELL_SIZE_M, 0.0, (rural_cells + urban_cells) * CELL_SIZE_M, CELL_SIZE_M
                    ),
                }
            )
        return gpd.GeoDataFrame(records, geometry="geometry", crs=PROJECTED_CRS)

    return build
