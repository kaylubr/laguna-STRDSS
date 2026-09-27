import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from str_suitability.config import PROJECTED_CRS
from str_suitability.rural.suitability import (
    HAS_NEARBY_COLUMN,
    NEARBY_DISTANCE_COLUMN,
    attach_training_support,
)


def make_output() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "cell_id": ["c0", "c1"],
            "longitude": [121.0, 121.5],
            "latitude": [14.0, 14.0],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)],
        },
        geometry="geometry",
        crs=PROJECTED_CRS,
    )


def test_training_support_flags_near_cells_and_exposes_distance():
    result, report = attach_training_support(
        make_output(), pd.DataFrame({"longitude": [121.0], "latitude": [14.0]})
    )
    assert result[NEARBY_DISTANCE_COLUMN].notna().all()
    assert bool(result[HAS_NEARBY_COLUMN].iloc[0]) is True
    assert bool(result[HAS_NEARBY_COLUMN].iloc[1]) is False
    assert report["cells_with_nearby_training_data"] == 1
