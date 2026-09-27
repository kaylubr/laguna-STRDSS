import geopandas as gpd
import numpy as np
import pandas as pd

from str_suitability.config import GEOGRAPHIC_CRS, PROJECTED_CRS
from str_suitability.features.accessibility import distance_to_nearest_km

POBLACION_PATTERN = r"poblaci[oó]n|\(pob"
METRES_PER_KM = 1000.0
POLYGON_NAME_COLUMN = "adm4_name"
MUNICIPALITY_NAME_COLUMN = "adm3_name"
LONGITUDE_COLUMN = "center_lon"
LATITUDE_COLUMN = "center_lat"

LAKE_DISTANCE_COLUMN = "distance_to_laguna_de_bay"
OTHER_WATER_DISTANCE_COLUMN = "distance_to_other_water"
POBLACION_DISTANCE_COLUMN = "distance_to_poblacion"
POBLACION_FALLBACK_NAME = "(municipality centroid fallback)"


def _projected(water: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return water.to_crs(PROJECTED_CRS)


def largest_water_body(water: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if len(water) == 0:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=PROJECTED_CRS)
    projected = _projected(water)
    areas = projected.geometry.area
    return gpd.GeoDataFrame(projected.loc[[areas.idxmax()]], geometry="geometry", crs=PROJECTED_CRS)


def other_water_bodies(water: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if len(water) == 0:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=PROJECTED_CRS)
    projected = _projected(water)
    areas = projected.geometry.area
    return gpd.GeoDataFrame(
        projected.drop(index=areas.idxmax()), geometry="geometry", crs=PROJECTED_CRS
    )


def distance_to_polygons_km(
    grid: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame
) -> np.ndarray:
    if len(polygons) == 0:
        return np.full(len(grid), np.nan)
    union = polygons.to_crs(PROJECTED_CRS).geometry.union_all()
    points = grid.geometry.representative_point().to_crs(PROJECTED_CRS)
    return points.distance(union).to_numpy(dtype=float) / METRES_PER_KM


def add_water_features(
    grid: gpd.GeoDataFrame, water: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    enriched = grid.copy()
    enriched[LAKE_DISTANCE_COLUMN] = distance_to_polygons_km(grid, largest_water_body(water))
    enriched[OTHER_WATER_DISTANCE_COLUMN] = distance_to_polygons_km(grid, other_water_bodies(water))
    return gpd.GeoDataFrame(enriched, geometry="geometry", crs=grid.crs)


def poblacion_anchors(barangays: gpd.GeoDataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = barangays.copy()
    matches = frame[POLYGON_NAME_COLUMN].fillna("").str.contains(
        POBLACION_PATTERN, case=False, regex=True
    )
    matched = frame.loc[matches]
    municipalities = sorted(frame[MUNICIPALITY_NAME_COLUMN].dropna().unique())
    matched_names = set(matched[MUNICIPALITY_NAME_COLUMN])
    missing = [name for name in municipalities if name not in matched_names]

    anchors = matched[
        [MUNICIPALITY_NAME_COLUMN, POLYGON_NAME_COLUMN, LONGITUDE_COLUMN, LATITUDE_COLUMN]
    ]
    fallback = _municipality_centroid_anchors(frame, missing)
    anchors = pd.concat([anchors, fallback], ignore_index=True)
    anchors = anchors.dropna(subset=[LONGITUDE_COLUMN, LATITUDE_COLUMN])
    assert len(anchors) > 0, "no Poblacion anchor could be derived for any municipality"

    return anchors, {
        "poblacion_barangays": int(matches.sum()),
        "municipalities": len(municipalities),
        "municipalities_without_poblacion": missing,
        "fallback_anchors": int(len(fallback)),
    }


def _municipality_centroid_anchors(
    frame: gpd.GeoDataFrame, missing: list[str]
) -> pd.DataFrame:
    records = []
    for name in missing:
        subset = frame.loc[frame[MUNICIPALITY_NAME_COLUMN] == name]
        if len(subset) == 0:
            continue
        centre = (
            gpd.GeoSeries([subset.to_crs(PROJECTED_CRS).geometry.union_all()], crs=PROJECTED_CRS)
            .to_crs(GEOGRAPHIC_CRS)
            .iloc[0]
            .representative_point()
        )
        records.append(
            {
                MUNICIPALITY_NAME_COLUMN: name,
                POLYGON_NAME_COLUMN: POBLACION_FALLBACK_NAME,
                LONGITUDE_COLUMN: centre.x,
                LATITUDE_COLUMN: centre.y,
            }
        )
    return pd.DataFrame(
        records,
        columns=[MUNICIPALITY_NAME_COLUMN, POLYGON_NAME_COLUMN, LONGITUDE_COLUMN, LATITUDE_COLUMN],
    )


def add_poblacion_distance(
    grid: gpd.GeoDataFrame, barangays: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
    anchors, report = poblacion_anchors(barangays)
    enriched = grid.copy()
    enriched[POBLACION_DISTANCE_COLUMN] = distance_to_nearest_km(
        grid["longitude"].to_numpy(),
        grid["latitude"].to_numpy(),
        anchors[LONGITUDE_COLUMN].to_numpy(dtype=float),
        anchors[LATITUDE_COLUMN].to_numpy(dtype=float),
    )
    return gpd.GeoDataFrame(enriched, geometry="geometry", crs=grid.crs), report
