import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from str_suitability.config import PROJECTED_CRS
from str_suitability.features.context import (
    POBLACION_FALLBACK_NAME,
    largest_water_body,
    other_water_bodies,
    poblacion_anchors,
)


def make_water() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "osm_id": [1, 2, 3],
            "geometry": [box(0, 0, 10_000, 10_000), box(20_000, 0, 21_000, 1_000), box(30_000, 0, 30_500, 500)],
        },
        geometry="geometry",
        crs=PROJECTED_CRS,
    )


def test_largest_water_body_is_the_biggest_polygon():
    largest = largest_water_body(make_water())
    assert len(largest) == 1
    assert largest["osm_id"].iloc[0] == 1


def test_other_water_bodies_exclude_the_largest():
    others = other_water_bodies(make_water())
    assert set(others["osm_id"]) == {2, 3}


def test_largest_water_body_of_empty_layer_is_empty():
    empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=PROJECTED_CRS)
    assert len(largest_water_body(empty)) == 0
    assert len(other_water_bodies(empty)) == 0


def make_barangays() -> gpd.GeoDataFrame:
    records = [
        {"adm3_name": "Town A", "adm4_name": "Poblacion", "center_lon": 121.0, "center_lat": 14.0},
        {"adm3_name": "Town A", "adm4_name": "Other", "center_lon": 121.1, "center_lat": 14.1},
        {"adm3_name": "Town B", "adm4_name": "San Agustin (Pob.)", "center_lon": 121.2, "center_lat": 14.2},
        {"adm3_name": "Town C", "adm4_name": "Remote", "center_lon": 121.3, "center_lat": 14.3},
    ]
    return gpd.GeoDataFrame(
        records,
        geometry=[box(x, y, x + 0.01, y + 0.01) for x, y in [(121, 14), (121.1, 14.1), (121.2, 14.2), (121.3, 14.3)]],
        crs="EPSG:4326",
    )


def test_poblacion_anchors_match_names_and_supply_a_fallback():
    anchors, report = poblacion_anchors(make_barangays())
    names = set(anchors["adm4_name"])
    assert "Poblacion" in names
    assert "San Agustin (Pob.)" in names
    assert report["municipalities_without_poblacion"] == ["Town C"]
    assert report["fallback_anchors"] == 1
    assert POBLACION_FALLBACK_NAME in names


def test_poblacion_anchors_report_counts_matched_barangays():
    _, report = poblacion_anchors(make_barangays())
    assert report["poblacion_barangays"] == 2
    assert report["municipalities"] == 3
