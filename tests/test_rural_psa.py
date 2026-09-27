import json

import geopandas as gpd
import pytest
from shapely.geometry import box

from str_suitability.config import PROJECTED_CRS
from str_suitability.rural.psa import (
    RURAL,
    URBAN,
    URBAN_RURAL_COLUMN,
    classify_barangay_polygons,
    load_barangay_classification,
    load_barangay_polygons,
    report_barangay_classification,
)


def write_classification(tmp_path, records):
    path = tmp_path / "barangay_with_classification.json"
    path.write_text(json.dumps({"count": len(records), "results": records}))
    return path


def write_polygons(tmp_path, pcodes):
    path = tmp_path / "laguna.geojson"
    frame = gpd.GeoDataFrame(
        {
            "adm4_pcode": pcodes,
            "adm4_name": [f"Barangay {code}" for code in pcodes],
            "geometry": [box(index, 0, index + 1, 1) for index in range(len(pcodes))],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    frame.to_file(path, driver="GeoJSON")
    return path


def record(code, urban_rural, name="Barangay"):
    return {"code": code, "area_name": name, "geographic_level": "Bgy", URBAN_RURAL_COLUMN: urban_rural}


def test_psa_codes_are_normalised_stripped_and_padded(tmp_path):
    path = write_classification(
        tmp_path,
        [record("0403401001", RURAL), record("04340199", URBAN)],
    )
    frame = load_barangay_classification(path)
    assert frame["psgc"].tolist() == ["0403401001", "0434019900"]


def test_duplicate_psa_codes_are_rejected(tmp_path):
    path = write_classification(
        tmp_path, [record("0403401001", RURAL), record("PH0403401001", URBAN)]
    )
    with pytest.raises(AssertionError, match="must be unique"):
        load_barangay_classification(path)


def test_missing_urban_rural_classification_is_rejected(tmp_path):
    path = write_classification(tmp_path, [record("0403401001", None)])
    with pytest.raises(AssertionError, match="urban_rural"):
        load_barangay_classification(path)


def test_every_psa_barangay_carries_r_or_u(tmp_path):
    path = write_classification(
        tmp_path, [record("0403401001", RURAL), record("0403401002", URBAN)]
    )
    frame = load_barangay_classification(path)
    assert set(frame[URBAN_RURAL_COLUMN]) == {RURAL, URBAN}
    assert frame[URBAN_RURAL_COLUMN].notna().all()


def test_polygon_psgc_comes_from_adm4_pcode(tmp_path):
    path = write_polygons(tmp_path, ["PH0403401001", "PH0403401002"])
    polygons = load_barangay_polygons(path)
    assert polygons["psgc"].tolist() == ["0403401001", "0403401002"]
    assert polygons.crs == PROJECTED_CRS


def test_classification_is_joined_on_code_not_name(tmp_path):
    classification = load_barangay_classification(
        write_classification(
            tmp_path,
            [
                {"code": "0403401001", "area_name": "Del Carmen", "geographic_level": "Bgy", URBAN_RURAL_COLUMN: RURAL},
                {"code": "0403401002", "area_name": "Renamed Since PSA", "geographic_level": "Bgy", URBAN_RURAL_COLUMN: URBAN},
            ],
        )
    )
    polygons = load_barangay_polygons(write_polygons(tmp_path, ["PH0403401001", "PH0403401002"]))
    classified = classify_barangay_polygons(polygons, classification)
    assert classified[URBAN_RURAL_COLUMN].tolist() == [RURAL, URBAN]


def test_polygon_without_a_psa_record_is_left_unclassified(tmp_path):
    classification = load_barangay_classification(
        write_classification(tmp_path, [record("0403401001", RURAL)])
    )
    polygons = load_barangay_polygons(
        write_polygons(tmp_path, ["PH0403401001", "PH0403418901"])
    )
    report = report_barangay_classification(
        classify_barangay_polygons(polygons, classification), classification
    )
    assert report["polygons_without_psa_record"] == 1
    assert report["polygons_classified"] == 1
    assert report["psa_codes_without_polygon"] == 0


def test_report_counts_psa_rural_and_urban(tmp_path):
    classification = load_barangay_classification(
        write_classification(
            tmp_path,
            [record("0403401001", RURAL), record("0403401002", URBAN), record("0403401003", RURAL)],
        )
    )
    polygons = load_barangay_polygons(
        write_polygons(tmp_path, ["PH0403401001", "PH0403401002", "PH0403401003"])
    )
    report = report_barangay_classification(
        classify_barangay_polygons(polygons, classification), classification
    )
    assert report["psa_barangays"] == 3
    assert report["psa_rural"] == 2
    assert report["psa_urban"] == 1
    assert report["polygons_classified"] == 3
