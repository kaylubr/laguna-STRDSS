import json

import numpy as np
import pandas as pd
import pytest

from str_suitability import config
from str_suitability.modeling.location_classifier import _group_splits
from str_suitability.modeling.presence import (
    PRESENCE_FEATURES,
    evaluate_presence,
    refresh_listed_places,
    select_training_rows,
)
from str_suitability.modeling.site_candidate_v2 import (
    ATTRACTION_CATEGORIES,
    FORBIDDEN_FEATURE_COLUMNS,
    V2_FEATURES,
    assert_feature_matrix,
    attraction_count_within_radius,
    dedupe_places,
    load_places_with_category,
    nearest_major_road_km,
    require_observed_values,
    research_use_decision,
)


def _square(cell_id: str, cell_class: str, municipality: str, listings: int) -> dict:
    return {
        "cell_id": cell_id,
        "cell_class": cell_class,
        "municipality": municipality,
        "listings_in_cell": listings,
        "listed_places_within_radius": 1,
        "distance_to_listed_tourist_place": 2.0,
        "road_distance_km": 0.2,
    }


def test_duplicate_place_id_is_kept_once_and_blank_ids_remain():
    frame = pd.DataFrame(
        [
            {"place_id": "a", "name": "First", "category": "Falls", "longitude": 121.0, "latitude": 14.0},
            {"place_id": "a", "name": "Second", "category": "Mountains", "longitude": 121.1, "latitude": 14.1},
            {"place_id": "", "name": "Blank", "category": "Falls", "longitude": 121.2, "latitude": 14.2},
            {"place_id": "b", "name": "Missing", "category": "Falls", "longitude": None, "latitude": None},
        ]
    )
    places, report = dedupe_places(frame)
    assert list(places["name"]) == ["First", "Blank"]
    assert report["duplicate_place_ids_removed"] == 1
    assert report["without_coordinates"] == 1
    assert report["places_used"] == 2


def test_major_road_distance_is_kilometres_after_projection_to_epsg_32651():
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import LineString, Point

    grid = geopandas.GeoDataFrame(
        {"cell_id": ["a"]},
        geometry=[Point(500000, 1500000).buffer(10)],
        crs="EPSG:32651",
    )
    roads = geopandas.GeoDataFrame(
        {"road_class": ["major"]},
        geometry=[LineString([(501000, 1500000), (501000, 1500100)])],
        crs="EPSG:32651",
    )
    projected = nearest_major_road_km(grid, roads)
    reprojected = nearest_major_road_km(grid.to_crs("EPSG:4326"), roads.to_crs("EPSG:4326"))
    assert projected.iloc[0] == pytest.approx(1.0, abs=0.001)
    assert reprojected.iloc[0] == pytest.approx(1.0, abs=0.001)


def test_place_distance_uses_kilometres():
    cells = pd.DataFrame({"longitude": [0.0], "latitude": [0.0]})
    places = pd.DataFrame({"longitude": [1.0], "latitude": [0.0], "name": ["east"]})
    refreshed = refresh_listed_places(cells, places)
    expected_km = config.EARTH_RADIUS_KM * np.deg2rad(1.0)
    assert refreshed["distance_to_listed_tourist_place"].iloc[0] == pytest.approx(expected_km, rel=1e-6)
    assert refreshed["distance_to_listed_tourist_place"].iloc[0] > 100


def test_missing_major_roads_stay_missing_and_are_not_filled_with_zero():
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import LineString, Point

    grid = geopandas.GeoDataFrame(
        {"cell_id": ["a"]},
        geometry=[Point(500000, 1500000).buffer(10)],
        crs="EPSG:32651",
    )
    roads = geopandas.GeoDataFrame(
        {"road_class": ["local"]},
        geometry=[LineString([(501000, 1500000), (501000, 1500100)])],
        crs="EPSG:32651",
    )
    distances = nearest_major_road_km(grid, roads)
    assert distances.isna().all()
    frame = pd.DataFrame({"distance_to_major_road_km": distances.to_numpy()})
    with pytest.raises(AssertionError, match="distance_to_major_road_km"):
        require_observed_values(frame, ["distance_to_major_road_km"])
    assert np.isnan(frame.iloc[0, 0])


def test_missing_town_center_is_not_replaced_with_a_constant():
    frame = pd.DataFrame({"distance_to_nearest_town_center_km": [1.2, np.nan]})
    with pytest.raises(AssertionError, match="distance_to_nearest_town_center_km"):
        require_observed_values(frame, ["distance_to_nearest_town_center_km"])
    assert np.isnan(frame.iloc[1, 0])


def test_attraction_absence_is_zero_and_a_missing_file_is_named():
    places, _report = dedupe_places(
        pd.DataFrame(
            [
                {
                    "place_id": "falls",
                    "name": "Falls",
                    "category": "Falls",
                    "longitude": 121.3,
                    "latitude": 14.2,
                }
            ]
        )
    )
    near = pd.DataFrame({"longitude": [121.3], "latitude": [14.2], "listings_in_cell": [0]})
    far = pd.DataFrame({"longitude": [120.0], "latitude": [10.0], "listings_in_cell": [0]})
    assert int(attraction_count_within_radius(near, places).iloc[0]) == 1
    assert int(attraction_count_within_radius(far, places).iloc[0]) == 0
    with pytest.raises(FileNotFoundError, match="poi_laguna"):
        load_places_with_category(config.PROJECT_ROOT / "missing-poi_laguna.json")


def test_forbidden_leakage_columns_cannot_enter_the_feature_matrix():
    assert set(V2_FEATURES).isdisjoint(FORBIDDEN_FEATURE_COLUMNS)
    assert_feature_matrix(V2_FEATURES)
    assert_feature_matrix(PRESENCE_FEATURES)
    with pytest.raises(AssertionError, match="forbidden leakage"):
        assert_feature_matrix(["listed_places_within_radius", "surrounding_mean_revenue"])
    with pytest.raises(AssertionError, match="fixed"):
        assert_feature_matrix([*V2_FEATURES, "built_up_share"])


def test_negative_sampling_uses_random_state_42():
    assert config.RANDOM_STATE == 42
    rows = [
        _square("listed-a", "rural", "A", 1),
        _square("listed-b", "rural", "B", 1),
        _square("empty-a", "rural", "A", 0),
        _square("empty-b", "rural", "B", 0),
        _square("empty-c", "rural", "C", 0),
        _square("empty-d", "rural", "D", 0),
    ]
    frame = pd.DataFrame(rows)
    sample, _report = select_training_rows(frame)
    again, _again_report = select_training_rows(frame)
    pool = frame.loc[frame["listings_in_cell"].eq(0)]
    expected = pool.sample(n=2, random_state=42)
    assert sample["cell_id"].tolist() == again["cell_id"].tolist()
    assert sample.loc[sample["presence"].eq(0), "cell_id"].tolist() == expected["cell_id"].tolist()


def test_outer_folds_do_not_share_a_municipality():
    rows = []
    for name in ("A", "B", "C", "D", "E", "F"):
        rows.append(_square(f"{name}-yes", "rural", name, 1))
        rows.append(_square(f"{name}-no", "rural", name, 0))
    sample, _report = select_training_rows(pd.DataFrame(rows))
    folds, scheme = _group_splits(
        sample[list(PRESENCE_FEATURES)],
        sample["presence"].astype(int),
        sample["municipality"].astype(str),
    )
    assert scheme == "stratified_group_kfold_5"
    for train_index, test_index in folds:
        train_names = set(sample.iloc[train_index]["municipality"])
        test_names = set(sample.iloc[test_index]["municipality"])
        assert train_names.isdisjoint(test_names)


def test_reimplemented_baseline_matches_the_stored_screen():
    training = pd.read_parquet(config.RURAL_PROCESSED_DIR / "presence_training.parquet")
    stored = json.loads((config.RURAL_PROCESSED_DIR / "presence_screen.json").read_text(encoding="utf-8"))
    replicated = evaluate_presence(training)
    documented = {"roc_auc": 0.634, "accuracy": 0.575, "macro_f1": 0.570}
    for metric, expected in documented.items():
        replicated_mean = replicated["forest"][metric]["mean"]
        stored_mean = float(stored["forest"][metric]["mean"])
        assert replicated_mean == pytest.approx(expected, abs=0.0015)
        assert replicated_mean == pytest.approx(stored_mean, abs=1e-9)


def test_empty_airbnb_cell_passes_the_place_features_without_a_listing_row():
    places, _report = dedupe_places(
        pd.DataFrame(
            [
                {
                    "place_id": "falls",
                    "name": "Falls",
                    "category": "Falls",
                    "longitude": 121.3,
                    "latitude": 14.2,
                },
                {
                    "place_id": "lake",
                    "name": "Lake",
                    "category": "Lakes & Ponds",
                    "longitude": 121.3,
                    "latitude": 14.2,
                },
            ]
        )
    )
    empty_cell = pd.DataFrame(
        {
            "cell_id": ["empty"],
            "longitude": [121.3],
            "latitude": [14.2],
            "listings_in_cell": [0],
        }
    )
    counts = attraction_count_within_radius(empty_cell, places)
    refreshed = refresh_listed_places(empty_cell, places[["longitude", "latitude", "name"]])
    assert "listing_id" not in empty_cell.columns
    assert int(counts.iloc[0]) == 1
    assert int(refreshed["listed_places_within_radius"].iloc[0]) == 2
    assert set(ATTRACTION_CATEGORIES) == {"Falls", "Mountains"}


def test_research_decision_follows_the_matched_fold_rule():
    assert research_use_decision([0.02, 0.01, 0.03, 0.01, -0.01], True, True) == "deploy v2 as a separate layer"
    assert research_use_decision([-0.02, -0.01, 0.03, -0.01, -0.04], True, True) == "retain baseline"
    assert research_use_decision([0.02, 0.01, -0.03, -0.01, 0.04], True, True) == "retain baseline"
    assert research_use_decision([0.02, 0.01, 0.03, 0.01, 0.04], False, False) == (
        "do not deploy either beyond research use"
    )
