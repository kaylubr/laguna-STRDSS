import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from str_suitability.config import GEOGRAPHIC_CRS, POI_BLOC_INDICATORS, SITE_SCORE_WEIGHTS
from str_suitability.rural.site_score import (
    ACTIVE_LISTINGS_WITHIN_RADIUS,
    COMPETITION_COLUMN,
    DISTANCE_COLUMN,
    LISTED_PLACES_WITHIN_RADIUS,
    NEARBY_OCCUPANCY_COLUMN,
    NEARBY_REVENUE_COLUMN,
    attach_forest_features,
    compute_site_score,
    load_listed_tourist_places,
)
from str_suitability.rural.suitability import RURAL_SCORE_COLUMN

RADIUS_KM = 5.0


def _cell(cell_id: str, longitude: float, latitude: float) -> dict:
    return {
        "cell_id": cell_id,
        "longitude": longitude,
        "latitude": latitude,
        "geometry": box(longitude - 0.004, latitude - 0.004, longitude + 0.004, latitude + 0.004),
    }


def _listing(longitude: float, latitude: float, revenue: float, occupancy: float) -> dict:
    return {
        "longitude": longitude,
        "latitude": latitude,
        "ttm_revenue": revenue,
        "ttm_occupancy": occupancy,
    }


def _places(*points: tuple[float, float, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"longitude": longitude, "latitude": latitude, "name": name}
            for longitude, latitude, name in points
        ]
    )


def _grid(*cells: dict) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(list(cells), geometry="geometry", crs=GEOGRAPHIC_CRS)


def _fillers(count: int = 5, latitude: float = 15.2) -> tuple[list[dict], list[tuple], list[dict]]:
    cells, places, listings = [], [], []
    for index in range(count):
        longitude = 120.2 + index * 0.2
        cells.append(_cell(f"filler_{index}", longitude, latitude))
        places.append((longitude, latitude, f"Filler {index}"))
        listings.append(_listing(longitude, latitude, 10_000.0 * (index + 1), 0.05 * (index + 1)))
    return cells, places, listings


def test_forest_features_use_only_listed_places_and_listings():
    cells = _grid(_cell("near", 121.30, 14.20), _cell("far", 121.30, 14.50))
    places = _places((121.30, 14.20, "Near falls"), (121.305, 14.20, "Second falls"))
    listings = pd.DataFrame(
        [
            _listing(121.30, 14.20, 200_000.0, 0.8),
            _listing(121.301, 14.20, 80_000.0, 0.4),
        ]
    )

    featured = attach_forest_features(cells, listings, places, radius_km=RADIUS_KM)
    by_cell = featured.set_index("cell_id")

    assert by_cell.loc["near", DISTANCE_COLUMN] == pytest.approx(0.0, abs=1e-6)
    assert by_cell.loc["near", LISTED_PLACES_WITHIN_RADIUS] == 2
    assert by_cell.loc["near", ACTIVE_LISTINGS_WITHIN_RADIUS] == 2
    assert by_cell.loc["far", DISTANCE_COLUMN] > by_cell.loc["near", DISTANCE_COLUMN]
    assert by_cell.loc["far", ACTIVE_LISTINGS_WITHIN_RADIUS] == 0
    assert set(featured.columns).isdisjoint(
        {
            "poi_density_restaurants",
            "poi_density_recreation",
            "distance_to_nearest_tourist_attraction",
        }
    )


def test_listed_places_keep_coordinates_and_drop_ratings(tmp_path):
    path = tmp_path / "poi_laguna.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "Mt. Example",
                    "latitude": 14.2,
                    "longitude": 121.3,
                    "rating": 4.8,
                    "status": "OK",
                },
                {"name": "Missing", "latitude": None, "longitude": None},
            ]
        ),
        encoding="utf-8",
    )

    places = load_listed_tourist_places(path)

    assert list(places["name"]) == ["Mt. Example"]
    assert "rating" not in places.columns


def test_closer_tourist_access_and_stronger_market_raise_the_score():
    filler_cells, filler_places, filler_listings = _fillers()
    cells = _grid(
        _cell("near", 121.30, 14.20),
        _cell("quiet", 121.30, 14.50),
        _cell("crowded", 121.50, 14.20),
        *filler_cells,
    )
    places = _places(
        (121.30, 14.20, "Near falls"),
        (121.50, 14.20, "Crowded falls"),
        *filler_places,
    )
    listings = pd.DataFrame(
        [
            _listing(121.30, 14.20, 200_000.0, 0.8),
            _listing(121.30, 14.50, 40_000.0, 0.2),
            _listing(121.50, 14.20, 200_000.0, 0.8),
            _listing(121.51, 14.20, 200_000.0, 0.8),
            _listing(121.49, 14.20, 200_000.0, 0.8),
            _listing(121.50, 14.21, 200_000.0, 0.8),
            _listing(121.50, 14.19, 200_000.0, 0.8),
            *filler_listings,
        ]
    )

    scored, report = compute_site_score(cells, listings, places, radius_km=RADIUS_KM)
    by_cell = scored.set_index("cell_id")

    assert by_cell.loc["near", DISTANCE_COLUMN] == pytest.approx(0.0, abs=1e-6)
    assert by_cell.loc["near", NEARBY_REVENUE_COLUMN] == pytest.approx(
        by_cell.loc["crowded", NEARBY_REVENUE_COLUMN]
    )
    assert by_cell.loc["crowded", COMPETITION_COLUMN] > by_cell.loc["near", COMPETITION_COLUMN]
    assert by_cell.loc["near", RURAL_SCORE_COLUMN] > by_cell.loc["crowded", RURAL_SCORE_COLUMN]
    assert by_cell.loc["near", RURAL_SCORE_COLUMN] > by_cell.loc["quiet", RURAL_SCORE_COLUMN]
    assert by_cell.loc["quiet", NEARBY_REVENUE_COLUMN] < by_cell.loc["near", NEARBY_REVENUE_COLUMN]
    assert set(report["weights"]) == set(SITE_SCORE_WEIGHTS)
    assert report["weights"]["distance_to_listed_tourist_place"] == pytest.approx(1.0 / 3.0)
    assert (
        report["weights"]["nearby_mean_revenue"] + report["weights"]["nearby_mean_occupancy"]
        == pytest.approx(1.0 / 3.0)
    )
    assert report["weights"]["competition_listing_count"] == pytest.approx(1.0 / 3.0)
    assert set(report["excluded_osm_poi_densities"]) == set(POI_BLOC_INDICATORS)
    assert set(report["weights"]).isdisjoint(POI_BLOC_INDICATORS)


def test_a_cell_with_no_nearby_listings_gets_a_zero_market():
    filler_cells, filler_places, filler_listings = _fillers()
    cells = _grid(
        _cell("empty", 121.10, 14.20),
        _cell("served", 121.40, 14.20),
        *filler_cells,
    )
    places = _places((121.40, 14.20, "Served falls"), *filler_places)
    listings = pd.DataFrame([_listing(121.40, 14.20, 150_000.0, 0.6), *filler_listings])

    scored, report = compute_site_score(cells, listings, places, radius_km=RADIUS_KM)
    by_cell = scored.set_index("cell_id")

    assert by_cell.loc["empty", NEARBY_REVENUE_COLUMN] == 0.0
    assert by_cell.loc["empty", NEARBY_OCCUPANCY_COLUMN] == 0.0
    assert by_cell.loc["empty", COMPETITION_COLUMN] == 0
    assert by_cell.loc["served", NEARBY_REVENUE_COLUMN] == pytest.approx(150_000.0)
    assert report["cells_with_no_nearby_listings"] == 1
