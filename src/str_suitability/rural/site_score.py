import json

import geopandas as gpd
import numpy as np
import pandas as pd

from str_suitability.config import (
    GEOGRAPHIC_CRS,
    LOCAL_COMPETITION_BLOC,
    NEARBY_MARKET_BLOC,
    NEIGHBORHOOD_RADIUS_KM,
    POI_BLOC_INDICATORS,
    SITE_SCORE_DIRECTIONS,
    SITE_SCORE_WEIGHTS,
    TOURIST_ACCESS_BLOC,
)
from str_suitability.rural.classify import CELL_ID_COLUMN
from str_suitability.rural.suitability import RURAL_CLASS_COLUMN, RURAL_SCORE_COLUMN
from str_suitability.spatial.haversine import haversine_km
from str_suitability.suitability.classify import classify_traffic_light
from str_suitability.suitability.composite_score import composite_suitability_score
from str_suitability.suitability.normalize import minimum_maximum_normalize, normalized_column

DISTANCE_COLUMN = "distance_to_listed_tourist_place"
LISTED_PLACES_WITHIN_RADIUS = "listed_places_within_radius"
ACTIVE_LISTINGS_WITHIN_RADIUS = "active_listings_within_radius"
FOREST_FEATURE_COLUMNS = (
    DISTANCE_COLUMN,
    LISTED_PLACES_WITHIN_RADIUS,
    ACTIVE_LISTINGS_WITHIN_RADIUS,
)
NEAREST_PLACE_COLUMN = "nearest_listed_tourist_place"
NEARBY_REVENUE_COLUMN = "nearby_mean_revenue"
NEARBY_OCCUPANCY_COLUMN = "nearby_mean_occupancy"
LISTINGS_IN_CELL_COLUMN = "listings_in_cell"
LISTINGS_NEARBY_COLUMN = "listings_nearby"
COMPETITION_COLUMN = "competition_listing_count"
TOURIST_ACCESS_CONTRIBUTION = "tourist_access_contribution"
MARKET_CONTRIBUTION = "nearby_market_contribution"
COMPETITION_CONTRIBUTION = "local_competition_contribution"

LONGITUDE_COLUMN = "longitude"
LATITUDE_COLUMN = "latitude"
REVENUE_COLUMN = "ttm_revenue"
OCCUPANCY_COLUMN = "ttm_occupancy"
WEIGHTING_METHOD = "reserved_blocs"

SITE_COLUMNS = (
    CELL_ID_COLUMN,
    RURAL_SCORE_COLUMN,
    RURAL_CLASS_COLUMN,
    DISTANCE_COLUMN,
    NEAREST_PLACE_COLUMN,
    NEARBY_REVENUE_COLUMN,
    NEARBY_OCCUPANCY_COLUMN,
    LISTINGS_IN_CELL_COLUMN,
    LISTINGS_NEARBY_COLUMN,
    COMPETITION_COLUMN,
    TOURIST_ACCESS_CONTRIBUTION,
    MARKET_CONTRIBUTION,
    COMPETITION_CONTRIBUTION,
)


def load_listed_tourist_places(path) -> pd.DataFrame:
    records = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(records, list) and records, f"{path} does not contain a list of places"
    places = pd.DataFrame(records)
    places[LONGITUDE_COLUMN] = pd.to_numeric(places.get(LONGITUDE_COLUMN), errors="coerce")
    places[LATITUDE_COLUMN] = pd.to_numeric(places.get(LATITUDE_COLUMN), errors="coerce")
    places = places.dropna(subset=[LONGITUDE_COLUMN, LATITUDE_COLUMN]).reset_index(drop=True)
    assert len(places) > 0, f"{path} has no places with coordinates"
    places["name"] = places.get("name", pd.Series("", index=places.index)).fillna("").astype(str)
    empty_name = places["name"].str.strip() == ""
    places.loc[empty_name, "name"] = "Unnamed place"
    return places[[LONGITUDE_COLUMN, LATITUDE_COLUMN, "name"]].copy()


def attach_forest_features(
    grid: gpd.GeoDataFrame,
    listings: pd.DataFrame,
    places: pd.DataFrame,
    radius_km: float = NEIGHBORHOOD_RADIUS_KM,
) -> gpd.GeoDataFrame:
    assert radius_km > 0, f"neighborhood radius must be positive, got {radius_km}"
    _assert_coordinates(grid, "grid cell")
    _assert_coordinates(listings, "active listing")
    _assert_coordinates(places, "listed tourist place")
    assert len(listings) > 0, "there are no active listings to describe the forest features"
    assert len(places) > 0, "there are no listed tourist places to describe the forest features"

    place_distances = haversine_km(
        grid[LONGITUDE_COLUMN].to_numpy()[:, None],
        grid[LATITUDE_COLUMN].to_numpy()[:, None],
        places[LONGITUDE_COLUMN].to_numpy()[None, :],
        places[LATITUDE_COLUMN].to_numpy()[None, :],
    )
    listing_distances = haversine_km(
        grid[LONGITUDE_COLUMN].to_numpy()[:, None],
        grid[LATITUDE_COLUMN].to_numpy()[:, None],
        listings[LONGITUDE_COLUMN].to_numpy()[None, :],
        listings[LATITUDE_COLUMN].to_numpy()[None, :],
    )
    enriched = grid.copy()
    enriched[DISTANCE_COLUMN] = place_distances.min(axis=1)
    enriched[LISTED_PLACES_WITHIN_RADIUS] = (place_distances <= radius_km).sum(axis=1).astype(int)
    enriched[ACTIVE_LISTINGS_WITHIN_RADIUS] = (listing_distances <= radius_km).sum(axis=1).astype(int)
    assert enriched[DISTANCE_COLUMN].notna().all(), "a cell has no distance to a listed tourist place"
    return gpd.GeoDataFrame(enriched, geometry="geometry", crs=grid.crs)


def compute_site_score(
    rural_cells: gpd.GeoDataFrame,
    listings: pd.DataFrame,
    places: pd.DataFrame,
    radius_km: float = NEIGHBORHOOD_RADIUS_KM,
) -> tuple[pd.DataFrame, dict[str, object]]:
    assert len(rural_cells) > 0, "there are no rural cells to score"
    assert rural_cells[CELL_ID_COLUMN].is_unique, "the rural site score repeats a cell"
    assert radius_km > 0, f"neighborhood radius must be positive, got {radius_km}"
    _assert_coordinates(rural_cells, "rural cell")
    _assert_coordinates(listings, "active listing")
    _assert_coordinates(places, "listed tourist place")
    assert len(listings) > 0, "there are no active listings to measure the nearby market against"
    assert {REVENUE_COLUMN, OCCUPANCY_COLUMN} <= set(listings.columns), (
        "active listings must carry trailing-twelve-month revenue and occupancy"
    )

    indicators = _indicator_frame(rural_cells, listings, places, radius_km)
    normalized = minimum_maximum_normalize(indicators, directions=SITE_SCORE_DIRECTIONS)
    weights = pd.Series(
        {normalized_column(name): weight for name, weight in SITE_SCORE_WEIGHTS.items()}
    )
    scores = composite_suitability_score(normalized, weights)
    labels, classification_report = classify_traffic_light(scores)

    access = normalized[normalized_column(DISTANCE_COLUMN)] * TOURIST_ACCESS_BLOC
    market = (
        normalized[normalized_column(NEARBY_REVENUE_COLUMN)] * SITE_SCORE_WEIGHTS[NEARBY_REVENUE_COLUMN]
        + normalized[normalized_column(NEARBY_OCCUPANCY_COLUMN)]
        * SITE_SCORE_WEIGHTS[NEARBY_OCCUPANCY_COLUMN]
    )
    competition = normalized[normalized_column(COMPETITION_COLUMN)] * LOCAL_COMPETITION_BLOC

    result = pd.DataFrame(
        {
            CELL_ID_COLUMN: rural_cells[CELL_ID_COLUMN].to_numpy(),
            RURAL_SCORE_COLUMN: scores.to_numpy(),
            RURAL_CLASS_COLUMN: labels.to_numpy(),
            DISTANCE_COLUMN: indicators[DISTANCE_COLUMN].to_numpy(),
            NEAREST_PLACE_COLUMN: indicators[NEAREST_PLACE_COLUMN].to_numpy(),
            NEARBY_REVENUE_COLUMN: indicators[NEARBY_REVENUE_COLUMN].to_numpy(),
            NEARBY_OCCUPANCY_COLUMN: indicators[NEARBY_OCCUPANCY_COLUMN].to_numpy(),
            LISTINGS_IN_CELL_COLUMN: indicators[LISTINGS_IN_CELL_COLUMN].to_numpy(),
            LISTINGS_NEARBY_COLUMN: indicators[LISTINGS_NEARBY_COLUMN].to_numpy(),
            COMPETITION_COLUMN: indicators[COMPETITION_COLUMN].to_numpy(),
            TOURIST_ACCESS_CONTRIBUTION: access.to_numpy(),
            MARKET_CONTRIBUTION: market.to_numpy(),
            COMPETITION_CONTRIBUTION: competition.to_numpy(),
        }
    )
    assert result[CELL_ID_COLUMN].tolist() == rural_cells[CELL_ID_COLUMN].tolist(), (
        "the site score changed the cell order"
    )
    assert result[RURAL_SCORE_COLUMN].notna().all(), "a rural cell has no site score"
    assert set(SITE_SCORE_WEIGHTS).isdisjoint(POI_BLOC_INDICATORS), (
        "an OpenStreetMap point-of-interest density entered the site score"
    )
    assert abs(sum(SITE_SCORE_WEIGHTS.values()) - 1.0) <= 1e-9, "site weights must sum to 1"

    report = {
        "weighting_method": WEIGHTING_METHOD,
        "neighborhood_radius_km": float(radius_km),
        "listed_tourist_places": int(len(places)),
        "active_listings": int(len(listings)),
        "indicators": list(SITE_SCORE_DIRECTIONS),
        "directions": dict(SITE_SCORE_DIRECTIONS),
        "weights": dict(SITE_SCORE_WEIGHTS),
        "weight_sum": float(sum(SITE_SCORE_WEIGHTS.values())),
        "blocs": {
            "tourist_access": float(TOURIST_ACCESS_BLOC),
            "nearby_market": float(NEARBY_MARKET_BLOC),
            "local_competition": float(LOCAL_COMPETITION_BLOC),
        },
        "excluded_osm_poi_densities": list(POI_BLOC_INDICATORS),
        "rural_cells_scored": int(len(result)),
        "jenks_breaks": classification_report["breaks"],
        "jenks_class_counts": classification_report["class_counts"],
        "cells_with_no_nearby_listings": int((result[LISTINGS_NEARBY_COLUMN] == 0).sum()),
    }
    return result[list(SITE_COLUMNS)], report


def _indicator_frame(
    rural_cells: gpd.GeoDataFrame,
    listings: pd.DataFrame,
    places: pd.DataFrame,
    radius_km: float,
) -> pd.DataFrame:
    listing_cell = _cell_of_each_listing(rural_cells, listings)
    distances = haversine_km(
        rural_cells[LONGITUDE_COLUMN].to_numpy()[:, None],
        rural_cells[LATITUDE_COLUMN].to_numpy()[:, None],
        listings[LONGITUDE_COLUMN].to_numpy()[None, :],
        listings[LATITUDE_COLUMN].to_numpy()[None, :],
    )
    within = distances <= radius_km
    revenue = listings[REVENUE_COLUMN].to_numpy(dtype=float)
    occupancy = listings[OCCUPANCY_COLUMN].to_numpy(dtype=float)
    nearby_counts = within.sum(axis=1).astype(int)
    revenue_means = _masked_mean(within, revenue, nearby_counts)
    occupancy_means = _masked_mean(within, occupancy, nearby_counts)

    cell_ids = rural_cells[CELL_ID_COLUMN].astype(str).to_numpy()
    in_cell = np.zeros(len(rural_cells), dtype=int)
    competition = np.zeros(len(rural_cells), dtype=int)
    for index, cell_id in enumerate(cell_ids):
        inside = listing_cell == cell_id
        in_cell[index] = int(inside.sum())
        ring = int((within[index] & ~inside).sum())
        competition[index] = in_cell[index] + ring

    place_distances = haversine_km(
        rural_cells[LONGITUDE_COLUMN].to_numpy()[:, None],
        rural_cells[LATITUDE_COLUMN].to_numpy()[:, None],
        places[LONGITUDE_COLUMN].to_numpy()[None, :],
        places[LATITUDE_COLUMN].to_numpy()[None, :],
    )
    nearest = place_distances.argmin(axis=1)
    rows = np.arange(len(rural_cells))

    frame = rural_cells[[CELL_ID_COLUMN]].copy()
    frame[DISTANCE_COLUMN] = place_distances[rows, nearest]
    frame[NEAREST_PLACE_COLUMN] = places["name"].to_numpy()[nearest]
    frame[NEARBY_REVENUE_COLUMN] = revenue_means
    frame[NEARBY_OCCUPANCY_COLUMN] = occupancy_means
    frame[LISTINGS_IN_CELL_COLUMN] = in_cell
    frame[LISTINGS_NEARBY_COLUMN] = nearby_counts
    frame[COMPETITION_COLUMN] = competition
    assert (frame[COMPETITION_COLUMN] >= frame[LISTINGS_IN_CELL_COLUMN]).all(), (
        "competition dropped a listing that sits inside the cell"
    )
    assert frame[DISTANCE_COLUMN].notna().all(), "a rural cell has no distance to a listed place"
    return frame


def _cell_of_each_listing(rural_cells: gpd.GeoDataFrame, listings: pd.DataFrame) -> np.ndarray:
    points = gpd.GeoDataFrame(
        {CELL_ID_COLUMN: np.arange(len(listings))},
        geometry=gpd.points_from_xy(listings[LONGITUDE_COLUMN], listings[LATITUDE_COLUMN]),
        crs=GEOGRAPHIC_CRS,
    ).to_crs(rural_cells.crs)
    joined = gpd.sjoin(
        points,
        rural_cells[[CELL_ID_COLUMN, "geometry"]],
        predicate="within",
        how="left",
        lsuffix="listing",
        rsuffix="cell",
    )
    cell_column = f"{CELL_ID_COLUMN}_cell"
    assert joined[f"{CELL_ID_COLUMN}_listing"].is_unique, (
        "a listing fell inside more than one rural cell"
    )
    joined = joined.sort_values(f"{CELL_ID_COLUMN}_listing")
    return joined[cell_column].fillna("").astype(str).to_numpy()


def _masked_mean(mask: np.ndarray, values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    totals = mask @ values
    means = np.zeros(len(counts), dtype=float)
    covered = counts > 0
    means[covered] = totals[covered] / counts[covered]
    return means


def _assert_coordinates(frame: pd.DataFrame, label: str) -> None:
    assert {LONGITUDE_COLUMN, LATITUDE_COLUMN} <= set(frame.columns), (
        f"a {label} is missing longitude or latitude"
    )
    coordinates = frame[[LONGITUDE_COLUMN, LATITUDE_COLUMN]].apply(pd.to_numeric, errors="coerce")
    assert coordinates.notna().all().all(), f"a {label} has a missing coordinate"
