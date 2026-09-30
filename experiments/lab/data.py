"""Loads the same source data the current thesis pipeline reads, once, read-only.

Nothing in this file writes to data/processed or data/interim. It only reads
existing files and existing pure functions from str_suitability.
"""

from __future__ import annotations

import json

import geopandas as gpd
import pandas as pd

from str_suitability import config
from str_suitability.modeling.location_classifier import MUNICIPALITY_COLUMN
from str_suitability.rural.site_score import LATITUDE_COLUMN, LONGITUDE_COLUMN
from str_suitability.spatial.join import assign_listings_to_cells

GRID_PATH = config.PROCESSED_DIR / "grid_features.parquet"
CELL_CLASSIFICATION_PATH = config.RURAL_PROCESSED_DIR / "grid_rural_classification.parquet"
TARGETS_PATH = config.INTERIM_DIR / "airroi_targets.parquet"
MODEL_SUMMARY_PATH = config.RURAL_PROCESSED_DIR / "model_summary.json"
GRID_ROAD_DISTANCE_PATH = config.PROCESSED_DIR / "grid_road_distance.parquet"


def load_grid() -> gpd.GeoDataFrame:
    return gpd.read_parquet(GRID_PATH)


def load_cell_classification() -> pd.DataFrame:
    return pd.read_parquet(CELL_CLASSIFICATION_PATH)


def load_active_listings() -> pd.DataFrame:
    targets = pd.read_parquet(TARGETS_PATH)
    return targets.loc[targets["in_training_population"]].copy().reset_index(drop=True)


def load_places_with_category() -> pd.DataFrame:
    """The same poi_laguna.json the production model reads, but keeping the real
    `category` field that str_suitability.rural.site_score.load_listed_tourist_places
    strips out before returning. This is a separate loader; it does not change what
    the production TOURIST_FEATURES use."""
    payload = json.loads(config.LISTED_TOURIST_PLACES_PATH.read_text(encoding="utf-8"))
    records = payload["places"] if isinstance(payload, dict) and "places" in payload else payload
    places = pd.DataFrame(records)
    places[LONGITUDE_COLUMN] = pd.to_numeric(places[LONGITUDE_COLUMN], errors="coerce")
    places[LATITUDE_COLUMN] = pd.to_numeric(places[LATITUDE_COLUMN], errors="coerce")
    places = places.dropna(subset=[LONGITUDE_COLUMN, LATITUDE_COLUMN]).reset_index(drop=True)
    columns = [LONGITUDE_COLUMN, LATITUDE_COLUMN, "name"]
    if "category" in places.columns:
        columns.append("category")
    return places[columns].copy()


def load_roads() -> gpd.GeoDataFrame:
    from str_suitability.features.road_distance import LAGUNA_ROADS_PATH

    return gpd.read_file(LAGUNA_ROADS_PATH)


def load_grid_road_distance() -> pd.DataFrame:
    return pd.read_parquet(GRID_ROAD_DISTANCE_PATH)


def load_model_summary() -> dict:
    return json.loads(MODEL_SUMMARY_PATH.read_text(encoding="utf-8"))


def attach_listing_cell_id(active_listings: pd.DataFrame, grid: gpd.GeoDataFrame) -> pd.DataFrame:
    """Each active listing's cell_id, via the same spatial join
    (str_suitability.spatial.join.assign_listings_to_cells) the older pipeline
    uses. Needed so multi-radius Airbnb features can exclude a cell's own
    listings, the same rule the production surrounding-market features use."""
    joined, _report = assign_listings_to_cells(active_listings, grid)
    return joined


def municipality_of(grid: gpd.GeoDataFrame) -> pd.Series:
    if MUNICIPALITY_COLUMN in grid.columns:
        return grid[MUNICIPALITY_COLUMN].fillna("unassigned").astype(str)
    raise AssertionError("grid has no municipality column; rebuild it with join_demographics")
