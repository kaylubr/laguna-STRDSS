"""Grid-size sensitivity (experiment group M).

Rebuilds the grid at an alternate cell size and reruns the SAME five-feature
market model and municipality-grouped validation. Reuses build_grid,
market_features, and label_performance as pure functions; does not modify them
or the production 1 km grid_features.parquet.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from str_suitability import config
from str_suitability.ingest.load_osm import load_boundaries
from str_suitability.modeling.location_classifier import label_performance, market_features
from str_suitability.pipeline import load_water
from str_suitability.preprocess.clean_psa import attach_psgc_codes, load_psa_population
from str_suitability.spatial.grid import build_grid, derive_land_boundary
from str_suitability.spatial.join import build_municipality_table, join_demographics

FEATURE_DIR = config.PROJECT_ROOT / "assets" / "osm"


def build_alt_grid(cell_size_m: int) -> tuple[gpd.GeoDataFrame, dict]:
    """A grid at cell_size_m, with the same municipality assignment (dominant-overlap
    PSGC join) the production 1 km grid uses. Everything else about the study area
    (province boundary, water cut-out) is unchanged."""
    province, municipalities = load_boundaries(config.BOUNDARY_DIR)
    water = load_water(FEATURE_DIR)
    land_polygon = derive_land_boundary(province, water)
    grid = build_grid(land_polygon, cell_size_m)

    population = load_psa_population(config.PSA_PATH)
    population, psgc_report = attach_psgc_codes(population, municipalities)
    municipality_table = build_municipality_table(municipalities, population)
    grid, demo_report = join_demographics(grid, municipality_table)
    report = {
        "cell_size_m": int(cell_size_m),
        "cells": int(len(grid)),
        "psgc_attach": psgc_report,
        "demographics": demo_report,
    }
    return grid, report


def labeled_frame_for_grid(
    grid: gpd.GeoDataFrame, active_listings: pd.DataFrame, places: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    features = market_features(grid, active_listings, places)
    labeled, label_report = label_performance(features)
    return labeled, label_report
