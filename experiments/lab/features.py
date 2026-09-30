"""Candidate feature engineering for the model-improvement experiment.

Every feature here is documented in experiments/feature_engineering/leakage_log.json,
built by run_experiments.py. None of these functions change the five features or the
label that the current thesis model on the 2ndver branch uses. They read the same
underlying data (listings, places, roads, the grid) as separate candidate columns.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from str_suitability.spatial.haversine import haversine_km

LON = "longitude"
LAT = "latitude"

EXISTING_GRID_COLUMNS = (
    "distance_to_laguna_de_bay",
    "distance_to_other_water",
    "distance_to_poblacion",
    "distance_to_nearest_transportation_facility",
    "distance_to_nearest_tourist_attraction",
    "poi_count_restaurants",
    "poi_count_commercial",
    "poi_count_transportation",
    "poi_count_recreation",
    "poi_count_tourist_attraction",
    "poi_count_other_facilities",
    "poi_count_total",
    "poi_density_restaurants",
    "poi_density_commercial",
    "poi_density_transportation",
    "poi_density_recreation",
    "poi_density_tourist_attraction",
    "poi_density_other_facilities",
    "poi_density_total",
    "population",
    "population_density_per_km2",
)


def _pairwise_km(lon_a, lat_a, lon_b, lat_b) -> np.ndarray:
    return haversine_km(
        np.asarray(lon_a, dtype=float)[:, None],
        np.asarray(lat_a, dtype=float)[:, None],
        np.asarray(lon_b, dtype=float)[None, :],
        np.asarray(lat_b, dtype=float)[None, :],
    )


def _radius_suffix(radius_km: float) -> str:
    if radius_km < 1:
        return f"{int(round(radius_km * 1000))}m"
    if radius_km == int(radius_km):
        return f"{int(radius_km)}km"
    return f"{radius_km:g}km"


def _slug(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", text.lower())
    return text.strip("_")


def _masked_mean(mask, values, counts):
    totals = mask @ values
    out = np.full(len(counts), np.nan)
    covered = counts > 0
    out[covered] = totals[covered] / counts[covered]
    return out


def _masked_sum(mask, values, counts):
    totals = mask @ values
    out = np.full(len(counts), np.nan)
    covered = counts > 0
    out[covered] = totals[covered]
    return out


def _masked_median(mask, values, counts):
    out = np.full(len(counts), np.nan)
    for i in np.flatnonzero(counts > 0):
        out[i] = float(np.median(values[mask[i]]))
    return out


def _masked_std(mask, values, counts):
    out = np.full(len(counts), np.nan)
    for i in np.flatnonzero(counts > 1):
        out[i] = float(np.std(values[mask[i]], ddof=1))
    return out


def _masked_iqr(mask, values, counts):
    out = np.full(len(counts), np.nan)
    for i in np.flatnonzero(counts > 0):
        v = values[mask[i]]
        out[i] = float(np.percentile(v, 75) - np.percentile(v, 25))
    return out


def _masked_tier_count(mask, tier, target_tier):
    return (mask & (tier[None, :] == target_tier)).sum(axis=1)


def listing_performance_tier(
    listings: pd.DataFrame, revenue_col: str = "ttm_revenue", occupancy_col: str = "ttm_occupancy"
) -> pd.Series:
    """Low/Moderate/High tier for each listing, from its own revenue and occupancy,
    with the same 25/75 percentile rule and equal-weight min-max score the baseline
    model uses for cells. Computed at listing level so a 'nearby high performer
    count' feature can describe OTHER listings without needing that listing's own
    cell to already have a label."""
    revenue = listings[revenue_col].astype(float)
    occupancy = listings[occupancy_col].astype(float)
    revenue_norm = (revenue - revenue.min()) / (revenue.max() - revenue.min())
    occupancy_norm = (occupancy - occupancy.min()) / (occupancy.max() - occupancy.min())
    score = 0.5 * revenue_norm + 0.5 * occupancy_norm
    low_cut = np.percentile(score, 25)
    high_cut = np.percentile(score, 75)
    tier = pd.Series(1, index=listings.index)
    tier[score <= low_cut] = 0
    tier[score >= high_cut] = 2
    return tier


def build_airbnb_multiradius(
    cells: pd.DataFrame,
    listings: pd.DataFrame,
    radii_km: tuple,
    listing_cell_col: str,
    cell_id_col: str = "cell_id",
) -> pd.DataFrame:
    """Count, density, revenue/occupancy mean-median-total-std-iqr, and performer
    counts/percentages, for each radius in radii_km. Listings inside the cell are
    excluded at every radius, the same rule the baseline surrounding features use.
    `listings[listing_cell_col]` must already hold the cell_id each listing falls in."""
    listing_lon = listings[LON].to_numpy(dtype=float)
    listing_lat = listings[LAT].to_numpy(dtype=float)
    revenue = listings["ttm_revenue"].to_numpy(dtype=float)
    occupancy = listings["ttm_occupancy"].to_numpy(dtype=float)
    tier = listing_performance_tier(listings).to_numpy(dtype=int)
    listing_cells = listings[listing_cell_col].astype(str).to_numpy()

    cell_lon = cells[LON].to_numpy(dtype=float)
    cell_lat = cells[LAT].to_numpy(dtype=float)
    cell_ids = cells[cell_id_col].astype(str).to_numpy()

    distances = _pairwise_km(cell_lon, cell_lat, listing_lon, listing_lat)
    inside = listing_cells[None, :] == cell_ids[:, None]

    out = pd.DataFrame({cell_id_col: cell_ids})
    for radius in radii_km:
        mask = (distances <= radius) & ~inside
        counts = mask.sum(axis=1)
        area_km2 = np.pi * radius ** 2
        suffix = _radius_suffix(radius)
        out[f"airbnb_count_{suffix}"] = counts
        out[f"airbnb_density_{suffix}"] = counts / area_km2
        out[f"avg_revenue_{suffix}"] = _masked_mean(mask, revenue, counts)
        out[f"median_revenue_{suffix}"] = _masked_median(mask, revenue, counts)
        out[f"total_revenue_{suffix}"] = _masked_sum(mask, revenue, counts)
        out[f"avg_occupancy_{suffix}"] = _masked_mean(mask, occupancy, counts)
        out[f"median_occupancy_{suffix}"] = _masked_median(mask, occupancy, counts)
        out[f"revenue_std_{suffix}"] = _masked_std(mask, revenue, counts)
        out[f"occupancy_std_{suffix}"] = _masked_std(mask, occupancy, counts)
        out[f"revenue_iqr_{suffix}"] = _masked_iqr(mask, revenue, counts)
        out[f"occupancy_iqr_{suffix}"] = _masked_iqr(mask, occupancy, counts)
        high = _masked_tier_count(mask, tier, 2)
        moderate = _masked_tier_count(mask, tier, 1)
        low = _masked_tier_count(mask, tier, 0)
        out[f"high_performer_count_{suffix}"] = high
        out[f"moderate_performer_count_{suffix}"] = moderate
        out[f"low_performer_count_{suffix}"] = low
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"high_performer_pct_{suffix}"] = np.where(counts > 0, high / np.maximum(counts, 1), np.nan)
            out[f"moderate_performer_pct_{suffix}"] = np.where(counts > 0, moderate / np.maximum(counts, 1), np.nan)
            out[f"low_performer_pct_{suffix}"] = np.where(counts > 0, low / np.maximum(counts, 1), np.nan)
    return out


def build_poi_multiradius(
    cells: pd.DataFrame, places: pd.DataFrame, radii_km: tuple, cell_id_col: str = "cell_id"
) -> pd.DataFrame:
    """POI count and density at each radius, distance to the nearest POI and the
    average distance to the 3 and 5 nearest, and per-category counts within 5 km
    using the categories that already exist in poi_laguna.json (Falls, Mountains,
    Lakes & Ponds, Rivers & Landscape). No category is invented."""
    cell_lon = cells[LON].to_numpy(dtype=float)
    cell_lat = cells[LAT].to_numpy(dtype=float)
    cell_ids = cells[cell_id_col].astype(str).to_numpy()
    place_lon = places[LON].to_numpy(dtype=float)
    place_lat = places[LAT].to_numpy(dtype=float)
    distances = _pairwise_km(cell_lon, cell_lat, place_lon, place_lat)
    sorted_d = np.sort(distances, axis=1)

    out = pd.DataFrame({cell_id_col: cell_ids})
    out["nearest_poi_distance_km"] = sorted_d[:, 0]
    out["avg_distance_3_nearest_poi_km"] = sorted_d[:, : min(3, sorted_d.shape[1])].mean(axis=1)
    out["avg_distance_5_nearest_poi_km"] = sorted_d[:, : min(5, sorted_d.shape[1])].mean(axis=1)
    for radius in radii_km:
        suffix = _radius_suffix(radius)
        counts = (distances <= radius).sum(axis=1)
        out[f"poi_count_{suffix}"] = counts
        out[f"poi_density_{suffix}"] = counts / (np.pi * radius ** 2)
    if "category" in places.columns:
        categories = places["category"].astype(str).to_numpy()
        for category in sorted(set(categories)):
            cat_mask = categories == category
            cat_distances = distances[:, cat_mask]
            out[f"{_slug(category)}_count_5km"] = (cat_distances <= 5.0).sum(axis=1)
    return out


def build_road_features(
    cells_projected,
    roads_projected,
    density_radii_km: tuple = (1.0, 3.0, 5.0),
    cell_id_col: str = "cell_id",
) -> pd.DataFrame:
    """Nearest mapped road (all roads, then major/local/other separately using the
    already-existing road_class field), and road count/density within each radius.
    cells_projected and roads_projected must already be in EPSG:32651 with points
    for cells_projected's geometry (the cell centroid)."""
    cell_ids = cells_projected[cell_id_col].astype(str).to_numpy()
    out = pd.DataFrame({cell_id_col: cell_ids})

    all_roads = roads_projected.loc[~roads_projected.geometry.is_empty & roads_projected.geometry.notna()]
    out["nearest_road_distance_km"] = _nearest_km(cells_projected, all_roads, cell_id_col)

    for road_class in ("major", "local", "other"):
        subset = all_roads.loc[all_roads["road_class"] == road_class]
        column = f"nearest_{road_class}_road_distance_km"
        out[column] = _nearest_km(cells_projected, subset, cell_id_col) if len(subset) else np.nan

    sindex = all_roads.sindex
    for radius in density_radii_km:
        buffers = cells_projected.geometry.buffer(radius * 1000.0)
        counts = np.array([len(sindex.query(buf, predicate="intersects")) for buf in buffers])
        suffix = _radius_suffix(radius)
        out[f"road_count_{suffix}"] = counts
        out[f"road_density_{suffix}"] = counts / (np.pi * radius ** 2)
    return out


def _nearest_km(points_gdf, lines_gdf, cell_id_col: str) -> np.ndarray:
    joined = points_gdf[[cell_id_col, "geometry"]].sjoin_nearest(
        lines_gdf[["geometry"]], how="left", distance_col="_dist_m"
    )
    joined = joined.sort_values("_dist_m").drop_duplicates(cell_id_col, keep="first")
    ordered = points_gdf[[cell_id_col]].merge(joined[[cell_id_col, "_dist_m"]], on=cell_id_col, how="left")
    return (ordered["_dist_m"] / 1000.0).to_numpy()


def reuse_existing_grid_columns(grid: pd.DataFrame, cell_id_col: str = "cell_id") -> pd.DataFrame:
    """Columns already computed by the older src/str_suitability/pipeline.py build:
    OSM POI densities by category, distance to Laguna de Bay/other water/the
    poblacion, and PSA population. These already sit on
    data/processed/grid_features.parquet. This function only selects and joins
    them; it does not change how any of them were calculated."""
    available = [c for c in EXISTING_GRID_COLUMNS if c in grid.columns]
    return grid[[cell_id_col, *available]].copy()


def build_urban_proximity(
    grid: pd.DataFrame, cell_classification: pd.DataFrame, radius_km: float = 5.0, cell_id_col: str = "cell_id"
) -> pd.DataFrame:
    """Distance to the nearest urban grid cell, and the share of grid cells within
    radius_km that are urban. Reuses the existing rural/urban classification
    already computed for the map (rural/classify.py); this is a coarse,
    cell-count-based proxy for urban area share, not a repeat of the polygon-area
    rural_area_share calculation."""
    merged = grid[[cell_id_col, LON, LAT]].merge(
        cell_classification[[cell_id_col, "cell_class"]], on=cell_id_col, how="left"
    )
    lon = merged[LON].to_numpy(dtype=float)
    lat = merged[LAT].to_numpy(dtype=float)
    is_urban = (merged["cell_class"] == "urban").to_numpy()
    distances = _pairwise_km(lon, lat, lon, lat)
    np.fill_diagonal(distances, np.inf)
    nearest_urban_km = distances[:, is_urban].min(axis=1) if is_urban.any() else np.full(len(merged), np.nan)
    within = distances <= radius_km
    counts_within = within.sum(axis=1)
    urban_within = (within & is_urban[None, :]).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        urban_share = np.where(counts_within > 0, urban_within / np.maximum(counts_within, 1), np.nan)
    out = pd.DataFrame({cell_id_col: merged[cell_id_col].to_numpy()})
    out["distance_to_nearest_urban_cell_km"] = nearest_urban_km
    out[f"urban_area_share_{_radius_suffix(radius_km)}"] = urban_share
    return out


def build_spatial_context(
    labeled: pd.DataFrame,
    grid: pd.DataFrame,
    radius_km: float = 5.0,
    cell_id_col: str = "cell_id",
    class_col: str = "performance_class",
) -> pd.DataFrame:
    """Nearby labeled-cell performance counts within radius_km, always excluding the
    cell itself. This uses only OTHER cells' historical labels, never the target
    cell's own label, so it is safe by the same reasoning as the existing
    surrounding-market features."""
    merged = labeled[[cell_id_col, class_col]].merge(grid[[cell_id_col, LON, LAT]], on=cell_id_col, how="left")
    lon = merged[LON].to_numpy(dtype=float)
    lat = merged[LAT].to_numpy(dtype=float)
    classes = merged[class_col].astype(int).to_numpy()
    distances = _pairwise_km(lon, lat, lon, lat)
    np.fill_diagonal(distances, np.inf)
    within = distances <= radius_km
    out = pd.DataFrame({cell_id_col: merged[cell_id_col].astype(str).to_numpy()})
    suffix = _radius_suffix(radius_km)
    for tier, name in enumerate(("low", "moderate", "high")):
        out[f"nearby_{name}_square_count_{suffix}"] = (within & (classes[None, :] == tier)).sum(axis=1)
    out[f"nearby_labeled_square_count_{suffix}"] = within.sum(axis=1)
    return out
