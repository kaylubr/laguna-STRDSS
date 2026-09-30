"""Builds every candidate feature (experiment groups A, B, C, E, F, G, I, J, K)
for the labeled cells and saves them to experiments/feature_engineering/, along
with a leakage log documenting Feature/Source/Calculation/leakage-safety for
every single candidate column. Nothing here is used by the production pipeline.

Run with: uv run python -m experiments.build_feature_table
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from str_suitability import config
from str_suitability.modeling.location_classifier import (
    CELL_ID_COLUMN,
    CLASSIFIER_FEATURES,
    MUNICIPALITY_COLUMN,
    PERFORMANCE_CLASS,
    PERFORMANCE_SCORE,
    label_performance,
    market_features,
)

from experiments.lab import data as lab_data
from experiments.lab import features as lab_features

OUT_DIR = Path(__file__).parent / "feature_engineering"
AIRBNB_RADII_KM = (0.5, 1.0, 3.0, 5.0, 10.0)
POI_RADII_KM = (0.5, 1.0, 3.0, 5.0, 10.0)
ROAD_DENSITY_RADII_KM = (1.0, 3.0, 5.0)
URBAN_RADIUS_KM = 5.0
SPATIAL_CONTEXT_RADII_KM = (3.0, 5.0)


def main() -> None:
    grid = lab_data.load_grid()
    cell_classification = lab_data.load_cell_classification()
    active_listings = lab_data.load_active_listings()
    listings_with_cell = lab_data.attach_listing_cell_id(active_listings, grid)
    places = lab_data.load_places_with_category()

    base_features = market_features(grid, active_listings, places)
    labeled, label_report = label_performance(base_features)
    labeled_cell_ids = labeled[[CELL_ID_COLUMN]].copy()
    labeled_grid = grid.merge(labeled_cell_ids, on=CELL_ID_COLUMN, how="inner")
    labeled_coords = labeled.merge(
        pd.DataFrame(grid[[CELL_ID_COLUMN, "longitude", "latitude"]]), on=CELL_ID_COLUMN, how="left"
    )

    print(f"labeled cells: {len(labeled)}, municipalities: {labeled[MUNICIPALITY_COLUMN].nunique()}")

    # --- Group A/B: Airbnb multi-radius market, density, stability, performer mix
    airbnb = lab_features.build_airbnb_multiradius(
        labeled_coords, listings_with_cell, AIRBNB_RADII_KM, listing_cell_col=CELL_ID_COLUMN
    )

    # --- Group C: POI multi-radius + real categories
    poi = lab_features.build_poi_multiradius(labeled_coords, places, POI_RADII_KM)

    # --- Group E: road accessibility (all/major/local/other, density at radii)
    roads_geo = lab_data.load_roads()
    roads_projected = roads_geo.to_crs(config.PROJECTED_CRS)
    labeled_points = gpd.GeoDataFrame(
        labeled[[CELL_ID_COLUMN]].copy(),
        geometry=gpd.points_from_xy(labeled_grid["centroid_x"], labeled_grid["centroid_y"]),
        crs=config.PROJECTED_CRS,
    )
    road = lab_features.build_road_features(labeled_points, roads_projected, ROAD_DENSITY_RADII_KM)
    grid_road_distance = lab_data.load_grid_road_distance()
    road = road.merge(
        grid_road_distance[[CELL_ID_COLUMN, "road_distance_km"]].rename(
            columns={"road_distance_km": "nearest_road_distance_km_precomputed"}
        ),
        on=CELL_ID_COLUMN,
        how="left",
    )

    # --- Groups F/G/I: reuse already-computed grid columns (POI density/services proxy,
    # water distance, poblacion distance, population)
    existing = lab_features.reuse_existing_grid_columns(labeled_grid)

    # --- Group J: urban proximity, from the existing rural/urban cell classification
    urban = lab_features.build_urban_proximity(grid, cell_classification, radius_km=URBAN_RADIUS_KM)
    urban = urban.loc[urban[CELL_ID_COLUMN].isin(labeled[CELL_ID_COLUMN])].reset_index(drop=True)

    # --- Group K: spatial context, nearby labeled squares' own historical class (self excluded)
    spatial_context_frames = [
        lab_features.build_spatial_context(labeled, grid, radius_km=radius) for radius in SPATIAL_CONTEXT_RADII_KM
    ]

    engineered = labeled[[CELL_ID_COLUMN, MUNICIPALITY_COLUMN, PERFORMANCE_CLASS, PERFORMANCE_SCORE, *CLASSIFIER_FEATURES]].copy()
    for frame in (airbnb, poi, road, existing, urban, *spatial_context_frames):
        engineered = engineered.merge(frame, on=CELL_ID_COLUMN, how="left")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    engineered.to_parquet(OUT_DIR / "engineered_features.parquet", index=False)
    engineered.drop(columns=["geometry"], errors="ignore").to_csv(OUT_DIR / "engineered_features.csv", index=False)

    candidate_cols = [c for c in engineered.columns if c not in {CELL_ID_COLUMN, MUNICIPALITY_COLUMN, PERFORMANCE_CLASS, PERFORMANCE_SCORE, *CLASSIFIER_FEATURES}]
    write_leakage_log(candidate_cols)

    print(f"engineered feature table: {engineered.shape[0]} rows x {engineered.shape[1]} columns")
    print(f"candidate (non-baseline) columns: {len(candidate_cols)}")
    print("missing-value rate (top 10, highest first):")
    na_rate = engineered[candidate_cols].isna().mean().sort_values(ascending=False)
    print(na_rate.head(10).to_string())


def _poi_radius_columns() -> list[str]:
    suffixes = [lab_features._radius_suffix(r) for r in POI_RADII_KM]
    return [f"poi_count_{s}" for s in suffixes] + [f"poi_density_{s}" for s in suffixes]


def write_leakage_log(candidate_cols: list[str]) -> None:
    rows = []

    def _matches(column: str, pattern: str) -> bool:
        return column == pattern or (pattern.endswith("_") and column.startswith(pattern))

    def entry(pattern_columns, source, calculation, uses_target_info, uses_own_airbnb, uses_future, safe, note=""):
        for column in [c for c in candidate_cols if any(_matches(c, p) for p in pattern_columns)]:
            rows.append({
                "feature": column,
                "source": source,
                "calculation": calculation,
                "uses_target_class_label": uses_target_info,
                "uses_target_squares_own_airbnb": uses_own_airbnb,
                "uses_future_information": uses_future,
                "safe_for_prediction": safe,
                "note": note,
            })

    entry(
        ["airbnb_count_", "airbnb_density_"], "laguna_listings.json (active listings)",
        "Count/density of OTHER listings within radius, excluding listings inside the target cell.",
        False, False, False, True,
        "Uses the TTM revenue/occupancy window, same as the production surrounding features.",
    )
    entry(
        ["avg_revenue_", "median_revenue_", "total_revenue_", "revenue_std_", "revenue_iqr_",
         "avg_occupancy_", "median_occupancy_", "occupancy_std_", "occupancy_iqr_"],
        "laguna_listings.json (active listings)",
        "Mean/median/total/std/IQR of ttm_revenue or ttm_occupancy among OTHER listings within radius.",
        False, False, False, True,
        "Same exclusion rule as SURROUNDING_REVENUE/SURROUNDING_OCCUPANCY in the production model.",
    )
    entry(
        ["high_performer_count_", "moderate_performer_count_", "low_performer_count_",
         "high_performer_pct_", "moderate_performer_pct_", "low_performer_pct_"],
        "laguna_listings.json (active listings)",
        "Count/percentage of OTHER listings within radius whose OWN 25/75-percentile-of-score tier "
        "(computed at listing level, not cell level) is Low/Moderate/High.",
        False, False, False, True,
        "Tiers are computed from every listing's own revenue/occupancy, not from any cell label.",
    )
    entry(
        ["nearest_poi_distance_km", "avg_distance_3_nearest_poi_km", "avg_distance_5_nearest_poi_km",
         *_poi_radius_columns()],
        "poi_laguna.json (curated landscape places)",
        "Distance/count/density of landscape places (falls, mountains, lakes and ponds, rivers and "
        "landscape) within radius. poi_count_/poi_density_ here duplicate the production tourist-place "
        "count logic at additional radii.",
        False, False, False, True, "",
    )
    for category_col in [
        c for c in candidate_cols
        if c.endswith("_count_5km")
        and not c.startswith(("high_", "moderate_", "low_", "poi_", "nearby_", "airbnb_", "road_"))
    ]:
        rows.append({
            "feature": category_col, "source": "poi_laguna.json category field",
            "calculation": "Count of places in this specific real category within 5 km.",
            "uses_target_class_label": False, "uses_target_squares_own_airbnb": False,
            "uses_future_information": False, "safe_for_prediction": True,
            "note": "Category values are exactly what is in the data (Falls, Mountains, Lakes & Ponds, "
                    "Rivers & Landscape). None invented.",
        })
    entry(
        ["nearest_road_distance_km", "nearest_major_road_distance_km", "nearest_local_road_distance_km",
         "nearest_other_road_distance_km", "road_count_", "road_density_"],
        "laguna_roads.geojson (OSM/Geofabrik)",
        "Straight-line distance from the cell centroid to the nearest mapped road (all roads, or "
        "filtered to road_class local/major/other), and road segment count/density within radius.",
        False, False, False, True,
        "road_class is a real tagged field on the road data; not invented.",
    )
    rows.append({
        "feature": "nearest_road_distance_km_precomputed", "source": "data/processed/grid_road_distance.parquet",
        "calculation": "The already-computed production road-distance column, joined by cell_id, for a "
                        "cross-check against this file's own nearest_road_distance_km.",
        "uses_target_class_label": False, "uses_target_squares_own_airbnb": False,
        "uses_future_information": False, "safe_for_prediction": True, "note": "Sanity check column, not a new feature.",
    })
    entry(
        ["distance_to_laguna_de_bay", "distance_to_other_water"], "assets/osm water layer (via grid_features.parquet)",
        "Distance from the cell centroid to the Laguna de Bay polygon / to other mapped water bodies. "
        "Already computed by the older non-rural pipeline; reused, not recomputed.",
        False, False, False, True, "",
    )
    entry(
        ["distance_to_poblacion"], "assets/boundaries barangay layer (via grid_features.parquet)",
        "Distance from the cell centroid to the nearest poblacion (town/city center) barangay. "
        "Already computed by the older non-rural pipeline; reused, not recomputed.",
        False, False, False, True, "",
    )
    entry(
        ["distance_to_nearest_transportation_facility", "distance_to_nearest_tourist_attraction"],
        "OSM POI layer (via grid_features.parquet)",
        "Distance to the nearest OSM-tagged transportation facility / tourist attraction. Already "
        "computed by the older non-rural pipeline; reused, not recomputed.",
        False, False, False, True, "",
    )
    entry(
        ["poi_count_restaurants", "poi_count_commercial", "poi_count_transportation", "poi_count_recreation",
         "poi_count_tourist_attraction", "poi_count_other_facilities", "poi_count_total",
         "poi_density_restaurants", "poi_density_commercial", "poi_density_transportation",
         "poi_density_recreation", "poi_density_tourist_attraction", "poi_density_other_facilities",
         "poi_density_total"],
        "OSM POI layer (via grid_features.parquet)",
        "OSM point-of-interest counts/densities by category (a services/amenity proxy). Already "
        "computed by the older non-rural pipeline; reused, not recomputed.",
        False, False, False, True,
        "This is the closest available stand-in for Group F 'access to services' features; the "
        "dataset does not separately tag hospitals, markets, or supermarkets.",
    )
    entry(
        ["population", "population_density_per_km2"], "PSA.json (municipality-level population)",
        "Municipality population / population density, spatially joined onto the cell by dominant "
        "municipality overlap. Already computed by the older non-rural pipeline; reused, not recomputed.",
        False, False, False, True,
        "ADR-0013 caution: this value is constant within a municipality and correlates strongly with "
        "municipality identity, which is why the production model excludes municipality identity "
        "itself as a predictor. Treated here as a documented, not hidden, risk.",
    )
    entry(
        ["distance_to_nearest_urban_cell_km", "urban_area_share_"], "grid_rural_classification.parquet",
        "Distance to the nearest urban grid cell centroid, and the share of grid cells within radius "
        "classified urban. Reuses the existing rural/urban classification; a coarse, cell-count proxy, "
        "not a repeat of the polygon-area rural_area_share already on the grid.",
        False, False, False, True, "",
    )
    entry(
        ["nearby_low_square_count_", "nearby_moderate_square_count_", "nearby_high_square_count_",
         "nearby_labeled_square_count_"],
        "This experiment's own labeled cell table (self excluded)",
        "Count of OTHER labeled cells within radius whose historical performance_class is Low/"
        "Moderate/High, or labeled at all. The target cell's own label is excluded by construction "
        "(the distance matrix diagonal is set to infinity before thresholding).",
        True, False, False, True,
        "Uses OTHER cells' target labels, the same way the production surrounding-market features use "
        "OTHER listings' revenue/occupancy. Never uses the target cell's own label. Flagged as "
        "uses_target_class_label=true because it is built from the target VARIABLE (just not this row's "
        "own value), so it deserves extra scrutiny even though it is not literally the same leak as "
        "using a cell's own historical performance to predict its own class.",
    )

    documented = {row["feature"] for row in rows}
    undocumented = [c for c in candidate_cols if c not in documented]
    if undocumented:
        for column in undocumented:
            rows.append({
                "feature": column, "source": "undocumented - see build_feature_table.py",
                "calculation": "undocumented", "uses_target_class_label": None,
                "uses_target_squares_own_airbnb": None, "uses_future_information": None,
                "safe_for_prediction": None, "note": "This column was produced but not matched by any "
                "leakage-log rule; treat as unverified until documented.",
            })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "leakage_log.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        "# Leakage log for engineered candidate features",
        "",
        "Every column produced by build_feature_table.py, with its source, how it is",
        "calculated, and whether it uses target-class information, the target square's",
        "own Airbnb history, or future information. `safe_for_prediction` follows the",
        "same rule the production model uses for its own surrounding features: a",
        "feature may describe OTHER squares' Airbnb history or OTHER squares' labels,",
        "but never the target square's own history or label.",
        "",
        "| Feature | Source | Uses target label | Uses own Airbnb | Uses future info | Safe |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['feature']}` | {row['source']} | {row['uses_target_class_label']} | "
            f"{row['uses_target_squares_own_airbnb']} | {row['uses_future_information']} | "
            f"{row['safe_for_prediction']} |"
        )
    lines.append("")
    lines.append("## Calculation detail")
    lines.append("")
    seen_calc = set()
    for row in rows:
        key = (row["source"], row["calculation"])
        if key in seen_calc:
            continue
        seen_calc.add(key)
        lines.append(f"- **{row['source']}** — {row['calculation']} {row.get('note', '')}".rstrip())
    (OUT_DIR / "leakage_log.md").write_text("\n".join(lines), encoding="utf-8")
    if undocumented:
        print("WARNING: undocumented columns:", undocumented)


if __name__ == "__main__":
    main()
