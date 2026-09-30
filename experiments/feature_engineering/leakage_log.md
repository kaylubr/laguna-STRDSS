# Leakage log for engineered candidate features

Every column produced by build_feature_table.py, with its source, how it is
calculated, and whether it uses target-class information, the target square's
own Airbnb history, or future information. `safe_for_prediction` follows the
same rule the production model uses for its own surrounding features: a
feature may describe OTHER squares' Airbnb history or OTHER squares' labels,
but never the target square's own history or label.

| Feature | Source | Uses target label | Uses own Airbnb | Uses future info | Safe |
|---|---|---|---|---|---|
| `airbnb_count_500m` | laguna_listings.json (active listings) | False | False | False | True |
| `airbnb_density_500m` | laguna_listings.json (active listings) | False | False | False | True |
| `airbnb_count_1km` | laguna_listings.json (active listings) | False | False | False | True |
| `airbnb_density_1km` | laguna_listings.json (active listings) | False | False | False | True |
| `airbnb_count_3km` | laguna_listings.json (active listings) | False | False | False | True |
| `airbnb_density_3km` | laguna_listings.json (active listings) | False | False | False | True |
| `airbnb_count_5km` | laguna_listings.json (active listings) | False | False | False | True |
| `airbnb_density_5km` | laguna_listings.json (active listings) | False | False | False | True |
| `airbnb_count_10km` | laguna_listings.json (active listings) | False | False | False | True |
| `airbnb_density_10km` | laguna_listings.json (active listings) | False | False | False | True |
| `avg_revenue_500m` | laguna_listings.json (active listings) | False | False | False | True |
| `median_revenue_500m` | laguna_listings.json (active listings) | False | False | False | True |
| `total_revenue_500m` | laguna_listings.json (active listings) | False | False | False | True |
| `avg_occupancy_500m` | laguna_listings.json (active listings) | False | False | False | True |
| `median_occupancy_500m` | laguna_listings.json (active listings) | False | False | False | True |
| `revenue_std_500m` | laguna_listings.json (active listings) | False | False | False | True |
| `occupancy_std_500m` | laguna_listings.json (active listings) | False | False | False | True |
| `revenue_iqr_500m` | laguna_listings.json (active listings) | False | False | False | True |
| `occupancy_iqr_500m` | laguna_listings.json (active listings) | False | False | False | True |
| `avg_revenue_1km` | laguna_listings.json (active listings) | False | False | False | True |
| `median_revenue_1km` | laguna_listings.json (active listings) | False | False | False | True |
| `total_revenue_1km` | laguna_listings.json (active listings) | False | False | False | True |
| `avg_occupancy_1km` | laguna_listings.json (active listings) | False | False | False | True |
| `median_occupancy_1km` | laguna_listings.json (active listings) | False | False | False | True |
| `revenue_std_1km` | laguna_listings.json (active listings) | False | False | False | True |
| `occupancy_std_1km` | laguna_listings.json (active listings) | False | False | False | True |
| `revenue_iqr_1km` | laguna_listings.json (active listings) | False | False | False | True |
| `occupancy_iqr_1km` | laguna_listings.json (active listings) | False | False | False | True |
| `avg_revenue_3km` | laguna_listings.json (active listings) | False | False | False | True |
| `median_revenue_3km` | laguna_listings.json (active listings) | False | False | False | True |
| `total_revenue_3km` | laguna_listings.json (active listings) | False | False | False | True |
| `avg_occupancy_3km` | laguna_listings.json (active listings) | False | False | False | True |
| `median_occupancy_3km` | laguna_listings.json (active listings) | False | False | False | True |
| `revenue_std_3km` | laguna_listings.json (active listings) | False | False | False | True |
| `occupancy_std_3km` | laguna_listings.json (active listings) | False | False | False | True |
| `revenue_iqr_3km` | laguna_listings.json (active listings) | False | False | False | True |
| `occupancy_iqr_3km` | laguna_listings.json (active listings) | False | False | False | True |
| `avg_revenue_5km` | laguna_listings.json (active listings) | False | False | False | True |
| `median_revenue_5km` | laguna_listings.json (active listings) | False | False | False | True |
| `total_revenue_5km` | laguna_listings.json (active listings) | False | False | False | True |
| `avg_occupancy_5km` | laguna_listings.json (active listings) | False | False | False | True |
| `median_occupancy_5km` | laguna_listings.json (active listings) | False | False | False | True |
| `revenue_std_5km` | laguna_listings.json (active listings) | False | False | False | True |
| `occupancy_std_5km` | laguna_listings.json (active listings) | False | False | False | True |
| `revenue_iqr_5km` | laguna_listings.json (active listings) | False | False | False | True |
| `occupancy_iqr_5km` | laguna_listings.json (active listings) | False | False | False | True |
| `avg_revenue_10km` | laguna_listings.json (active listings) | False | False | False | True |
| `median_revenue_10km` | laguna_listings.json (active listings) | False | False | False | True |
| `total_revenue_10km` | laguna_listings.json (active listings) | False | False | False | True |
| `avg_occupancy_10km` | laguna_listings.json (active listings) | False | False | False | True |
| `median_occupancy_10km` | laguna_listings.json (active listings) | False | False | False | True |
| `revenue_std_10km` | laguna_listings.json (active listings) | False | False | False | True |
| `occupancy_std_10km` | laguna_listings.json (active listings) | False | False | False | True |
| `revenue_iqr_10km` | laguna_listings.json (active listings) | False | False | False | True |
| `occupancy_iqr_10km` | laguna_listings.json (active listings) | False | False | False | True |
| `high_performer_count_500m` | laguna_listings.json (active listings) | False | False | False | True |
| `moderate_performer_count_500m` | laguna_listings.json (active listings) | False | False | False | True |
| `low_performer_count_500m` | laguna_listings.json (active listings) | False | False | False | True |
| `high_performer_pct_500m` | laguna_listings.json (active listings) | False | False | False | True |
| `moderate_performer_pct_500m` | laguna_listings.json (active listings) | False | False | False | True |
| `low_performer_pct_500m` | laguna_listings.json (active listings) | False | False | False | True |
| `high_performer_count_1km` | laguna_listings.json (active listings) | False | False | False | True |
| `moderate_performer_count_1km` | laguna_listings.json (active listings) | False | False | False | True |
| `low_performer_count_1km` | laguna_listings.json (active listings) | False | False | False | True |
| `high_performer_pct_1km` | laguna_listings.json (active listings) | False | False | False | True |
| `moderate_performer_pct_1km` | laguna_listings.json (active listings) | False | False | False | True |
| `low_performer_pct_1km` | laguna_listings.json (active listings) | False | False | False | True |
| `high_performer_count_3km` | laguna_listings.json (active listings) | False | False | False | True |
| `moderate_performer_count_3km` | laguna_listings.json (active listings) | False | False | False | True |
| `low_performer_count_3km` | laguna_listings.json (active listings) | False | False | False | True |
| `high_performer_pct_3km` | laguna_listings.json (active listings) | False | False | False | True |
| `moderate_performer_pct_3km` | laguna_listings.json (active listings) | False | False | False | True |
| `low_performer_pct_3km` | laguna_listings.json (active listings) | False | False | False | True |
| `high_performer_count_5km` | laguna_listings.json (active listings) | False | False | False | True |
| `moderate_performer_count_5km` | laguna_listings.json (active listings) | False | False | False | True |
| `low_performer_count_5km` | laguna_listings.json (active listings) | False | False | False | True |
| `high_performer_pct_5km` | laguna_listings.json (active listings) | False | False | False | True |
| `moderate_performer_pct_5km` | laguna_listings.json (active listings) | False | False | False | True |
| `low_performer_pct_5km` | laguna_listings.json (active listings) | False | False | False | True |
| `high_performer_count_10km` | laguna_listings.json (active listings) | False | False | False | True |
| `moderate_performer_count_10km` | laguna_listings.json (active listings) | False | False | False | True |
| `low_performer_count_10km` | laguna_listings.json (active listings) | False | False | False | True |
| `high_performer_pct_10km` | laguna_listings.json (active listings) | False | False | False | True |
| `moderate_performer_pct_10km` | laguna_listings.json (active listings) | False | False | False | True |
| `low_performer_pct_10km` | laguna_listings.json (active listings) | False | False | False | True |
| `nearest_poi_distance_km` | poi_laguna.json (curated landscape places) | False | False | False | True |
| `avg_distance_3_nearest_poi_km` | poi_laguna.json (curated landscape places) | False | False | False | True |
| `avg_distance_5_nearest_poi_km` | poi_laguna.json (curated landscape places) | False | False | False | True |
| `poi_count_500m` | poi_laguna.json (curated landscape places) | False | False | False | True |
| `poi_density_500m` | poi_laguna.json (curated landscape places) | False | False | False | True |
| `poi_count_1km` | poi_laguna.json (curated landscape places) | False | False | False | True |
| `poi_density_1km` | poi_laguna.json (curated landscape places) | False | False | False | True |
| `poi_count_3km` | poi_laguna.json (curated landscape places) | False | False | False | True |
| `poi_density_3km` | poi_laguna.json (curated landscape places) | False | False | False | True |
| `poi_count_5km` | poi_laguna.json (curated landscape places) | False | False | False | True |
| `poi_density_5km` | poi_laguna.json (curated landscape places) | False | False | False | True |
| `poi_count_10km` | poi_laguna.json (curated landscape places) | False | False | False | True |
| `poi_density_10km` | poi_laguna.json (curated landscape places) | False | False | False | True |
| `falls_count_5km` | poi_laguna.json category field | False | False | False | True |
| `lakes_ponds_count_5km` | poi_laguna.json category field | False | False | False | True |
| `mountains_count_5km` | poi_laguna.json category field | False | False | False | True |
| `rivers_landscape_count_5km` | poi_laguna.json category field | False | False | False | True |
| `nearest_road_distance_km` | laguna_roads.geojson (OSM/Geofabrik) | False | False | False | True |
| `nearest_major_road_distance_km` | laguna_roads.geojson (OSM/Geofabrik) | False | False | False | True |
| `nearest_local_road_distance_km` | laguna_roads.geojson (OSM/Geofabrik) | False | False | False | True |
| `nearest_other_road_distance_km` | laguna_roads.geojson (OSM/Geofabrik) | False | False | False | True |
| `road_count_1km` | laguna_roads.geojson (OSM/Geofabrik) | False | False | False | True |
| `road_density_1km` | laguna_roads.geojson (OSM/Geofabrik) | False | False | False | True |
| `road_count_3km` | laguna_roads.geojson (OSM/Geofabrik) | False | False | False | True |
| `road_density_3km` | laguna_roads.geojson (OSM/Geofabrik) | False | False | False | True |
| `road_count_5km` | laguna_roads.geojson (OSM/Geofabrik) | False | False | False | True |
| `road_density_5km` | laguna_roads.geojson (OSM/Geofabrik) | False | False | False | True |
| `nearest_road_distance_km_precomputed` | data/processed/grid_road_distance.parquet | False | False | False | True |
| `distance_to_laguna_de_bay` | assets/osm water layer (via grid_features.parquet) | False | False | False | True |
| `distance_to_other_water` | assets/osm water layer (via grid_features.parquet) | False | False | False | True |
| `distance_to_poblacion` | assets/boundaries barangay layer (via grid_features.parquet) | False | False | False | True |
| `distance_to_nearest_transportation_facility` | OSM POI layer (via grid_features.parquet) | False | False | False | True |
| `distance_to_nearest_tourist_attraction` | OSM POI layer (via grid_features.parquet) | False | False | False | True |
| `poi_count_restaurants` | OSM POI layer (via grid_features.parquet) | False | False | False | True |
| `poi_count_commercial` | OSM POI layer (via grid_features.parquet) | False | False | False | True |
| `poi_count_transportation` | OSM POI layer (via grid_features.parquet) | False | False | False | True |
| `poi_count_recreation` | OSM POI layer (via grid_features.parquet) | False | False | False | True |
| `poi_count_tourist_attraction` | OSM POI layer (via grid_features.parquet) | False | False | False | True |
| `poi_count_other_facilities` | OSM POI layer (via grid_features.parquet) | False | False | False | True |
| `poi_count_total` | OSM POI layer (via grid_features.parquet) | False | False | False | True |
| `poi_density_restaurants` | OSM POI layer (via grid_features.parquet) | False | False | False | True |
| `poi_density_commercial` | OSM POI layer (via grid_features.parquet) | False | False | False | True |
| `poi_density_transportation` | OSM POI layer (via grid_features.parquet) | False | False | False | True |
| `poi_density_recreation` | OSM POI layer (via grid_features.parquet) | False | False | False | True |
| `poi_density_tourist_attraction` | OSM POI layer (via grid_features.parquet) | False | False | False | True |
| `poi_density_other_facilities` | OSM POI layer (via grid_features.parquet) | False | False | False | True |
| `poi_density_total` | OSM POI layer (via grid_features.parquet) | False | False | False | True |
| `population` | PSA.json (municipality-level population) | False | False | False | True |
| `population_density_per_km2` | PSA.json (municipality-level population) | False | False | False | True |
| `distance_to_nearest_urban_cell_km` | grid_rural_classification.parquet | False | False | False | True |
| `urban_area_share_5km` | grid_rural_classification.parquet | False | False | False | True |
| `nearby_low_square_count_3km` | This experiment's own labeled cell table (self excluded) | True | False | False | True |
| `nearby_moderate_square_count_3km` | This experiment's own labeled cell table (self excluded) | True | False | False | True |
| `nearby_high_square_count_3km` | This experiment's own labeled cell table (self excluded) | True | False | False | True |
| `nearby_labeled_square_count_3km` | This experiment's own labeled cell table (self excluded) | True | False | False | True |
| `nearby_low_square_count_5km` | This experiment's own labeled cell table (self excluded) | True | False | False | True |
| `nearby_moderate_square_count_5km` | This experiment's own labeled cell table (self excluded) | True | False | False | True |
| `nearby_high_square_count_5km` | This experiment's own labeled cell table (self excluded) | True | False | False | True |
| `nearby_labeled_square_count_5km` | This experiment's own labeled cell table (self excluded) | True | False | False | True |

## Calculation detail

- **laguna_listings.json (active listings)** — Count/density of OTHER listings within radius, excluding listings inside the target cell. Uses the TTM revenue/occupancy window, same as the production surrounding features.
- **laguna_listings.json (active listings)** — Mean/median/total/std/IQR of ttm_revenue or ttm_occupancy among OTHER listings within radius. Same exclusion rule as SURROUNDING_REVENUE/SURROUNDING_OCCUPANCY in the production model.
- **laguna_listings.json (active listings)** — Count/percentage of OTHER listings within radius whose OWN 25/75-percentile-of-score tier (computed at listing level, not cell level) is Low/Moderate/High. Tiers are computed from every listing's own revenue/occupancy, not from any cell label.
- **poi_laguna.json (curated landscape places)** — Distance/count/density of landscape places (falls, mountains, lakes and ponds, rivers and landscape) within radius. poi_count_/poi_density_ here duplicate the production tourist-place count logic at additional radii.
- **poi_laguna.json category field** — Count of places in this specific real category within 5 km. Category values are exactly what is in the data (Falls, Mountains, Lakes & Ponds, Rivers & Landscape). None invented.
- **laguna_roads.geojson (OSM/Geofabrik)** — Straight-line distance from the cell centroid to the nearest mapped road (all roads, or filtered to road_class local/major/other), and road segment count/density within radius. road_class is a real tagged field on the road data; not invented.
- **data/processed/grid_road_distance.parquet** — The already-computed production road-distance column, joined by cell_id, for a cross-check against this file's own nearest_road_distance_km. Sanity check column, not a new feature.
- **assets/osm water layer (via grid_features.parquet)** — Distance from the cell centroid to the Laguna de Bay polygon / to other mapped water bodies. Already computed by the older non-rural pipeline; reused, not recomputed.
- **assets/boundaries barangay layer (via grid_features.parquet)** — Distance from the cell centroid to the nearest poblacion (town/city center) barangay. Already computed by the older non-rural pipeline; reused, not recomputed.
- **OSM POI layer (via grid_features.parquet)** — Distance to the nearest OSM-tagged transportation facility / tourist attraction. Already computed by the older non-rural pipeline; reused, not recomputed.
- **OSM POI layer (via grid_features.parquet)** — OSM point-of-interest counts/densities by category (a services/amenity proxy). Already computed by the older non-rural pipeline; reused, not recomputed. This is the closest available stand-in for Group F 'access to services' features; the dataset does not separately tag hospitals, markets, or supermarkets.
- **PSA.json (municipality-level population)** — Municipality population / population density, spatially joined onto the cell by dominant municipality overlap. Already computed by the older non-rural pipeline; reused, not recomputed. ADR-0013 caution: this value is constant within a municipality and correlates strongly with municipality identity, which is why the production model excludes municipality identity itself as a predictor. Treated here as a documented, not hidden, risk.
- **grid_rural_classification.parquet** — Distance to the nearest urban grid cell centroid, and the share of grid cells within radius classified urban. Reuses the existing rural/urban classification; a coarse, cell-count proxy, not a repeat of the polygon-area rural_area_share already on the grid.
- **This experiment's own labeled cell table (self excluded)** — Count of OTHER labeled cells within radius whose historical performance_class is Low/Moderate/High, or labeled at all. The target cell's own label is excluded by construction (the distance matrix diagonal is set to infinity before thresholding). Uses OTHER cells' target labels, the same way the production surrounding-market features use OTHER listings' revenue/occupancy. Never uses the target cell's own label. Flagged as uses_target_class_label=true because it is built from the target VARIABLE (just not this row's own value), so it deserves extra scrutiny even though it is not literally the same leak as using a cell's own historical performance to predict its own class.