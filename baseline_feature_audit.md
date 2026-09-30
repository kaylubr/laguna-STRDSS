# Baseline feature audit

This audit was made before `site_candidate_v2` was compared with the baseline. It traces the three features used by `src/str_suitability/modeling/presence.py`.

The word `listed` in the feature names means a place that is on the curated landscape list in `data/poi_laguna.json`. It does not mean an Airbnb listing.

## `listed_places_within_radius`

| Item | Finding |
| --- | --- |
| Source file | `data/poi_laguna.json` |
| Source columns | `latitude`, `longitude`, `place_id`, `name` |
| What the file contains | Falls, lakes and ponds, mountains, and rivers and landscape. Category is stored and is not used for this count |
| Filter | Rows without a numeric latitude or longitude are dropped |
| De-duplication | A non-empty `place_id` is kept once, first row wins. Rows with a blank `place_id` are kept |
| Spatial operation | Haversine kilometres from the cell longitude and latitude to each place. The count is places at or within 5 km (`NEIGHBORHOOD_RADIUS_KM`) |
| Units | Count of places |
| Airbnb in the calculation | No. Listings, revenue, occupancy, price, bedrooms, reviews, ratings, and hosts are not read for this feature |

Code: `listed_places_for_training` and `refresh_listed_places` in `presence.py`. The distance function is `haversine_km` in `src/str_suitability/spatial/haversine.py`, which multiplies the central angle by `EARTH_RADIUS_KM`.

## `distance_to_listed_tourist_place`

| Item | Finding |
| --- | --- |
| Source file | The same `data/poi_laguna.json` rows that enter the count above |
| Source columns | `latitude`, `longitude`, and `name` for the nearest place label |
| Filter and de-duplication | Identical to `listed_places_within_radius` |
| Spatial operation | Minimum haversine kilometres from the cell to those places |
| Units | Kilometres |
| Airbnb in the calculation | No |

The nearest-place name is stored for the map. It is not a model feature.

## `road_distance_km`

| Item | Finding |
| --- | --- |
| Source file | `data/laguna_roads.geojson` |
| Source columns | Road geometry. `road_class` and `highway_type` are present on the file and are not used as a filter for this feature |
| Filter | Empty geometries are dropped. Every remaining mapped road is eligible. `road_distance.py` states that `road_class` is not a filter |
| Spatial operation | Cell centroid to the nearest road with `sjoin_nearest`, after both layers are projected to EPSG:32651. The grid centroid is `geometry.centroid` in that CRS. Longitude and latitude degrees are not the distance |
| Units | Metres in `road_distance_m`, then kilometres in `road_distance_km` by dividing by 1000. The CRS axis unit is metres |
| Stored table | `data/processed/grid_road_distance.parquet`. The presence screen joins this table. It does not recompute the baseline distance |
| Airbnb in the calculation | No |

## Conclusion

None of the three baseline features uses Airbnb listing counts, revenue, occupancy, price, bedrooms, reviews, ratings, hosts, or any other field derived from `laguna_listings.json`.

`listings_in_cell` is the training label. It is not one of the three features.

The baseline is not contaminated. v2 may be compared against these three features unchanged.
