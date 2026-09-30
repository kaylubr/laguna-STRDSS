# Distance to the nearest mapped road

This is a candidate accessibility feature. It is not in the five-feature map model, and it is not assumed to improve that model.

The measurement is the straight-line distance from the geometric centroid of each existing grid polygon to the nearest line in the mapped Laguna road network. It is not a driving distance, a travel time, a route, a distance to a major road only, or a distance to a bus stop or terminal.

## Source

`data/laguna_roads.geojson` is an OpenStreetMap extract distributed by Geofabrik. It is the project's mapped Laguna road network. It is not official Philippine government data, not a DPWH inventory, and not a legal road classification.

Every feature in that file is eligible. The `road_class` field (`local`, `major`, `other`) is not used as a filter. Highway types in the file, including residential, tertiary, service, track, path, and footway, all count as mapped roads for this first measurement. A closer local road is preferred over a farther major road.

## Grid

The cells are the existing 1,764 rows in `data/processed/grid_features.parquet`. The grid is not rebuilt. Cell ids and polygon geometries are unchanged.

The centroid is the geometric centroid of each stored cell polygon (`geometry.centroid`) after the polygon is in the distance CRS.

## CRS and units

Distances are calculated in EPSG:32651, WGS 84 / UTM zone 51N. Laguna sits between 120°E and 126°E, which is this zone, and the axes are metres. The existing grid is already stored in EPSG:32651. The roads are stored as longitude/latitude (CRS84) and are projected into EPSG:32651 only for the distance calculation. The original road file and the original grid geometries are not rewritten.

A degree-based distance would treat longitude and latitude as if they were metres. That is why the projected CRS is required.

`road_distance_m` is the result in metres. `road_distance_km` is that value divided by 1000.

Output: `data/processed/grid_road_distance.parquet`, one row per `cell_id`.

## What the number is not

A cell with no nearby Airbnb still has a road distance. The road distance does not fill in missing revenue or occupancy, and a long distance to a road is not, by itself, an investment opportunity or a lack of one.

## Controlled comparison

Model A is the current five features. Model B adds `road_distance_km`. Labels, the 288 cells, the municipality groups, the five `StratifiedGroupKFold` splits, and the forest settings are the same. The map model stays Model A.

On the latest run the mean macro F1 moved from 0.314 (SD 0.084) to 0.355 (SD 0.106), a paired change of +0.041 (SD 0.082). Mean ROC-AUC moved from 0.533 (SD 0.072) to 0.546 (SD 0.059), a paired change of +0.013 (SD 0.034). Fold 3 was worse on both, and fold 1 did not gain ROC-AUC. The average rose. The gain was not on every held-out town. Road distance stays out of the final model. A higher score would still be an association, not a cause.
