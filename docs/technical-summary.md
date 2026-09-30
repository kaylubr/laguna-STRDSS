# Technical summary: Laguna rural location screener

This document describes the model and map that are current in the repository. It is a location screener for rural squares in Laguna. It assigns each rural square a Low, Moderate, or High probability from its surroundings, and tests whether those surroundings resemble areas where Airbnb performance was historically Low, Moderate, or High.

The class is the highest of P(Low), P(Moderate), and P(High). A High square is a weak screen for further looking. It is a resemblance to past performance. The held-out test is part of the result: on held-out towns the forest’s mean macro F1 was 0.314, a guess that uses the class frequencies scored 0.368, and a High call matched a real top-quarter square 25 times out of 96.

Numbers below are taken from `data/processed/rural/model_summary.json`, `data/processed/rural/road_accessibility_comparison.json`, `data/processed/grid_road_distance_summary.json`, and the current map build. Standard deviations are the sample standard deviation across the five municipality folds.

## Study frame

Rural Laguna grid, then four surroundings signals, then a random forest, then three probabilities, then a map that colors rural squares only.

| Signal | Variables | Role |
| --- | --- | --- |
| Nature | Count of listed landscape places within 5 km; distance to the nearest listed landscape place | Surroundings that match a rural stay oriented to landscape |
| Market | Mean trailing-twelve-month revenue and mean occupancy of other Airbnb listings within 5 km, outside the square | Observed nearby performance |
| Competition | Count of those other listings | Nearby supply |
| Access | Straight-line distance from the square’s centroid to the nearest mapped road | Measured and tested. Left out of the color because the gain was not on every held-out town |

Food, shops, a town center, population, hospitals, rooms, baths, building quality, land price, cost, return, flood, signal, water, electricity, and safety are outside this model. Shops stay out because the guest in this study is there to rest. The house, flood, utilities, and safety are the property check after a square is chosen.

## What the output means

Someone can say: this rural square’s highest probability is High, the held-out test shows that call is often wrong, and the place still has to be visited.

A square with no other Airbnb within 5 km is marked as no market evidence. That mark sits beside the probability. It is a blank market record. It is neither a proven gap nor proven demand.

Urban squares are scored by the same forest and then left white on the map. They are outside the rural screen. The historical labels used in the test come from every Laguna square that had an Airbnb, including urban squares. The map colors rural squares only.

## Data

| Item | Specification |
| --- | --- |
| Study area | Laguna, 30 municipalities |
| Listings | `data/laguna_listings.json`. Active trailing-twelve-month population, August 2025 through July 2026: 1,277 listings |
| Landscape places | `data/poi_laguna.json`. Falls, mountains, lakes and ponds, rivers and landscape. Counted together. Category is not a feature. The map build loaded 171 places |
| Roads | `data/laguna_roads.geojson`. OpenStreetMap extract from Geofabrik. 56,776 features. Mapped roads, not a DPWH inventory and not a legal classification |
| Grid | `data/processed/grid_features.parquet`. 1,764 cells. The grid is not rebuilt for later features |

Listing classification in the model report: 2,080 listings examined for rural or urban match, of which 439 rural, 1,589 urban, and 52 unmatched. The forest’s training labels use the 288 grid cells that contain at least one active listing, not the rural-listing subset from the older training block in the same JSON file.

## Grid

Cells are 1 km by 1 km. The stored grid uses EPSG:32651 (WGS 84 / UTM zone 51N). Laguna lies between 120°E and 126°E, so this zone’s axes are metres. Longitude and latitude are used to draw the map. Distances that must be metres, including road distance, are computed after projecting into EPSG:32651.

A cell is retained when it meets the land-coverage rule used to build the grid. `EXPECTED_CELL_COUNT` in config is 1,928 candidates. The study grid is the retained 1,764 cells. Cell ids and polygons stay fixed.

The road-distance centroid is `geometry.centroid` in EPSG:32651. The grid build stores a representative point, which can differ from the geometric centroid on a clipped cell. Road distance uses the geometric centroid.

## Rural and urban squares

A cell is rural when the PSA barangay area share inside it is at least 0.5. The model report counts 1,041 rural cells, 693 urban cells, and 30 unclassified cells. PSA coverage behind that rule: 681 PSA barangays (437 rural, 244 urban) and 682 polygons, with one polygon lacking a PSA record.

Rural status gates the map color. It does not gate training. Training uses every cell with at least one active listing, urban cells included. Dropping urban listing cells would shrink the 288-cell label set. The display choice is to color rural squares and to say, on the map, that the historical labels include urban squares.

## Historical label

Only cells with `listings_in_cell` of at least 1 are labeled. That is 288 cells. The other 1,476 cells stay unlabeled. An empty cell is not called Low.

On the labeled cells only:

1. Min-max normalize the cell’s own mean trailing-twelve-month revenue.
2. Min-max normalize the cell’s own mean occupancy.
3. Score = 0.5 × normalized revenue + 0.5 × normalized occupancy.

Cuts on that 0–1 score: Low at or below 0.06473022773613508 (25th percentile, 72 cells), High at or above 0.2300019191921241 (75th percentile, 72 cells), Moderate strictly between them (144 cells). High means the top quarter of existing Laguna listing cells. It does not mean a high peso return.

Own-cell revenue, own-cell occupancy, listings inside the cell, and the performance score are label columns. They are excluded from the feature list.

## Features of the map model

Radius is 5 km, haversine, from the cell. Surrounding features use listings within 5 km and outside the cell. A listing inside the cell can enter the label. It does not enter that cell’s features.

| Feature | Group | Missing rule |
| --- | --- | --- |
| `surrounding_mean_revenue` | Market | Blank when `surrounding_listing_count` is 0. Blank is not filled with zero |
| `surrounding_mean_occupancy` | Market | Same |
| `surrounding_listing_count` | Competition | 0 is an observed zero |
| `listed_places_within_radius` | Nature | Count of listed landscape places within 5 km |
| `distance_to_listed_tourist_place` | Nature | Distance to the nearest listed landscape place |

Median revenue and median occupancy are computed and stored. They are not in the map model.

122 cells have `surrounding_listing_count` of 0 at 5 km. Among labeled cells at 5 km, that count is 0. All 122 are unlabeled. Among rural cells on the map, 984 have at least one other Airbnb within 5 km and 57 have none. The 57 keep a class color and a dashed outline. The outline means no market evidence.

The forest accepts the blank market means through its missing-value branch and still uses the landscape features. A blank mean is missing evidence. A true zero mean can occur only when surrounding listings exist and their mean is zero.

## Classifier

`RandomForestClassifier` from scikit-learn, native missing values, no imputation.

| Setting | Value |
| --- | --- |
| Trees | 200 |
| Max depth | 10 |
| Max features | sqrt |
| Min samples per leaf | 1 |
| Min samples to split | 2 |
| Class weight | balanced |
| Random state | 42 |
| `n_jobs` | −1 |

There is no grid search on this classifier. `PARAM_GRID` in config belongs to an older regressor and is not used by `scripts/run_rural.py`.

The predicted class is the argmax of the three probabilities, not a separate decision rule. A square at 0.40 / 0.35 / 0.25 receives the same class label as a square at 0.90 / 0.05 / 0.05. The probabilities are what differ. The map model is refit on all 288 labeled cells after the cross-validation metrics are computed. Validation metrics come from the held-out folds, not from that final refit.

## Validation

`StratifiedGroupKFold`, 5 folds, shuffle, random state 42. The group is the municipality, so a whole town is held out together. Neighboring squares in the same town are not split across train and test. A grouped shuffle is the fallback only if a class has fewer than two municipality groups. That fallback was not required: the scheme on the saved model is `stratified_group_kfold_5`.

Fold test sizes: 65, 54, 60, 57, 52. They sum to 288.

| Metric | Forest mean (SD) | Majority baseline mean (SD) | Stratified baseline mean (SD) |
| --- | --- | --- | --- |
| Accuracy | 0.361 (0.066) | 0.498 (0.044) | 0.425 (0.137) |
| Balanced accuracy | 0.326 (0.091) | 0.333 (0.000) | 0.368 (0.125) |
| Macro F1 | 0.314 (0.084) | 0.221 (0.013) | 0.368 (0.127) |
| Weighted F1 | 0.358 (0.070) | 0.332 (0.049) | 0.424 (0.139) |
| ROC-AUC, one-vs-rest macro | 0.533 (0.072) | 0.500 (0.000) | 0.532 (0.101) |

The majority baseline always predicts Moderate, the largest class. The stratified baseline draws labels from the class frequencies. The forest’s macro F1 is above the majority baseline and below the stratified baseline. Accuracy is below both. ROC-AUC matches the stratified baseline at about 0.53. Fold macro F1 for the forest runs from 0.184 to 0.375.

Summed confusion matrix across the five test folds, rows true Low / Moderate / High, columns predicted Low / Moderate / High:

|  | Predicted Low | Predicted Moderate | Predicted High |
| --- | ---: | ---: | ---: |
| True Low | 14 | 38 | 20 |
| True Moderate | 28 | 65 | 51 |
| True High | 21 | 26 | 25 |

High calls on held-out labeled cells: 20 + 51 + 25 = 96. Of those, 25 were actually High. Recall of High is 25 of 72.

`cells_outside_training_feature_range` in the model report is 0, computed against the first fold’s training range only. The map model is fit on all labeled cells. The zero does not describe the 122 cells whose market means are blank. Those cells are still scored.

## Experiments that did not replace the model

The same municipality folds were reused. A higher score was recorded and was not adopted. The map stays at 5 km, means, all five features, and `min_samples_leaf` 1.

| Check | Macro F1 mean (SD) | ROC-AUC mean (SD) | Note |
| --- | --- | --- | --- |
| Tourist features only | 0.358 (0.057) | 0.547 (0.072) | Place count and distance only |
| Market and competition | 0.333 (0.056) | 0.554 (0.046) | Revenue, occupancy, listing count |
| All five features | 0.314 (0.084) | 0.533 (0.072) | The map model |
| All five minus market and competition, paired | −0.019 (0.095) | −0.021 (0.072) | Tourist features do not add a stable gain |
| Radius 1 km, all signals, means | 0.319 (0.058) | 0.535 (0.081) | 77 labeled cells with no surrounding listings |
| Radius 3 km | 0.344 (0.044) | 0.522 (0.042) | 1 such labeled cell |
| Radius 5 km | 0.314 (0.084) | 0.533 (0.072) | 0 such labeled cells |
| Medians instead of means, all signals, 5 km | 0.359 (0.053) | 0.529 (0.045) | Not adopted |

Leaf size was compared on inner municipality folds of each outer training set. The outer test fold was not used to pick the leaf. Inner macro F1: leaf 1 at 0.338, leaf 2 at 0.336, leaf 5 at 0.332, leaf 10 at 0.320. Leaf 1 was chosen on 3 of 5 training sets, leaf 2 on 1, leaf 5 on 1. Zero folds were skipped. The final leaf stays 1.

## Permutation importance

Importance is the change in macro F1 when a feature is shuffled on a held-out municipality fold, 30 repeats, then averaged across the five folds. A positive value would mean shuffling lowered macro F1. Every mean is negative. This is a predictive association, not a cause.

| Feature | Mean change in macro F1 | SD across folds |
| --- | ---: | ---: |
| Surrounding listing count | −0.008 | 0.067 |
| Distance to listed place | −0.010 | 0.058 |
| Place count | −0.027 | 0.039 |
| Surrounding mean occupancy | −0.032 | 0.071 |
| Surrounding mean revenue | −0.043 | 0.027 |

Surrounding revenue is negative on all five folds. The other four features change sign across folds.

Impurity importance, averaged across the evaluation models, is a different quantity. It describes how often a feature was used to split, not whether shuffling it hurt held-out macro F1. Means: distance 0.233, occupancy 0.230, revenue 0.213, listing count 0.211, place count 0.113.

## Road distance

Candidate feature only. File: `data/processed/grid_road_distance.parquet`, one row per `cell_id`, 1,764 rows, no nulls, all distances at least 0.

The distance is the straight line from the geometric centroid to the nearest line in the full mapped network. `road_class` is not a filter. Local, major, and other roads all count, including residential, service, track, path, and footway. A nearer local road is kept over a farther major road. It is not driving time, not a major-road-only distance, and not a bus stop.

| Statistic | Metres | Kilometres |
| --- | ---: | ---: |
| Minimum | 0.083 | 0.000083 |
| Median | 145.96 | 0.146 |
| Mean | 306.98 | 0.307 |
| Maximum | 3,644.30 | 3.644 |

`data/poi_laguna_with_road_distance.json` is a place-level file. It is not the grid distance and is not reused for cells.

Model B adds `road_distance_km` to the five features. Labels, the 288 cells, the municipality groups, the five splits, and the forest settings match Model A.

| Metric | Model B mean (SD) | Paired B − A mean (SD) |
| --- | ---: | ---: |
| Accuracy | 0.415 (0.080) | +0.053 (0.090) |
| Balanced accuracy | 0.378 (0.097) | +0.052 (0.077) |
| Macro F1 | 0.355 (0.106) | +0.041 (0.082) |
| Weighted F1 | 0.398 (0.095) | +0.040 (0.086) |
| ROC-AUC | 0.546 (0.059) | +0.013 (0.034) |

The average rose. Fold 3 was worse on both macro F1 and ROC-AUC, and fold 1 did not gain ROC-AUC. The reading stored with the comparison is that the gain was not present on every municipality fold, so the five-feature model was left unchanged. `final_model_changed` is false. A higher average would still be an association, not a cause.

The map draws the road lines and shows each cell’s road distance. The distance is not one of the five features that set the color. The roads layer can be turned off.

## Map

Viewer: `data/processed/frontend/index.html`, built by `scripts/build_visualisation.py` from `scripts/visualisation_template.html`. Tiles are Mapbox outdoors. The public token is read from `.mapbox_token` or `MAPBOX_TOKEN` and is not written into documentation.

The forest scores all 1,764 cells. The viewer colors the 1,041 rural cells only:

| Rural class | Cells |
| --- | ---: |
| Low | 410 |
| Moderate | 481 |
| High | 150 |

Colors: Low `#d73027`, Moderate `#fee08b`, High `#1a9850`. Urban cells are white and labeled not included. Unclassified cells are left unfilled and labeled not included. A dashed outline on a rural cell means no other Airbnb within 5 km.

The side panel states the holdout in one line: macro F1 0.314 versus a frequency-matched guess at 0.368, and High matched 25 of 96.

A cell click opens a card. The left side is the class, the three probabilities, and the measured surroundings: place count, nearest place, listing count, revenue, occupancy, and road distance. The right side walks tree 1 of 200 for that square. Each step shows the feature, the comparison, and Yes or No. The other branch is marked not taken. The card ends with that tree’s vote and the leaf’s three class shares. The map color is the average of all 200 trees, so tree 1’s leaf can differ from the color. When it differs on a rural square, the card names the forest vote.

Parallel fitting (`n_jobs=-1`) can move class counts between refits. The viewer rebuild checks that a refit with the same settings reproduces the saved probabilities before it embeds tree paths.

## How to read an empty rural square

| Situation | What the model does |
| --- | --- |
| Rural, no Airbnb inside the square, other Airbnbs within 5 km | Unlabeled. Colored from surrounding revenue, occupancy, listing count, and landscape places. Solid outline |
| Rural, no Airbnb within 5 km | Unlabeled. Revenue and occupancy stay blank. Listing count is 0. Landscape features still enter. Colored, with a dashed outline: no market evidence |
| Urban, with or without an Airbnb | Scored, then painted white. Not part of the rural screen |

## Code path

Run from `laguna-STRDSS` with `uv`.

| Step | Location |
| --- | --- |
| Rural pipeline | `scripts/run_rural.py` → `src/str_suitability/rural/pipeline.py` |
| Labels and forest | `src/str_suitability/modeling/location_classifier.py` |
| Signal, radius, mean or median, and leaf checks | `src/str_suitability/modeling/experiments.py` |
| Road distance | `src/str_suitability/features/road_distance.py`, `scripts/build_grid_road_distance.py` |
| Five-feature versus six-feature comparison | `scripts/compare_road_accessibility.py` |
| One-tree text paths | `src/str_suitability/rural/diagrams.py` |
| Map | `scripts/build_visualisation.py` |

Saved outputs: `data/processed/rural/model_summary.json`, `data/processed/rural/grid_suitability.parquet`, `data/processed/rural/location_training_cells.parquet`, `data/processed/rural/road_accessibility_comparison.json`, `data/processed/grid_road_distance.parquet`.

## Methods that remain in the repository and are not the map

Entropy weights, hybrid weights, the fixed site-score weights, Fisher-Jenks classes, and the revenue and occupancy regressors are still in the tree. `scripts/run_rural.py` does not call them. The map class is the random-forest argmax. Older single-holdout figures, a binary success model, and five Jenks classes are earlier results and are not the current test.
