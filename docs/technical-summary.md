# Technical summary: Laguna rural location screener

This document describes the model, pipeline, and evaluation that color the current map. The question is:

**Does this rural square look like rural squares where an Airbnb already exists?**

The answer is a probability, then a class. A square is **Looks listed** when that probability is at least 0.5. Otherwise it is **Not the pattern**. The color is a shortlist for an investor to inspect. It is not a forecast that a new listing will earn money, and it is not a Low / Moderate / High earnings grade.

Numbers below are taken from `data/processed/rural/presence_screen.json` and the map build of `data/processed/frontend/index.html`. Means and standard deviations are across the five municipality folds. The standard deviation uses the sample formula (divide by \(n - 1\)).

## Study frame

| Piece | What it is |
| --- | --- |
| Area | Rural 1 km squares in Laguna |
| Examples | 110 rural squares with at least one active Airbnb |
| Comparison | 110 rural squares with none, drawn at random from 931 empty rural squares |
| Inputs | Scenic places within 5 km, distance to the nearest scenic place, distance to the nearest mapped road |
| Left out of the forest | Nearby revenue, nearby occupancy, nearby listing count, the house, the host, price, flood, utilities, safety |
| Test | Whole municipalities held out, five folds |
| Map flags | Nearby occupancy and nearby listing count, shown on the square and not used to color it |

Food, shops, a town center, population, hospitals, rooms, baths, building quality, land price, cost, return, flood, signal, water, electricity, and safety stay outside the color. The house and the site conditions are the investor’s check after a square is chosen.

## Data

| Item | Specification |
| --- | --- |
| Study area | Laguna, 30 municipalities |
| Listings | `data/laguna_listings.json`. Active trailing-twelve-month population, August 2025 through July 2026: 1,277 listings |
| Place file | `data/poi_laguna.json`. Categories stored: Falls, Lakes and ponds, Mountains, Rivers and landscape. Category is not a feature |
| Place records | 159 |
| Records without coordinates | 39. These are not used |
| Repeated place ids removed | 9. A place id that appears more than once is kept once |
| Places used | 111 |
| Roads | `data/laguna_roads.geojson`. OpenStreetMap extract. 56,776 features inside the map clip. Mapped roads, not a DPWH inventory |
| Grid | `data/processed/grid_features.parquet` and `data/processed/rural/grid_suitability.parquet`. 1,764 cells. The grid polygons are not rebuilt for this screen |
| Road distance | `data/processed/grid_road_distance.parquet`. One distance per cell |

A cell is rural when the PSA barangay area share inside it is at least 0.5. The grid has 1,041 rural cells, 693 urban cells, and 30 unclassified cells.

## Pipeline

The screen is `src/str_suitability/modeling/presence.py`, run by `scripts/run_presence_screen.py`. The map is `scripts/build_visualisation.py`.

1. Read the existing suitability grid. Cell ids, polygons, municipality, rural or urban class, and `listings_in_cell` stay as already stored.
2. Read `poi_laguna.json`. Drop records with no latitude or longitude. Drop later copies of the same non-empty `place_id`.
3. Recompute, for every cell, the count of those places within 5 km and the distance to the nearest one. Distance is haversine kilometres from the cell’s stored longitude and latitude. The radius is `NEIGHBORHOOD_RADIUS_KM` (5).
4. Join `road_distance_km`. That value was measured earlier from the cell centroid to the nearest mapped road in EPSG:32651 metres, then stored in kilometres. It is straight-line distance, not driving time, and not degrees of longitude and latitude.
5. Keep rural cells that have a municipality. In this run, zero rural cells lacked a municipality.
6. Label a rural cell **1** when `listings_in_cell` is at least 1, and **0** when it is 0. That count is active listings whose point falls inside the cell. It is not the count of listings within 5 km.
7. Take all 110 labeled 1 cells. Draw 110 labeled 0 cells at random from the 931 empty rural cells. The draw uses `random_state` 42, so the same empty squares are chosen on a repeat run.
8. Fit and score the forest on municipality folds, described under Evaluation. Metrics in the report come from those folds only.
9. Refit the same forest on all 220 training squares. Score all 1,764 cells, including urban and empty squares that were not in the random sample.
10. Write `data/processed/rural/presence_scores.parquet`, `presence_training.parquet`, and `presence_screen.json`.
11. Build the map. The viewer colors rural cells from the saved presence class. It refits the forest on `presence_training.parquet` with one job and checks that the new probabilities match the saved ones before it draws tree paths.

Commands, from `laguna-STRDSS`:

```text
uv run python scripts/run_presence_screen.py
uv run python scripts/build_visualisation.py
```

## What the forest sees

| Feature | Meaning | Missing rule |
| --- | --- | --- |
| `listed_places_within_radius` | Count of the 111 places within 5 km of the cell | Always a count, including 0 |
| `distance_to_listed_tourist_place` | Kilometres to the nearest of those places | Always present in this run |
| `road_distance_km` | Kilometres from the cell centroid to the nearest mapped road | Always present. 1,764 cells, no nulls |

These columns are not inputs:

| Column | Why it stays out |
| --- | --- |
| `surrounding_mean_revenue` | It would teach the forest to copy the existing Airbnb market |
| `surrounding_mean_occupancy` | Same |
| `surrounding_listing_count` | Same. A model that sees nearby listings mostly learns “near existing Airbnbs” |
| Own-cell revenue, occupancy, bedrooms, rating, price | Those describe the house or the result, not the place pattern |

On the map, nearby occupancy and the nearby listing count are still visible. They are shade options and modal flags. A dashed outline means no other Airbnb within 5 km. That mark is a flag beside the color.

## Classifier

| Setting | Value |
| --- | --- |
| Algorithm | `RandomForestClassifier` |
| Trees | 200 |
| `max_depth` | 10 |
| `max_features` | `sqrt` |
| `min_samples_leaf` | 1 |
| `min_samples_split` | 2 |
| `class_weight` | `balanced` |
| `random_state` | 42 |
| `n_jobs` | 1, so a map rebuild can match the saved probabilities |

The class on the map is not a separate rule inside the trees. After the 200 trees vote, the square is **Looks listed** when the share of votes for class 1 is at least 0.5.

There is no saved model file. The trees exist while `run_presence_screen.py` runs, and the map refits them from `presence_training.parquet`.

## Evaluation

### Split

`StratifiedGroupKFold`, 5 folds, shuffle, `random_state` 42. The group is the municipality. A town is entirely in training or entirely in test. Neighboring squares in the same town are not split across the two sides. The scheme stored on this run is `stratified_group_kfold_5`.

The 220 training squares are split into folds of 40, 42, 53, 44, and 41.

### What is scored

Each held-out square has a true label: it already contains an Airbnb, or it was one of the empty squares drawn for comparison. The forest’s call is correct when that label matches.

Empty rural squares that were not drawn into the 110 are still colored on the map. They have no label in this test, so the metrics do not grade them.

### Metrics

For one class:

\[
\text{Precision} = \frac{TP}{TP + FP}, \quad
\text{Recall} = \frac{TP}{TP + FN}, \quad
F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\]

Macro F1 is the unweighted mean of the two class F1 scores, “Not the pattern” and “Looks listed.” Each class counts once.

\[
\text{Macro F1} = \frac{F1_{\text{not the pattern}} + F1_{\text{looks listed}}}{2}
\]

The **0.570** in the result is the mean of that macro F1 across the five folds. It is not one F1 computed on the five folds stacked together. Those two calculations are close, and they are not the same number.

ROC-AUC is the ranking score: does a square that already has an Airbnb tend to receive a higher probability than an empty comparison square? A score of 0.5 is no ranking. Accuracy is the share of held-out squares whose class call matches the label.

### Baselines

Both baselines see the same folds.

| Baseline | Rule |
| --- | --- |
| Stratified guess | Draws labels from the class frequencies in the training towns. This is the frequency guess |
| Majority | Always predicts the most frequent training class in that fold |

A result counts as separating listed squares from unlisted ones when the forest’s mean ROC-AUC is above 0.5 and above the stratified guess. This run meets that rule.

## Results

| Metric | Forest mean (SD) | Stratified guess mean (SD) | Majority mean (SD) |
| --- | ---: | ---: | ---: |
| Accuracy | 0.575 (0.041) | 0.450 (0.027) | 0.442 (0.020) |
| Macro F1 | 0.570 (0.039) | 0.449 (0.027) | 0.307 (0.010) |
| ROC-AUC | 0.634 (0.095) | 0.458 (0.025) | 0.500 (0.000) |

Fold scores for the forest:

| Fold | Squares | Accuracy | Macro F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| 1 | 40 | 0.525 | 0.525 | 0.532 |
| 2 | 42 | 0.548 | 0.547 | 0.593 |
| 3 | 53 | 0.623 | 0.616 | 0.730 |
| 4 | 44 | 0.568 | 0.557 | 0.577 |
| 5 | 41 | 0.610 | 0.604 | 0.739 |

Summed confusion matrix across the five test folds. Rows are the true label. Columns are the call. Order is Not the pattern, then Looks listed.

|  | Called “Not the pattern” | Called “Looks listed” |
| --- | ---: | ---: |
| Truly empty | 60 | 50 |
| Truly listed | 43 | 67 |

Listed squares kept on the shortlist: **67 of 110**. Calls of “Looks listed”: 50 + 67 = **117**. Of those, 67 were squares that already had an Airbnb.

The fold ranking scores run from 0.532 to 0.739. The pattern holds on every fold in this run, and it holds more strongly in some towns than in others.

## Map

Viewer: `data/processed/frontend/index.html`. Tiles are Mapbox outdoors. The public token is read from `.mapbox_token` or `MAPBOX_TOKEN` and is not written here.

The forest scores all 1,764 cells. The viewer colors the 1,041 rural cells from the presence probability. The held-out test still uses 0.5 as the call. The map splits that probability into three colors so the middle is visible.

| Rural color | Cells | Rule |
| --- | ---: | --- |
| Red, Not the pattern | 474 | Probability below 0.40 |
| Yellow, Uncertain | 243 | Probability from 0.40 up to 0.60 |
| Green, Looks listed | 324 | Probability 0.60 and above |

Colors: red `#d73027`, yellow `#fee08b`, green `#1a9850`. Urban cells are white and labeled not included. Unclassified cells are left unfilled.

A cell click opens a card. The left side shows the two probabilities, the place count, the nearest place, and the road distance. Nearby occupancy and the nearby listing count are labeled as flags. The right side walks tree 1 of 200. The map color is the average of all 200 trees, so that one tree can disagree with the color. When it disagrees on a rural square, the card names the forest vote.

The shade menu can draw P(looks listed), nearby occupancy, nearby listings, place distance, place count, or road distance. Occupancy and listing count are marked as flags and do not change the class color.

## Earnings side check

This check does not color the map.

Rural squares with at least three active listings that have a bedroom count above zero: **14**. For each listing, trailing-twelve-month revenue is divided by bedrooms. The square’s value is the mean of those ratios. Bedrooms of zero are left out.

Spearman rank correlation with the same three surroundings, \(n = 14\):

| Feature | Spearman |
| --- | ---: |
| Places within 5 km | −0.168 |
| Distance to nearest place | −0.073 |
| Road distance | −0.187 |

A five-fold municipality regression of earnings per bedroom on those three inputs is stored in the report. With 14 squares the \(R^2\) is largely negative and is not a stable reading. The rank correlations are the usable check: these surroundings do not track earnings per bedroom in this small set.

## How to read a colored square

| What you see | What it means |
| --- | --- |
| Green, Looks listed | Probability is 0.60 or higher. The surroundings resemble rural squares that already have an Airbnb |
| Yellow, Uncertain | Probability is from 0.40 up to 0.60. The two sides are close |
| Red, Not the pattern | Probability is below 0.40. The surroundings resemble the empty comparison squares |
| Dashed outline, or a listing-count flag of 0 | No other Airbnb within 5 km. That is a market flag, not an input |
| Occupancy shown on the card | Nearby demand, for the investor to read beside the color |
| Urban, white | Outside the rural screen |
| No color | Unclassified. Outside the rural screen |

## Earlier earnings grade

An earlier forest graded squares Low, Moderate, or High from their own trailing-twelve-month revenue and occupancy, using nearby revenue, nearby occupancy, nearby listing count, place count, and place distance. On held-out towns its macro F1 was 0.314 against 0.368 for a frequency guess, and a High call matched a real top-quarter square 25 times out of 96. That grade is stored in `data/processed/rural/model_summary.json`. It is not the color on the current map.

## Code map

| Role | Path |
| --- | --- |
| Screen, sampling, metrics, earnings check | `src/str_suitability/modeling/presence.py` |
| Run the screen | `scripts/run_presence_screen.py` |
| Build the map | `scripts/build_visualisation.py` |
| Map template | `scripts/visualisation_template.html` |
| Saved metrics | `data/processed/rural/presence_screen.json` |
| Saved scores | `data/processed/rural/presence_scores.parquet` |
| Training rows used for the map refit | `data/processed/rural/presence_training.parquet` |
| Viewer | `data/processed/frontend/index.html` |
