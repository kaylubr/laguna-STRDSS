# Experiment report

This report is for the `experiment/model-improvement` branch. It does not replace the thesis map or the five-feature forest on `2ndver`.

## Current baseline

The production model is a Random Forest (200 trees, depth 10, `max_features=sqrt`, leaf 1, class weight balanced, random state 42) trained on 288 labeled 1 km cells. Features: surrounding mean revenue, surrounding mean occupancy, surrounding listing count, listed landscape-place count, distance to the nearest listed place. Validation is `StratifiedGroupKFold` by municipality, 5 folds.

| Metric | Forest | Stratified guess | Majority (always Moderate) |
| --- | ---: | ---: | ---: |
| Macro F1 | 0.314 | 0.368 | 0.221 |
| Accuracy | 0.361 | 0.425 | 0.498 |
| Macro precision | 0.329 | — | — |
| Macro recall | 0.326 | — | — |
| Weighted F1 | 0.358 | — | — |
| High precision | 0.213 | — | — |
| High recall | 0.339 | — | — |

Held-out High calls: 96 predicted High, 25 actually High, 72 true High cells. High success rate 25/96. Leave-one-municipality-out mean macro F1: 0.292 (SD 0.241).

These numbers match `data/processed/rural/model_summary.json`.

## Validation

Every classification experiment reused the same municipality-grouped 5-fold split as the thesis model. Leave-one-municipality-out was recorded beside it. Hyperparameters were chosen on inner training-town folds only.

A configuration is treated as an improvement only when grouped-CV macro F1 and leave-one-municipality-out mean macro F1 both rise. A jump in one town is not enough.

## Feature experiments

Highest observed grouped-CV macro F1 among feature additions:

| Experiment | Grouped macro F1 | LOMO mean | High precision | High recall |
| --- | ---: | ---: | ---: | ---: |
| EXP-A05 | 0.389 | 0.322 | 0.340 | 0.470 |
| EXP-E03 | 0.383 | 0.319 | 0.276 | 0.339 |
| EXP-COMBINED-01 | 0.372 | 0.345 | 0.313 | 0.335 |
| EXP-E01 | 0.368 | 0.310 | 0.295 | 0.394 |
| EXP-SEL-lean | 0.363 | 0.328 | 0.290 | 0.297 |
| EXP-SEL-decorrelated | 0.355 | 0.317 | 0.304 | 0.324 |
| EXP-J01 | 0.350 | 0.353 | 0.238 | 0.327 |
| EXP-B02 | 0.347 | 0.312 | 0.270 | 0.398 |
| EXP-K01 | 0.345 | 0.285 | 0.283 | 0.350 |
| EXP-A01 | 0.340 | 0.313 | 0.290 | 0.351 |
| EXP-C03 | 0.337 | 0.281 | 0.229 | 0.312 |
| EXP-A04 | 0.336 | 0.292 | 0.226 | 0.339 |

EXP-A05 (nearby Low/Moderate/High mix of other listings) had the highest grouped macro F1 (0.389). EXP-J01 (urban proximity) and EXP-A02 (3 km market) had the highest leave-one-municipality-out means. EXP-COMBINED-01 stacked every addition that beat the baseline on both scores and reached grouped macro F1 0.372 and LOMO 0.345.

Road density (EXP-E03) and nearest-road distance (EXP-E01) also rose on grouped CV. Population density (EXP-G01) rose on grouped CV and fell on leave-one-municipality-out, so it is not treated as an improvement.

Elevation, slope, and protected-area distance were not built. Those layers are not in the repository. Laguna de Bay and other water used existing OSM-derived grid columns.

## Spatial experiments

| Radius / setup | Grouped macro F1 | LOMO mean |
| --- | ---: | ---: |
| 1 km market added (EXP-A01) | 0.340 | 0.313 |
| 3 km market added (EXP-A02) | 0.317 | 0.354 |
| 10 km market added (EXP-A03) | 0.329 | 0.231 |
| 1 km POI count (EXP-C01) | 0.313 | 0.313 |
| 10 km POI count (EXP-C02) | 0.318 | 0.293 |

A 3 km market neighborhood was the most stable extra market scale. 10 km raised grouped F1 and lowered leave-one-municipality-out.

## Grid comparison

| Grid | Labeled cells | Grouped macro F1 | LOMO mean |
| --- | ---: | ---: | ---: |
| 500 m | 409 | 0.320 | 0.312 |
| 1 km (thesis) | 288 | 0.314 | 0.292 |
| 2 km | 160 | 0.294 | 0.224 |

The 1 km grid remains the most defensible size. 500 m added labels but not a stable gain. 2 km lost labels and score.

## Model comparison

Same five features, same municipality folds.

| Model | Grouped macro F1 | LOMO mean | High precision |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.305 | 0.261 | 0.392 |
| Decision tree | 0.286 | 0.301 | 0.273 |
| Random Forest | 0.314 | 0.292 | 0.213 |
| Gradient boosting | 0.352 | 0.291 | 0.317 |

Gradient boosting had the highest grouped macro F1. Its leave-one-municipality-out mean stayed near the forest. Logistic regression had higher High precision and a lower macro F1.

## Target experiments

| Target | Grouped macro F1 | Note |
| --- | ---: | --- |
| 25 / 50 / 25 (thesis) | 0.314 | Official label |
| 20 / 60 / 20 | 0.334 | Not the thesis target |
| 30 / 40 / 30 | 0.336 | Not the thesis target |
| Binary High vs not | 0.496 | Not comparable to three-class F1 |

Regression of own-cell revenue and occupancy produced negative R² on held-out towns. Continuous prediction did not fit this sample.

## Hyperparameters

Nested search over 18 Random Forest settings:

| Feature set | Nested outer macro F1 |
| --- | ---: |
| Five production features | 0.305 |
| EXP-COMBINED-01 features | 0.357 |

Different outer folds chose different settings. Nested search on the five features scored below the fixed production settings. The production hyperparameters stay the ones to keep.

## Highest-observed configurations

Wording is "highest observed validation score," not "best model for the thesis."

1. EXP-A05, grouped macro F1 0.389. Nearby class mix of other listings.
2. EXP-E03, grouped macro F1 0.383. Road density within 3 km.
3. EXP-COMBINED-01, grouped macro F1 0.372, LOMO 0.345. Stacked additions.
4. EXP-SEL-lean, grouped macro F1 0.363, LOMO 0.328. Fewer stacked features.
5. Gradient boosting on the five production features, grouped macro F1 0.352.

None of these beat the stratified guess of 0.368 on grouped macro F1 except EXP-A05 (0.389) and EXP-E03 (0.383) and the stacked sets. Those gains are still modest, and fold-to-fold spread remains large.

## Generalization

Leave-one-municipality-out standard deviations stay around 0.17–0.30. Small towns with two or three labeled cells swing the score. A feature that helps Calamba can fail in Cavinti.

## Limitations

- 288 labeled cells, unevenly spread across about 30 towns.
- Many rural cells have no Airbnb, so they have no true class for this test.
- Historical Airbnb performance is not demand, profit, or a building decision.
- POI importance (major vs minor) has no objective field in `poi_laguna.json`.
- Elevation and slope were not available.
- Municipality population density tracks town identity.
- A higher score is an association on held-out towns, not a cause.

## Recommendation for the next thesis experiment

Investigate the nearby class-mix of other listings (EXP-A05) and 3 km road density (EXP-E03) on the same municipality folds, one group at a time, without stacking every extra column. Keep the 1 km grid, the 25 / 50 / 25 label, and the five-feature Random Forest as the official map until a later review says otherwise.
