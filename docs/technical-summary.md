# Laguna Rural Site-Candidate Random Forest

This is the primary site-similarity workflow. It asks: **How similar are empty rural grid cells to rural cells where an active STR is observed?** The score is a screening signal, not a forecast of revenue, occupancy, investment return, legality, or site feasibility.

## Study population

- Existing grid: 1 km cells in EPSG:32651; 1,764 retained cells.
- Analysis domain: rural cells with an assigned municipality; 1,041 cells.
- Labels: a cell is listed when it contains at least one active listing; otherwise it is empty.
- Rural label counts: 110 listed cells and 931 empty cells.
- Training sample: all 110 listed cells plus 110 empty cells randomly sampled with seed 42.

The workflow reads the existing grid and does not rebuild it. Urban, unclassified, and rural cells without a municipality are outside the scored domain.

## Six model features

1. `listed_places_within_radius`: all deduplicated curated landscape places within 5 km.
2. `distance_to_listed_tourist_place`: Haversine distance to the nearest curated landscape place.
3. `road_distance_km`: straight-line distance from the cell geometry centroid to the nearest mapped road.
4. `distance_to_nearest_town_center_km`: stored Haversine distance to a Poblacion anchor.
5. `distance_to_major_road_km`: straight-line distance from the cell geometry centroid to the nearest road tagged `major`.
6. `attraction_count_within_5km`: Falls and Mountains within 5 km.

These are location features; listing counts define labels and identify empty cells, but are not model predictors.

## Evaluation and scoring

The classifier is a 200-tree `RandomForestClassifier` with `max_depth=10`, `max_features="sqrt"`, `min_samples_leaf=1`, `min_samples_split=2`, `class_weight="balanced"`, and `random_state=42`.

Evaluation uses five-fold `StratifiedGroupKFold` by municipality. Each outer test fold contains municipalities excluded from that fold's training data. A grouped inner 3-fold procedure selects the Youden operating threshold from outer-training rows. Permutation importance is measured on held-out folds as the ROC-AUC drop after 20 shuffles per feature.

The reported mean held-out ROC-AUC is **0.7274** (sample SD **0.0922**); fold AUCs are **0.5833, 0.7361, 0.7595, 0.7208, and 0.8373**. These are evaluation scores from municipality-held-out predictions.

After evaluation, the forest is fit on all 220 sampled cells and scores the 1,041 rural cells with municipalities. The 931 empty cells are the candidate-screening population. The score file also contains scores for the 110 listed training cells; those scores are in-sample and must not be reported as model performance.

Scores are raw `predict_proba` outputs. Because training is balanced 110:110 while the rural cell population is 110:931, the scores are not calibrated population probabilities.

## Outputs

Run from the repository root:

```text
uv run python scripts/run_site_candidate.py
uv run python scripts/build_visualisation.py
```

The model run writes:

- `data/processed/rural/site_candidate_training.parquet`
- `data/processed/rural/site_candidate_oof_predictions.parquet`
- `data/processed/rural/site_candidate_scores.parquet`
- `data/processed/rural/site_candidate_report.json`

The map is `data/processed/frontend/index.html`; it colors only empty rural cells by RF score. Listed rural cells provide context, while urban and unclassified cells are not scored.
