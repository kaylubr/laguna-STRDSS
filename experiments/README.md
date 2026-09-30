# Experiment lab

This directory is isolated from the thesis map and from `scripts/run_rural.py`.

It lives on the `experiment/model-improvement` branch. Nothing here is merged into `2ndver` or `main`. Deleting the branch discards the experiments.

## What the current thesis model does

1. Laguna is cut into 1 km cells (1,764 retained).
2. Cells are rural, urban, or unclassified from PSA barangay area share.
3. Cells with at least one active Airbnb get a Low / Moderate / High label from their own trailing-twelve-month revenue and occupancy (25 / 50 / 25).
4. Features are surroundings only: nearby mean revenue, nearby mean occupancy, nearby listing count, listed landscape-place count, distance to the nearest listed place. Radius 5 km. Own-cell performance is never a feature.
5. A Random Forest (200 trees, depth 10, leaf 1, balanced) is scored with `StratifiedGroupKFold` by municipality, then refit on all 288 labeled cells.
6. The map colors rural cells only. Urban cells stay white.

Road distance is measured and was already tested. It is not in the five features that color the map.

## How to run

From `laguna-STRDSS`:

```
uv run python -m experiments.run_baseline
uv run python -m experiments.build_feature_table
uv run python -m experiments.run_feature_experiments
uv run python -m experiments.run_grid_size
uv run python -m experiments.run_target_definition
uv run python -m experiments.run_regression
uv run python -m experiments.run_model_comparison
uv run python -m experiments.run_feature_selection
uv run python -m experiments.run_hyperparameters
uv run python -m experiments.generate_findings
```

Hyperparameter search uses nested municipality folds. Outer test towns are never used to pick parameters.

## Layout

```
experiments/
    README.md
    results.csv
    FINAL_REPORT.md
    findings.html
    baseline/
    feature_engineering/
    feature_market/
    feature_poi/
    feature_road/
    feature_accessibility/
    feature_population/
    spatial_radius/
    grid_size/
    target_definition/
    model_comparison/
    hyperparameters/
    final_candidates/
```
