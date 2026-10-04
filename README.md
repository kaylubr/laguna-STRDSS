# Laguna Rural Site-Candidate Screener

The primary map uses a six-feature Random Forest to score empty rural 1 km grid cells by how
similar their location conditions are to rural cells where an active STR is observed. A score is
for screening and prioritisation; it is not a revenue forecast, probability calibrated to rural
prevalence, investment recommendation, or site-feasibility determination.

## Primary workflow

1. Read the validated active-listing targets, landscape-place file, road network, and existing
   1 km rural grid. The site-candidate workflow does not rebuild the grid.
2. Label each rural cell with an assigned municipality as listed when it contains at least one
   active listing, otherwise empty. The population is 110 listed and 931 empty cells.
3. Engineer six location features: landscape-place count and nearest-place distance, distance to
   a mapped road, distance to Poblacion, distance to a major road, and Falls/Mountains count.
4. Train and evaluate a Random Forest with five municipality-grouped folds. Outer test folds hold
   out municipalities; permutation importance is measured on those held-out folds.
5. Fit the same Random Forest on a balanced sample of 110 listed and 110 sampled empty cells.
   Score the rural grid and use the scores for the 931 empty rural cells as screening candidates.

The held-out mean ROC-AUC is 0.727 (sample SD 0.092). Raw scores for the balanced sample are not
calibrated probabilities for the full rural population. Training-cell scores in the full-grid
output are in-sample and are not model-performance results.

## Layout

```
assets/boundaries/   committed OSM boundary snapshots
data/                raw and derived data, not committed (see data/README.md)
docs/thesis-chr3.md  the methodology this build follows
docs/adr/            decisions taken where the methodology is silent
src/str_suitability/ pipeline code
tests/               unit tests
```

## Decisions where Chapter 3 is silent

Recorded as ADRs in `docs/adr/`. The load-bearing ones: the grid's projected CRS and cell
inclusion rule (0004), the exclusion of property characteristics from the models (0005), the
target-population exclusions (0006), the hyperparameter search protocol (0010), the municipality
assignment and PSGC reference (0011), what a cell's predicted value means (0012), and the
variables kept outside the models and indicator set (0013).

## Related project code

The repository also retains the earlier STR revenue/occupancy prediction and composite
suitability workflow. Those modules are separate from the primary site-candidate map and do not
produce its similarity scores.

## Environment

Python 3.14.4, managed with `uv`.

```
uv sync
uv run pytest
```
