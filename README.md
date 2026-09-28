# STR Investment Suitability — Laguna

A desk study that scores every majority-land 1km × 1km grid cell in Laguna, Philippines on
short-term-rental investment suitability. It implements the methodology of Chapter 3 of the
thesis, held at `docs/thesis-chr3.md`, which is the source of truth for this build.

The output is a *relative* suitability class per cell. It is not a profitability forecast for any
individual property, and it is not field-verified.

## Method, in the order it runs

1. Preprocess each source separately — AirROI listings and monthly history, PSA population, OSM.
2. Build the validated target population: dormant listings removed, uncorroborated
   trailing-twelve-month revenue removed → 1,277 listings.
3. Construct the 1km grid over Laguna's derived land boundary → 1,764 cells, with 500m and 2km
   grids for the sensitivity runs.
4. Assign rasters to spatial units: every active listing to its containing cell (1,277 → 1,231
   listings inside the retained grid), every cell to the municipality it overlaps most.
5. Engineer four feature groups: accessibility, POI, tourism and demographic.
6. On cells that contain at least one active listing, build a historical performance score:
   min-max normalize that cell's own mean trailing-twelve-month revenue and occupancy, then
   average the two with equal weight. Label the bottom 25% Low, the middle 50% Moderate, and
   the top 25% High. Cells with no listing stay unlabeled. This score is only the training
   target. It is not an entropy weight and it is not applied again after prediction.
7. Train a three-class Random Forest on the surrounding market within 5 km, excluding the
   cell's own listings, plus the count of and distance to landscape places in `poi_laguna.json`
   (falls, mountains, lakes, and rivers). Every municipality fold is scored
   (`StratifiedGroupKFold`, 5 folds).
8. Compare the forest with a majority-class baseline and a stratified baseline using
   accuracy, balanced accuracy, macro precision, macro recall, macro F1, weighted F1, and
   one-vs-rest ROC-AUC. Means and standard deviations are taken across the five folds.
   Report the result even when the forest does not beat a baseline.
9. Apply the forest to every retained grid cell. The map class is the class with the highest
   of P(Low), P(Moderate), and P(High). Entropy weighting and Fisher-Jenks are not used.
   Radius, mean-versus-median, signal-group, and leaf-size checks are saved in the model
   summary. They do not replace the pre-specified 5 km mean model.

The class is an estimate of which historical performance band a cell's surroundings resemble.
It is not a forecast of profit. Across the five municipality folds the forest's mean macro F1
was 0.314 (SD 0.084), above a majority-class baseline of 0.221 and below a stratified baseline
of 0.368. Mean one-vs-rest ROC-AUC was 0.533, against 0.500 for the majority baseline and
0.533 for the stratified baseline.

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

## Status

The map shows the random forest's predicted performance class for all 1,764 retained grid cells.
Labels exist for the 288 cells that contain an active listing: 72 Low, 144 Moderate, and 72 High.
The other 1,476 cells are unlabeled and are predicted from their surroundings only.

Entropy weighting remains in the repository for the earlier composite-score experiments. It does
not enter `scripts/run_rural.py` or the map. The five-class Fisher-Jenks cut is not used.

## Environment

Python 3.14.4, managed with `uv`.

```
uv sync
uv run pytest
```
