---
Status: accepted
---

# The rural analytical domain: PSA barangay classification, and two different rural labels

The study is province-wide in extent, but the question it now answers is about **rural** Laguna.
Chapter 3 does not define rurality, does not say whether the rural/urban distinction is a property
of a listing or of a grid cell, and does not say what happens to cells that are not rural. Those
four silences are decisions recorded here.

**Rurality comes from the PSA, not from the data.** Each barangay carries an authoritative
`urban_rural` field (`R` / `U`) in the PSA PSGC release, and that field is joined to the barangay
polygons on PSGC code — PSA `code` against polygon `adm4_pcode` with the `PH` prefix stripped.
Barangay names are deliberately not used: 147 polygon names differ from the PSA names through
`(Pob.)` suffixes and encoding differences such as `BiÃ±an` against `Biñan`. Rurality is never
reconstructed from population, establishment counts, facility counts or any other threshold.

**The study area and the grid are untouched.** The entire Laguna boundary remains the extent, all
30 municipalities remain, and there is no "rural municipality" list. The 1 km grid, its 1,764
`cell_id` values, its CRS and its geometries are exactly those of the existing model; the rural
branch reuses the persisted `grid_features.parquet` rather than rebuilding a grid. All twelve
predictors, both targets, the 80:20 split, the cross-validation folds and the hyperparameter search
space are unchanged, so a second 1 km grid was not constructed merely to express rurality.

**A grid cell is rural by majority area, not by its centroid.** Each cell is intersected with the
PSA-classified barangay polygons and is classified rural when at least 50% of its
**barangay-covered** area lies in PSA-rural barangays, urban otherwise, and `unclassified` when no
PSA barangay overlaps it. Shares are taken over the covered area rather than the cell area, so a
cell half of which falls in a coverage gap is not pushed toward urban. The threshold mirrors the
grid's existing `MIN_LAND_COVERAGE = 0.5` convention. On the current data this yields **1,041
rural, 693 urban and 30 unclassified cells**; the 30 are cells the barangay dataset does not cover,
including the non-PSA *Sierra Madre Mountains Forest Land* polygon and the lakeside gaps in
eastern Cavinti.

**A listing and a grid cell are classified by different rules, on purpose.** A listing takes the
classification of the PSA barangay containing its coordinate. A cell takes the majority-area class
of its barangays. The two can disagree, and they are not reconciled: **30 of the 245 fitted rural
listings sit in cells whose majority area is urban** (their barangay is rural but the cell is
mostly urban, near a boundary or within coordinate jitter). The listing label is authoritative for
deciding who trains the model; the cell label is authoritative for deciding which cells are
predicted and scored. Relabelling the listing to match its cell would silently discard 30 rural
observations; relabelling the cell to match a minority of its area would corrupt the prediction
domain.

## Consequences

Rural AirROI listings are the training population: 439 rural listings narrow to 264 active
(`in_training_population`) and then to **245 fitted** listings in 108 cells across 21
municipalities. The rural Random Forests keep the existing procedure exactly — same features, same
split, same CV, same search space — and are fitted on those 245 observations.

Only rural cells are predicted and scored. The urban 693 and the unclassified 30 are **outside the
rural analytical domain**: they receive no predicted revenue, no predicted occupancy, no normalized
indicator, no entropy weight, no composite score and no rural suitability class. They are reported
as `Urban / Outside Rural Analysis` or `Unclassified / Outside Rural Analysis`, which is not a
statement that they are unsuitable. The Entropy Weight Method is computed over the rural cells
alone, so no urban cell can influence rural normalization, entropy, weights or scores; likewise
Jenks is run on rural scores only. This matters materially: POI density carries roughly 87% of the
rural weight, and 667 of the 1,041 rural cells have zero POI density.

The rural sample is **fitted but fragile**, and the model summary reports the evidence rather than
asserting validity. 931 of the 1,041 rural cells contain no training listing at all, 70 of the 108
occupied cells hold a single listing, the busiest holds 30, and the three largest municipalities
supply 58% of the fitted observations. The existing random `KFold` also leaks co-located listings
between folds, so the rural R² is not a generalisation estimate for either model, and the rural
occupancy model's test R² is negative. Those numbers are reported, not hidden.

Sensitivity at 500 m and 2 km remains unimplemented, as it was before this decision: the grid
builder accepts other cell sizes and the classification takes any grid, but no code path builds or
scores those grids. The rural rule is therefore expressed in a scale-agnostic way so that it will
apply consistently when those runs are built, and the 1 km result stays the only primary analysis.
