# STR Investment Suitability — Laguna

A desk study that ranks 1km grid cells across the province of Laguna, Philippines by how attractive they are for short-term-rental investment. The output is a relative ranking within the province, not a profitability forecast for any individual property.

## Language

**Grid cell**:
A fixed 1km x 1km square of the surface of Laguna. The spatial unit that everything in the final assessment is reported per.
_Avoid_: Tile, patch, zone, hex

**Suitability indicator**:
A single measured attribute of a grid cell that contributes to the composite assessment. The thesis specifies five: predicted annual revenue, predicted occupancy, POI density, distance to the nearest tourist attraction, and distance to the nearest transportation facility.
_Avoid_: Factor, variable, metric, criterion

**Composite indicator set**:
The expanded set of place-characteristics the composite suitability score is currently computed over: the six POI category densities, the distances to the nearest transport facility, tourist attraction, Laguna de Bay, other water, and the poblacion, alongside predicted revenue and predicted occupancy. Population density and the aggregate POI density are deliberately outside it.
_Avoid_: Feature set, predictor set

**POI bloc**:
The six POI category densities taken together. Their summed weight is the measure of how far the composite leans on points of interest.
_Avoid_: POI factor, POI score

**Hybrid weight**:
An indicator's weight after blending its entropy weight with its random-forest permutation importance. The plain entropy weight is retained alongside it for comparison.
_Avoid_: Blended weight, composite weight

**Equal-weight reference**:
A uniform weighting (one over the number of indicators) computed alongside the entropy and hybrid weights, as a method-independent reference point.
_Avoid_: Baseline weight, naive score

**Composite suitability score**:
The single number per grid cell produced by weighting all suitability indicators and summing them.
_Avoid_: Suitability index, score, rating

**Suitability class**:
One of the five ordered bands (Very High through Very Low) that a grid cell's composite suitability score falls into.
_Avoid_: Rating, grade, tier

**Listing**:
One short-term-rental property as recorded by AirROI, identified by its listing ID.
_Avoid_: Property, unit, rental, Airbnb

**Active listing**:
A listing whose trailing-twelve-month revenue and occupancy are both non-zero and whose revenue the monthly history corroborates. The modelling population is drawn from these.
_Avoid_: Valid listing, live listing

**Dormant listing**:
A listing whose trailing-twelve-month revenue and occupancy are both zero because the property was not operating, as distinct from one whose figures are simply missing. Dormant listings are excluded from model training.
_Avoid_: Inactive listing, zero listing, dead listing

**Trailing twelve months (TTM)**:
The August 2025 to July 2026 window over which listing revenue and occupancy are measured.
_Avoid_: Annual, last year, prior period

**Predicted STR performance potential**:
The revenue or occupancy a trained model predicts for a grid cell's location and demographic characteristics. It is a modelled expectation for that location, not an observation of any property.
_Avoid_: Predicted performance, forecast, observed performance

**Municipality**:
One of the 30 city or municipal administrative units of Laguna. The coarsest geographic grouping used in reporting.
_Avoid_: Town, LGU, city

**Barangay**:
A sub-municipality administrative unit. No barangay-level attribute data is currently held for the province.
_Avoid_: Village, district, neighborhood

**Sensitivity resolution**:
A re-run of the whole assessment at a different grid cell size (500m and 2km alongside the primary 1km) to test whether the resulting suitability pattern depends on the resolution chosen.
_Avoid_: Scale test, robustness check
