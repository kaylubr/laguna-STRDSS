---
Status: accepted
---

# The composite indicator set is expanded, and weighted by EWM blended with random-forest importance

The composite suitability score was dominated by a single indicator. Province-wide, the Entropy
Weight Method over the five thesis indicators gave `poi_density_total` a weight of **0.86**; the
rural run over the 1,041 majority-rural cells gave it **0.87**. Restricting the scope had not fixed
the concentration, and the concentration did not come from the scope.

## The diagnosis came first

Before changing the method, a diagnostic (`scripts/diagnose_indicators.py` →
`data/processed/indicator_diagnostics.json`) measured each indicator's distribution, dispersion,
entropy and EWM weight on both the full grid and the rural cells, and evaluated candidate remedies.
Two findings shaped the decision:

- **Decomposing POI made the concentration worse, not better.** With the six POI category densities
  scored separately, their **summed bloc weight rose to 0.97** (province) and **0.98** (rural): each
  zero-inflated category is individually high-entropy-deficient, and splitting one dominant
  indicator into six made six dominant indicators. This is consistent with the earlier POI audit,
  which found the bloc weight rising from 86% to 98% under a six-way split.
- **A value transform is not the cause or the cure.** `log1p` before min–max moved the EWM bloc
  weight only from 0.970 to 0.935 (all indicators) or 0.960 (zero-inflated only), and the hybrid
  bloc weight only from 0.686 to 0.668. The concentration is driven by the *dispersion* of the POI
  densities, not by their skew alone, so the transform was **evaluated and rejected** rather than
  adopted. The transformed and untransformed scenarios are recorded in the diagnostics.

## The decision

1. **The composite indicator set is expanded** to every individually-computed spatial indicator that
   is a *characteristic of a place*: the six POI category densities (the aggregate
   `poi_density_total` is dropped because it is their sum, and one tourism representation is kept),
   transport distance, attraction distance, `distance_to_laguna_de_bay`,
   `distance_to_other_water`, and `distance_to_poblacion`, alongside `predicted_revenue` and
   `predicted_occupancy`. `population_density_per_km2` stays out: it is a predictor of performance,
   not an attractiveness characteristic, consistent with ADR 0013.
2. **Weighting is a hybrid** of the Entropy Weight Method and the random-forest permutation
   importance: `w_j = 0.5·w_EWM_j + 0.5·w_RFI_j` (`BLEND_RATIO` in `config.py`). `w_RFI` is the
   permutation importance averaged across the revenue and occupancy models, each normalised over the
   same indicator set, with negative importances clipped to zero. Indicators that are not
   model inputs — the two predictions — carry no RFI and therefore keep their EWM weight, and the
   blend is renormalised to sum to 1. An **equal-weight reference** (1/13 per indicator) is computed
   alongside, so "the method did not fix it" rests on more than one method.

`distance_to_laguna_de_bay` and `distance_to_other_water` are two indicators, not one, because the
diagnosis showed distance-to-all-water and distance-to-the-lake correlate at only **0.41** and differ
by more than 1 km in **1,550 of 1,764** cells — proximity to a stream is not proximity to the lake.

`distance_to_poblacion` is normalised `negative` (closer is better). The evidence is mixed and
recorded: it correlates **+0.19** with predicted revenue but **−0.25** with predicted occupancy
province-wide (rural: +0.18 / −0.35). The occupancy correlation dominates in magnitude and agrees
with the a-priori expectation that town-centre proximity is an amenity, so the negative direction is
chosen and the ambiguity is reported rather than hidden.

## Consequences

- The province-wide POI bloc weight falls from 0.97 (EWM) to **0.69** (hybrid); the largest single
  indicator is `poi_density_recreation` at **0.23**, down from 0.86. The equal-weight reference puts
  the bloc at 0.46.
- **The POI bloc is still the largest bloc, and that is reported as a finding, not hidden.**
  `poi_density_recreation` carries the highest RFI of any composite indicator, so the model itself
  finds recreation POIs the strongest spatial signal — the hybrid lowering POI's weight and POI's
  RFI remaining highest are both true and both recorded.
- **The two predictions are diluted** to roughly 0.003 each: they have no RFI, so the blend halves
  their already-small EWM weight while the RFI-bearing spatial indicators grow. The composite is
  therefore closer to a spatial-amenity score than to a predicted-performance score. This is a
  recorded consequence of blending, not a silent one.
- All three weight sets — the legacy five-indicator EWM, the expanded-set EWM, and the expanded-set
  hybrid — are written side by side in `suitability_summary.json` for both the full-province and
  rural runs. The hybrid is the headline; the legacy set is retained as the documented "before".
