# EXP-J01: feature_accessibility

**Question.** Does distance to the nearest urban grid cell, or the urban share of cells within 5 km, add information beyond the market/tourist features already in the baseline?

**Added features.** distance_to_nearest_urban_cell_km, urban_area_share_5km

**Grouped-CV macro F1.** 0.3499 (baseline reproduced macro F1 is in experiments/baseline/metrics.json)

**Leave-one-municipality-out macro F1.** mean 0.3534, std across 28 municipality folds 0.2546.

A feature addition is only worth adopting if it helps on the grouped-CV score AND does not make the leave-one-municipality-out result meaningfully less consistent (lower mean, similar or lower std) than the baseline; see experiments/FINAL_REPORT.md for the cross-experiment comparison.
