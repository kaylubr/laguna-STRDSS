# EXP-C03: feature_poi

**Question.** Does the nearest-place distance and average distance to the 3 nearest places add information beyond the baseline's single nearest-place distance and a 5 km count?

**Added features.** nearest_poi_distance_km, avg_distance_3_nearest_poi_km

**Grouped-CV macro F1.** 0.3370 (baseline reproduced macro F1 is in experiments/baseline/metrics.json)

**Leave-one-municipality-out macro F1.** mean 0.2811, std across 28 municipality folds 0.2133.

A feature addition is only worth adopting if it helps on the grouped-CV score AND does not make the leave-one-municipality-out result meaningfully less consistent (lower mean, similar or lower std) than the baseline; see experiments/FINAL_REPORT.md for the cross-experiment comparison.
