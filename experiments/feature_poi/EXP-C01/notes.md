# EXP-C01: feature_poi

**Question.** Does a landscape-place count at 1 km add information beyond the baseline's 5 km listed_places_within_radius?

**Added features.** poi_count_1km

**Grouped-CV macro F1.** 0.3131 (baseline reproduced macro F1 is in experiments/baseline/metrics.json)

**Leave-one-municipality-out macro F1.** mean 0.3125, std across 28 municipality folds 0.2266.

A feature addition is only worth adopting if it helps on the grouped-CV score AND does not make the leave-one-municipality-out result meaningfully less consistent (lower mean, similar or lower std) than the baseline; see experiments/FINAL_REPORT.md for the cross-experiment comparison.
