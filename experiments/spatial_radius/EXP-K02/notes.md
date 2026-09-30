# EXP-K02: spatial_radius

**Question.** Same question at 3 km instead of 5 km.

**Added features.** nearby_high_square_count_3km, nearby_moderate_square_count_3km, nearby_low_square_count_3km

**Grouped-CV macro F1.** 0.3257 (baseline reproduced macro F1 is in experiments/baseline/metrics.json)

**Leave-one-municipality-out macro F1.** mean 0.2912, std across 28 municipality folds 0.2232.

A feature addition is only worth adopting if it helps on the grouped-CV score AND does not make the leave-one-municipality-out result meaningfully less consistent (lower mean, similar or lower std) than the baseline; see experiments/FINAL_REPORT.md for the cross-experiment comparison.
