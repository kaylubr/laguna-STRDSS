# EXP-A01: feature_market

**Question.** Does swapping the default 5 km market neighborhood for a 1 km one, added alongside the baseline 5 km features, change validation performance?

**Added features.** airbnb_count_1km, avg_revenue_1km, avg_occupancy_1km

**Grouped-CV macro F1.** 0.3400 (baseline reproduced macro F1 is in experiments/baseline/metrics.json)

**Leave-one-municipality-out macro F1.** mean 0.3133, std across 28 municipality folds 0.2982.

A feature addition is only worth adopting if it helps on the grouped-CV score AND does not make the leave-one-municipality-out result meaningfully less consistent (lower mean, similar or lower std) than the baseline; see experiments/FINAL_REPORT.md for the cross-experiment comparison.
