# EXP-A02: feature_market

**Question.** Same question at 3 km.

**Added features.** airbnb_count_3km, avg_revenue_3km, avg_occupancy_3km

**Grouped-CV macro F1.** 0.3172 (baseline reproduced macro F1 is in experiments/baseline/metrics.json)

**Leave-one-municipality-out macro F1.** mean 0.3540, std across 28 municipality folds 0.2447.

A feature addition is only worth adopting if it helps on the grouped-CV score AND does not make the leave-one-municipality-out result meaningfully less consistent (lower mean, similar or lower std) than the baseline; see experiments/FINAL_REPORT.md for the cross-experiment comparison.
