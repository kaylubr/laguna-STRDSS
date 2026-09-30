# EXP-K01: spatial_radius

**Question.** Does knowing how many nearby OTHER labeled squares were historically Low/Moderate/High (self excluded) add information, the same way the production surrounding-market features use other listings?

**Added features.** nearby_high_square_count_5km, nearby_moderate_square_count_5km, nearby_low_square_count_5km

**Grouped-CV macro F1.** 0.3449 (baseline reproduced macro F1 is in experiments/baseline/metrics.json)

**Leave-one-municipality-out macro F1.** mean 0.2853, std across 28 municipality folds 0.2129.

A feature addition is only worth adopting if it helps on the grouped-CV score AND does not make the leave-one-municipality-out result meaningfully less consistent (lower mean, similar or lower std) than the baseline; see experiments/FINAL_REPORT.md for the cross-experiment comparison.
