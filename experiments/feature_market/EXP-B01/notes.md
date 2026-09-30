# EXP-B01: feature_market

**Question.** Does the SPREAD of nearby revenue/occupancy (not just the mean) carry information?

**Added features.** revenue_std_5km, occupancy_std_5km

**Grouped-CV macro F1.** 0.3174 (baseline reproduced macro F1 is in experiments/baseline/metrics.json)

**Leave-one-municipality-out macro F1.** mean 0.2518, std across 28 municipality folds 0.2024.

A feature addition is only worth adopting if it helps on the grouped-CV score AND does not make the leave-one-municipality-out result meaningfully less consistent (lower mean, similar or lower std) than the baseline; see experiments/FINAL_REPORT.md for the cross-experiment comparison.
