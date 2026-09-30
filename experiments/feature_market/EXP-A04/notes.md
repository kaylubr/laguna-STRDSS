# EXP-A04: feature_market

**Question.** Does adding listing density (count / circle area) at the production radius help beyond the raw surrounding listing count already in the baseline?

**Added features.** airbnb_density_5km

**Grouped-CV macro F1.** 0.3358 (baseline reproduced macro F1 is in experiments/baseline/metrics.json)

**Leave-one-municipality-out macro F1.** mean 0.2916, std across 28 municipality folds 0.2086.

A feature addition is only worth adopting if it helps on the grouped-CV score AND does not make the leave-one-municipality-out result meaningfully less consistent (lower mean, similar or lower std) than the baseline; see experiments/FINAL_REPORT.md for the cross-experiment comparison.
