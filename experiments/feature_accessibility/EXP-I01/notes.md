# EXP-I01: feature_accessibility

**Question.** Does distance to Laguna de Bay or to other mapped water add information?

**Added features.** distance_to_laguna_de_bay, distance_to_other_water

**Grouped-CV macro F1.** 0.3268 (baseline reproduced macro F1 is in experiments/baseline/metrics.json)

**Leave-one-municipality-out macro F1.** mean 0.3230, std across 28 municipality folds 0.2451.

A feature addition is only worth adopting if it helps on the grouped-CV score AND does not make the leave-one-municipality-out result meaningfully less consistent (lower mean, similar or lower std) than the baseline; see experiments/FINAL_REPORT.md for the cross-experiment comparison.
