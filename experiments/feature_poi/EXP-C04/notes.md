# EXP-C04: feature_poi

**Question.** Does knowing WHICH kind of landscape place is nearby (not just how many) help, using the real categories in poi_laguna.json?

**Added features.** falls_count_5km, mountains_count_5km, lakes_ponds_count_5km, rivers_landscape_count_5km

**Grouped-CV macro F1.** 0.3126 (baseline reproduced macro F1 is in experiments/baseline/metrics.json)

**Leave-one-municipality-out macro F1.** mean 0.3235, std across 28 municipality folds 0.2252.

A feature addition is only worth adopting if it helps on the grouped-CV score AND does not make the leave-one-municipality-out result meaningfully less consistent (lower mean, similar or lower std) than the baseline; see experiments/FINAL_REPORT.md for the cross-experiment comparison.
