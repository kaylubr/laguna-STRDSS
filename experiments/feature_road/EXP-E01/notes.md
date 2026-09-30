# EXP-E01: feature_road

**Question.** Independent replication: does straight-line distance to the nearest mapped road add information? (The production model already tested this in compare_road_accessibility and did not adopt it; this is a separate re-test.)

**Added features.** nearest_road_distance_km

**Grouped-CV macro F1.** 0.3675 (baseline reproduced macro F1 is in experiments/baseline/metrics.json)

**Leave-one-municipality-out macro F1.** mean 0.3105, std across 28 municipality folds 0.2808.

A feature addition is only worth adopting if it helps on the grouped-CV score AND does not make the leave-one-municipality-out result meaningfully less consistent (lower mean, similar or lower std) than the baseline; see experiments/FINAL_REPORT.md for the cross-experiment comparison.
