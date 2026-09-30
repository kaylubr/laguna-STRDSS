# EXP-E02: feature_road

**Question.** Does distinguishing major vs. local road distance (using the real road_class field) help more than one undifferentiated nearest-road distance?

**Added features.** nearest_major_road_distance_km, nearest_local_road_distance_km

**Grouped-CV macro F1.** 0.3226 (baseline reproduced macro F1 is in experiments/baseline/metrics.json)

**Leave-one-municipality-out macro F1.** mean 0.2835, std across 28 municipality folds 0.2419.

A feature addition is only worth adopting if it helps on the grouped-CV score AND does not make the leave-one-municipality-out result meaningfully less consistent (lower mean, similar or lower std) than the baseline; see experiments/FINAL_REPORT.md for the cross-experiment comparison.
