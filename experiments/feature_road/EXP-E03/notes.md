# EXP-E03: feature_road

**Question.** Does road density within 3 km (a network-coverage signal, not just nearest distance) help?

**Added features.** road_count_3km, road_density_3km

**Grouped-CV macro F1.** 0.3830 (baseline reproduced macro F1 is in experiments/baseline/metrics.json)

**Leave-one-municipality-out macro F1.** mean 0.3190, std across 28 municipality folds 0.2379.

A feature addition is only worth adopting if it helps on the grouped-CV score AND does not make the leave-one-municipality-out result meaningfully less consistent (lower mean, similar or lower std) than the baseline; see experiments/FINAL_REPORT.md for the cross-experiment comparison.
