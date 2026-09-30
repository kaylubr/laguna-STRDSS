# EXP-A06: feature_market

**Question.** Does having every radius's raw count available at once (letting the forest choose scale) beat any single fixed radius?

**Added features.** airbnb_count_500m, airbnb_count_1km, airbnb_count_3km, airbnb_count_5km, airbnb_count_10km

**Grouped-CV macro F1.** 0.3111 (baseline reproduced macro F1 is in experiments/baseline/metrics.json)

**Leave-one-municipality-out macro F1.** mean 0.2944, std across 28 municipality folds 0.2483.

A feature addition is only worth adopting if it helps on the grouped-CV score AND does not make the leave-one-municipality-out result meaningfully less consistent (lower mean, similar or lower std) than the baseline; see experiments/FINAL_REPORT.md for the cross-experiment comparison.
