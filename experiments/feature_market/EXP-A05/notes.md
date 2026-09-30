# EXP-A05: feature_market

**Question.** Does the market-distribution mix (share of nearby listings that are themselves historically Low/Moderate/High performers) add information beyond the mean?

**Added features.** high_performer_pct_5km, moderate_performer_pct_5km, low_performer_pct_5km

**Grouped-CV macro F1.** 0.3893 (baseline reproduced macro F1 is in experiments/baseline/metrics.json)

**Leave-one-municipality-out macro F1.** mean 0.3220, std across 28 municipality folds 0.2494.

A feature addition is only worth adopting if it helps on the grouped-CV score AND does not make the leave-one-municipality-out result meaningfully less consistent (lower mean, similar or lower std) than the baseline; see experiments/FINAL_REPORT.md for the cross-experiment comparison.
