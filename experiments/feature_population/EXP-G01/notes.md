# EXP-G01: feature_population

**Question.** Does municipality population density add information? (ADR-0013 caution: this value is constant within a municipality and correlates with municipality identity, which the production model deliberately excludes as a direct predictor.)

**Added features.** population_density_per_km2

**Grouped-CV macro F1.** 0.3352 (baseline reproduced macro F1 is in experiments/baseline/metrics.json)

**Leave-one-municipality-out macro F1.** mean 0.2658, std across 28 municipality folds 0.1721.

A feature addition is only worth adopting if it helps on the grouped-CV score AND does not make the leave-one-municipality-out result meaningfully less consistent (lower mean, similar or lower std) than the baseline; see experiments/FINAL_REPORT.md for the cross-experiment comparison.
