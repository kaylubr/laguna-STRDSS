# EXP-F01: feature_accessibility

**Question.** Does commercial/transport OSM POI density plus distance to the town center add information? (No hospital/market/supermarket tags exist in this dataset; this is a documented proxy, not the literal Group F feature list.)

**Added features.** poi_density_commercial, poi_density_transportation, distance_to_poblacion

**Grouped-CV macro F1.** 0.3175 (baseline reproduced macro F1 is in experiments/baseline/metrics.json)

**Leave-one-municipality-out macro F1.** mean 0.3411, std across 28 municipality folds 0.2614.

A feature addition is only worth adopting if it helps on the grouped-CV score AND does not make the leave-one-municipality-out result meaningfully less consistent (lower mean, similar or lower std) than the baseline; see experiments/FINAL_REPORT.md for the cross-experiment comparison.
