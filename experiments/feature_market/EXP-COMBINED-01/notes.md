# EXP-COMBINED-01: feature_market

**Question.** Combines every individual feature addition above that beat the baseline on BOTH grouped-CV macro F1 and mean leave-one-municipality-out macro F1: EXP-A01, EXP-A02, EXP-A05, EXP-B02, EXP-C02, EXP-E01, EXP-E03, EXP-F01, EXP-I01, EXP-J01. This tests whether the individual gains stack, or cancel out / overfit when combined.

**Added features.** airbnb_count_1km, airbnb_count_3km, avg_occupancy_1km, avg_occupancy_3km, avg_revenue_1km, avg_revenue_3km, distance_to_laguna_de_bay, distance_to_nearest_urban_cell_km, distance_to_other_water, distance_to_poblacion, high_performer_pct_5km, low_performer_pct_5km, moderate_performer_pct_5km, nearest_road_distance_km, occupancy_iqr_5km, poi_count_10km, poi_density_commercial, poi_density_transportation, revenue_iqr_5km, road_count_3km, road_density_3km, urban_area_share_5km

**Grouped-CV macro F1.** 0.3719 (baseline reproduced macro F1 is in experiments/baseline/metrics.json)

**Leave-one-municipality-out macro F1.** mean 0.3452, std across 28 municipality folds 0.2631.

A feature addition is only worth adopting if it helps on the grouped-CV score AND does not make the leave-one-municipality-out result meaningfully less consistent (lower mean, similar or lower std) than the baseline; see experiments/FINAL_REPORT.md for the cross-experiment comparison.
