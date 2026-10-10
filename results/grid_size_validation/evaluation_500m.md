# 500 m evaluation, same checks as the 1 km model

The forest, the six features, the municipality holdout, the Youden cutoff, and the ten negative-sample seeds are the ones used for the 1 km model. The 500 m balanced sample is 122 listed cells plus 122 empty cells. The 1 km sample is 110 plus 110. The all-empty run uses every empty rural cell.

| Check | 1 km | 500 m |
| --- | ---: | ---: |
| Seed 42 ROC-AUC | 0.727 | 0.776 |
| Seed 42 PR-AUC | 0.713 | 0.742 |
| Seed 42 macro F1 | 0.656 | 0.726 |
| Accuracy | 0.673 | 0.734 |
| Precision, listed | 0.634 | 0.699 |
| Recall, listed | 0.818 | 0.820 |
| F1, listed | 0.714 | 0.755 |
| Ten-seed ROC-AUC mean | 0.695 | 0.739 |
| Ten-seed ROC-AUC range | 0.621–0.763 | 0.704–0.776 |
| Ten-seed PR-AUC mean | 0.667 | 0.718 |
| All-empty ROC-AUC | 0.695 | 0.723 |
| All-empty PR-AUC | 0.219 | 0.091 |
| All-empty macro F1 | 0.479 | 0.360 |

Seed 42 confusion matrix. Rows are the actual class.

|  | Predicted empty | Predicted listed |
| --- | ---: | ---: |
| Actually empty | 79 | 43 |
| Actually listed | 22 | 100 |

The 1 km matrix was 58 true empty, 52 false listed, 20 missed listings, and 90 recovered listings.

Seed 42 permutation drop in ROC-AUC, largest first:

- road_distance_km: 0.121
- distance_to_major_road_km: 0.058
- distance_to_nearest_town_center_km: 0.032
- distance_to_listed_tourist_place: 0.015
- attraction_count_within_5km: 0.012
- listed_places_within_radius: 0.004
