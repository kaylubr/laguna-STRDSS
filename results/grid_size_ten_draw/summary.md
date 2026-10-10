# Grid size across ten empty-cell draws

Each draw keeps every listed rural cell and an equal number of empty rural cells. The empty cells change with the seed. The forest, the six features, and the five municipality folds stay the same. The 1 km row is the saved thesis grid. It reproduced the published ten-draw ROC-AUC: mean 0.695, range 0.621 to 0.763, and 0.727 at seed 42.

Standard deviation is the sample standard deviation across the ten draws.

| Grid | Listed | Empty | Training rows | ROC-AUC mean | ROC-AUC lowest | ROC-AUC highest | ROC-AUC std | PR-AUC mean | PR-AUC lowest | PR-AUC highest | PR-AUC std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 m | 122 | 4000 | 244 | 0.739 | 0.704 | 0.776 | 0.025 | 0.718 | 0.677 | 0.768 | 0.030 |
| 1000 m | 110 | 931 | 220 | 0.695 | 0.621 | 0.763 | 0.048 | 0.667 | 0.583 | 0.746 | 0.058 |
| 1500 m | 90 | 380 | 180 | 0.711 | 0.643 | 0.757 | 0.042 | 0.696 | 0.634 | 0.753 | 0.042 |
| 2000 m | 71 | 195 | 142 | 0.668 | 0.586 | 0.752 | 0.048 | 0.684 | 0.596 | 0.775 | 0.055 |

Paired with the 1 km draw that used the same empty-cell seed. A seed beats 1 km when its ROC-AUC is higher.

| Grid | Seeds above 1 km | Seeds below 1 km | Mean ROC-AUC difference | Std of the difference |
| --- | ---: | ---: | ---: | ---: |
| 500 m | 8 of 10 | 2 of 10 | +0.045 | 0.049 |
| 1500 m | 7 of 10 | 3 of 10 | +0.016 | 0.072 |
| 2000 m | 3 of 10 | 7 of 10 | -0.027 | 0.044 |

All empty rural cells, one run, no draw. Chance PR-AUC is the share of listed cells.

| Grid | ROC-AUC | PR-AUC | Chance PR-AUC |
| --- | ---: | ---: | ---: |
| 500 m | 0.723 | 0.091 | 0.030 |
| 1000 m | 0.695 | 0.219 | 0.106 |
| 1500 m | 0.729 | 0.388 | 0.191 |
| 2000 m | 0.677 | 0.469 | 0.267 |

## What is larger than the draw-to-draw spread

None of the three sizes is clearly better or clearly worse than 1 km once the empty-cell draw is allowed to change.

500 m is the closest to a real gain. Its mean ROC-AUC is 0.045 above the paired 1 km draw, and 8 of the 10 seeds are higher. The spread of those ten differences is 0.049, about the same size as the gain, so two seeds still fall below 1 km. The 500 m draws are the tightest of the four: they run from 0.704 to 0.776, while 1 km runs from 0.621 to 0.763.

1.5 km is 0.016 above 1 km, with 7 of 10 seeds higher. The spread of the differences is 0.072, several times the gap. That difference is noise.

2 km is 0.027 below 1 km, with only 3 of 10 seeds higher. The spread of the differences is 0.044, still larger than the gap. The direction leans lower, and one draw falls to 0.586, but it is not a clean loss on every seed.

The 1 km draws themselves span 0.143. Every mean gap above is smaller than that span.

The all-empty ROC-AUC values, which do not depend on a draw, are 0.723 at 500 m, 0.695 at 1 km, 0.729 at 1.5 km, and 0.677 at 2 km. The all-empty PR-AUC numbers are not on the same base: chance is 0.030 at 500 m and 0.106 at 1 km, because a smaller cell is listed less often. 500 m is about three times its chance. 1 km is about twice its chance.

The 1 km reproduction check passed, and the cell counts match the earlier builds. No check failed.

