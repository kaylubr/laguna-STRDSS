# Grid-size evaluation

Same six-feature random forest, seed 42, municipality-grouped five-fold holdout. Each size is a new grid. The saved 1 km files were not replaced.

The rebuilt 1 km mean ROC-AUC is 0.729. The saved run is 0.727. Those two are close enough that the other sizes can be compared with this same build.

| Cell size | Rural cells | Listed | Empty | Training rows | Mean ROC-AUC | Mean PR-AUC | Mean macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 m | 4,122 | 122 | 4,000 | 244 | 0.776 | 0.742 | 0.726 |
| 1 km (rebuilt) | 1,041 | 110 | 931 | 220 | 0.729 | 0.716 | 0.651 |
| 1.5 km | 470 | 90 | 380 | 180 | 0.745 | 0.704 | 0.681 |
| 2 km | 266 | 71 | 195 | 142 | 0.645 | 0.687 | 0.521 |

**500 m is the highest on all three scores.** Then 1.5 km, then 1 km, then 2 km on ROC-AUC and macro F1. On PR-AUC the order is 500 m, 1 km, 1.5 km, 2 km.

Fold ROC-AUC:

- 500 m: 0.788, 0.824, 0.733, 0.807, 0.726
- 1 km: 0.576, 0.738, 0.764, 0.727, 0.837
- 1.5 km: 0.803, 0.842, 0.701, 0.728, 0.653
- 2 km: 0.753, 0.641, 0.577, 0.441, 0.812

A larger cell puts more listings into fewer cells, so the training sample shrinks from 244 rows at 500 m to 142 rows at 2 km. The 2 km grid also has one fold below 0.50.

500 m is finer than the coordinate jitter described for this study (about 150 m). A listing can fall in a neighbouring 500 m cell because of that jitter. The higher 500 m score is the number this test produced. It is not a reason to replace the 1 km map.
