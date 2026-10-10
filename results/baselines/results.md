# How the main forest compares with simpler baselines

This test asks whether the six-feature random forest ranks listed rural cells better than a simpler score on the same cells. The comparison uses the original 1 km grid. The main forest was checked first. Seed 42 is 0.727 and the ten-draw mean is 0.695, matching the published result. The other scores were then computed on those same rows.

## How each score is made

Every score uses the same municipality-grouped five-fold split, the same ten empty-cell draws, and one run that keeps every empty cell. The seeds are 7, 13, 21, 42, 99, 101, 202, 303, 404, and 505. Each draw keeps all 110 listed rural cells and 110 empty rural cells. The all-empty run keeps all 110 listed cells and all 931 empty cells.

ROC-AUC is the primary measure. PR-AUC is the secondary measure. Both are computed on each held-out fold, then averaged over the five folds. The difference from the main forest is paired by seed: that score's ROC-AUC minus the main forest's ROC-AUC on the same empty cells. The standard deviation is the sample standard deviation of those ten differences.

The five scores are:

- **Nearest major road.** The score is the negative of the distance to the nearest major road. No model is fitted. A closer road scores higher.
- **Nearest mapped road.** The score is the negative of the distance to the nearest mapped road of any class. No model is fitted.
- **Logistic regression.** All six features. Each feature is transformed with log1p, then standardised on the training fold. Class weights are balanced. Nothing is tuned.
- **Random forest, roads only.** The same forest settings as the main model, using only the two road distances. With two features, the square-root setting considers one feature at each split.
- **Main random forest.** All six features. This is the reference.

## Ten balanced draws

The mean PR-AUC is also from these balanced draws. Chance PR-AUC on a balanced draw is 0.50.

| Score | Mean ROC-AUC | Lowest | Highest | All-empty ROC-AUC | Mean PR-AUC | Difference vs main forest | SD of difference | Seeds above the main forest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Nearest major road | 0.712 | 0.681 | 0.741 | 0.719 | 0.676 | +0.017 | 0.036 | 6 of 10 |
| Nearest mapped road | 0.698 | 0.652 | 0.753 | 0.710 | 0.678 | +0.004 | 0.034 | 5 of 10 |
| Logistic regression, six features | 0.733 | 0.703 | 0.768 | 0.751 | 0.693 | +0.038 | 0.042 | 8 of 10 |
| Random forest, two road distances | 0.674 | 0.609 | 0.722 | 0.703 | 0.652 | −0.021 | 0.042 | 3 of 10 |
| Main random forest, six features | 0.695 | 0.621 | 0.763 | 0.695 | 0.667 | 0 | 0 | 0 of 10 |

## What the test says

Logistic regression has the highest ten-draw ROC-AUC, 0.733, and the highest all-empty ROC-AUC, 0.751. It is above the main forest on 8 of the 10 seeds. The mean paired difference is +0.038 and the paired standard deviation is 0.042, so the gain is about the same size as the draw-to-draw spread.

Sorting cells by nearness to a major road, with no model fitted, averages 0.712. It is above the main forest on 6 of 10 seeds. The mean paired difference is +0.017 and the paired standard deviation is 0.036. Sorting by the nearest mapped road of any class averages 0.698 and is above the main forest on 5 of 10 seeds. That difference is +0.004, with a paired standard deviation of 0.034.

A random forest that sees only the two road distances averages 0.674. It is above the main forest on 3 of 10 seeds. The mean paired difference is −0.021 and the paired standard deviation is 0.042.

The reproduction check passed. Seed 42 for the main forest is 0.727, and the ten-draw mean is 0.695.
