# Hidden-label check of the six-feature forest

Each of the 220 seed-42 training cells was scored by a forest that did not train on that cell's municipality. The label was not an input. The predicted class uses the Youden cutoff chosen on the training side of that fold.

The held-out ROC-AUC is 0.727, PR-AUC is 0.713, and the mean of the five fold macro F1 scores is 0.656. Those match the original reported run.

## Original label against the prediction

|  | Predicted empty | Predicted listed |
| --- | ---: | ---: |
| Actually empty | 58 | 52 |
| Actually listed | 20 | 90 |

| Metric | This check | Original reported run |
| --- | ---: | ---: |
| ROC-AUC | 0.727 | 0.727 |
| PR-AUC | 0.713 | 0.713 |
| Macro F1, mean of five folds | 0.656 | 0.656 |
| Accuracy on all 220 predictions | 0.673 | not previously reported |
| Precision for listed | 0.634 | not previously reported |
| Recall for listed | 0.818 | not previously reported |
| F1 for listed | 0.714 | not previously reported |
| Macro F1 on the pooled 220 predictions | 0.666 | not previously reported |

The pooled macro F1 is 0.666 because it is computed once on all 220 predictions. The reported 0.656 is the average of the five fold scores. Both use the same predictions.

Of the 110 cells that really are listed, 90 were predicted listed and 20 were missed. Of the 110 empty cells, 52 were predicted listed.

## What this says about an unlabeled cell

This is the same municipality holdout as the original evaluation, with a yes/no class added. It is not a future check of cells that later gained a listing.

A predicted "listed" cell was actually listed 90 times out of 142 (90 + 52). That is too many false calls for the score to be treated as a reliable classification of a cell whose label is unknown. The ranking result, ROC-AUC 0.727, still stands: listed cells tend to score higher than empty cells, and the two groups still overlap.

Per-cell rows are in `cell_predictions.csv`.
