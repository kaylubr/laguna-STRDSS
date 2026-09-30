# Baseline (spec step 2/3)

This reproduces the production 5-feature Random Forest classifier's
municipality-grouped validation using a separate implementation of the
same splitting rule and metrics (experiments/lab/common.py). It is not a
copy of model_summary.json's numbers; it is a fresh fit-and-score.

Random Forest macro F1 (reproduced): 0.3136
Stratified baseline macro F1 (reproduced): 0.3675
Majority baseline macro F1 (reproduced): 0.2214
Predicted High: 96, actually High among those: 25, actual High total: 72

## From the current thesis model's own model_summary.json (not modified)
macro_f1: 0.31359021982352775
stratified_baseline macro_f1: 0.36753619031429274

## Comparison
The reproduced macro F1 matches model_summary.json within 0.0000 (StratifiedGroupKFold with shuffle=True and a fixed random_state gives the same folds for the same input rows, so a small residual difference, if any, is expected floating-point / library-version noise, not a methodology error).