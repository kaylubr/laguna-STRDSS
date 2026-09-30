# EXP-P-gradient_boosting

Same baseline 5 features, same target, same municipality-grouped folds as every other model in this comparison.

Preprocessing: median imputation before fitting (model cannot take NaN natively).

Grouped-CV macro F1: 0.3518. LOMO mean macro F1: 0.2912, std 0.2112.
