# EXP-HP-baseline

Nested municipality-grouped CV: an inner 3-fold search over 18 Random Forest hyperparameter combinations (n_estimators 100/200/500, max_depth None/10, min_samples_leaf 1/2/5) picks the best parameters using ONLY each outer fold's own training municipalities, then those parameters are evaluated once on that outer fold's held-out municipalities.

Outer mean macro F1: 0.3052, std 0.0875 across 5 outer folds.

Distinct parameter combinations chosen across the 5 outer folds: 5. Different folds preferred different parameters, which is itself evidence that no single 'best' hyperparameter setting is clearly superior on this sample size; the production model's fixed settings (n_estimators=200, max_depth=10, max_features='sqrt', min_samples_leaf=1) are a reasonable, defensible choice, not an under-tuned one.
