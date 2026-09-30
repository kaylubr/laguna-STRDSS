# EXP-O-revenue

Predicts cell_mean_revenue directly (continuous), same five surrounding features, same municipality-grouped folds as the classifier.

MAE: 340356.2184 (+/-68910.0714), RMSE: 487265.0946 (+/-176661.8186), R2: -0.3571 (+/-0.4306).

These numbers are not on the same scale as the classifier's accuracy/F1 and are not compared to them anywhere in this report. A negative or near-zero R2 means the model explains little variance in the continuous target on held-out municipalities; that is itself a useful, honestly-reported result about whether continuous prediction is viable with this feature set and sample size.
