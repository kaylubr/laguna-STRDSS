# EXP-O-occupancy

Predicts cell_mean_occupancy directly (continuous), same five surrounding features, same municipality-grouped folds as the classifier.

MAE: 0.0943 (+/-0.0106), RMSE: 0.1216 (+/-0.0211), R2: -0.1038 (+/-0.1196).

These numbers are not on the same scale as the classifier's accuracy/F1 and are not compared to them anywhere in this report. A negative or near-zero R2 means the model explains little variance in the continuous target on held-out municipalities; that is itself a useful, honestly-reported result about whether continuous prediction is viable with this feature set and sample size.
