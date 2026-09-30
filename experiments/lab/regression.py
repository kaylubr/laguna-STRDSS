"""Regression alternative (experiment group O): predict revenue or occupancy
directly instead of a Low/Moderate/High class. Continuous metrics (MAE, RMSE, R2)
are not compared to the classifier's accuracy/F1; the point is only to see whether
continuous prediction looks more appropriate than the three-class split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from experiments.lab.common import RANDOM_STATE, grouped_kfold_splits, mean_std, regression_metrics


def evaluate_regression(
    frame: pd.DataFrame, feature_cols: list, target_col: str, group_col: str, **rf_overrides
) -> dict:
    features = frame[feature_cols]
    target = frame[target_col].astype(float)
    groups = frame[group_col].astype(str)
    dummy_classes = pd.qcut(target, q=3, labels=False, duplicates="drop")
    folds, scheme = grouped_kfold_splits(features, dummy_classes.fillna(0).astype(int), groups)
    fold_metrics = []
    for train_idx, test_idx in folds:
        params = dict(n_estimators=200, max_depth=10, max_features="sqrt", random_state=RANDOM_STATE)
        params.update(rf_overrides)
        model = RandomForestRegressor(n_jobs=-1, **params)
        model.fit(features.iloc[train_idx], target.iloc[train_idx])
        predicted = model.predict(features.iloc[test_idx])
        fold_metrics.append(regression_metrics(target.iloc[test_idx], predicted))
    combined = {"scheme": scheme, "n_folds": len(fold_metrics), "folds": fold_metrics}
    for key in ("mae", "rmse", "r2"):
        stats = mean_std([fold.get(key) for fold in fold_metrics])
        combined[key] = stats["mean"]
        combined[f"{key}_std"] = stats["std"]
    combined["n"] = int(sum(int(fold["n"]) for fold in fold_metrics))
    return combined
