import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    train_test_split,
)

from str_suitability.config import CV_FOLDS, PARAM_GRID, RANDOM_STATE, TEST_SIZE

GROUP_COLUMN = "cell_id"
GROUPED_SCHEME = "grouped"
RANDOM_SCHEME = "random"


def split_observations(
    features: pd.DataFrame, target: pd.Series, groups: pd.Series | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if groups is None:
        return train_test_split(
            features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_index, test_index = next(splitter.split(features, target, groups))
    return (
        features.iloc[train_index],
        features.iloc[test_index],
        target.iloc[train_index],
        target.iloc[test_index],
    )


def tune_and_train(
    features: pd.DataFrame, target: pd.Series, groups: pd.Series | None = None
) -> tuple[RandomForestRegressor, dict, float]:
    search = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE),
        PARAM_GRID,
        cv=GroupKFold(n_splits=CV_FOLDS) if groups is not None else CV_FOLDS,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    if groups is None:
        search.fit(features, target)
    else:
        search.fit(features, target, groups=groups)
    return search.best_estimator_, dict(search.best_params_), float(search.best_score_)


def train_model(
    features: pd.DataFrame, target: pd.Series, groups: pd.Series | None = None
) -> dict:
    features_train, features_test, target_train, target_test = split_observations(
        features, target, groups
    )
    train_groups = None if groups is None else groups.loc[features_train.index]
    model, best_params, best_cv_score = tune_and_train(features_train, target_train, train_groups)
    return {
        "model": model,
        "best_params": best_params,
        "best_cv_rmse": -best_cv_score,
        "features_train": features_train,
        "features_test": features_test,
        "target_train": target_train,
        "target_test": target_test,
        "scheme": GROUPED_SCHEME if groups is not None else RANDOM_SCHEME,
    }
