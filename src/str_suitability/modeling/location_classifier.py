"""Random Forest of historical Airbnb performance.

The label is a three-class reading of each labeled cell's own revenue and
occupancy. The features are the surrounding market and the curated landscape
places in poi_laguna.json, excluding that cell's own listings. The map class
is the forest's predicted class. Entropy weights are not used, and the
historical score is not applied again after prediction.
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from str_suitability import config
from str_suitability.rural.classify import CELL_ID_COLUMN
from str_suitability.rural.site_score import (
    DISTANCE_COLUMN,
    LISTED_PLACES_WITHIN_RADIUS,
    LONGITUDE_COLUMN,
    LATITUDE_COLUMN,
    NEAREST_PLACE_COLUMN,
)
from str_suitability.spatial.haversine import haversine_km

SURROUNDING_REVENUE = "surrounding_mean_revenue"
SURROUNDING_OCCUPANCY = "surrounding_mean_occupancy"
SURROUNDING_MEDIAN_REVENUE = "surrounding_median_revenue"
SURROUNDING_MEDIAN_OCCUPANCY = "surrounding_median_occupancy"
SURROUNDING_LISTINGS = "surrounding_listing_count"
LISTINGS_IN_CELL = "listings_in_cell"
CELL_REVENUE = "cell_mean_revenue"
CELL_OCCUPANCY = "cell_mean_occupancy"
PERFORMANCE_SCORE = "historical_performance_score"
PERFORMANCE_CLASS = "performance_class"
MUNICIPALITY_COLUMN = "municipality"
LOW_PROBABILITY = "rf_low_probability"
MODERATE_PROBABILITY = "rf_moderate_probability"
HIGH_PROBABILITY = "rf_high_probability"
PREDICTED_CLASS = "rf_predicted_class"
CLASS_COLUMN = PREDICTED_CLASS
PROBABILITY_COLUMNS = (LOW_PROBABILITY, MODERATE_PROBABILITY, HIGH_PROBABILITY)

MARKET_FEATURES = (SURROUNDING_REVENUE, SURROUNDING_OCCUPANCY)
COMPETITION_FEATURES = (SURROUNDING_LISTINGS,)
TOURIST_FEATURES = (LISTED_PLACES_WITHIN_RADIUS, DISTANCE_COLUMN)
CLASSIFIER_FEATURES = MARKET_FEATURES + COMPETITION_FEATURES + TOURIST_FEATURES
MEDIAN_MARKET_FEATURES = (SURROUNDING_MEDIAN_REVENUE, SURROUNDING_MEDIAN_OCCUPANCY)
CV_METRIC_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "roc_auc_ovr_macro",
)
IMPORTANCE_NOTE = (
    "Permutation importance is the change in macro F1 when a feature is shuffled on a "
    "held-out municipality fold. importance_mean averages those fold-level results. "
    "importance_std is the sample standard deviation across folds. A positive value means "
    "shuffling lowered macro F1. This is a predictive association, not a cause."
)
LABEL_COLUMNS = (CELL_REVENUE, CELL_OCCUPANCY, LISTINGS_IN_CELL, PERFORMANCE_SCORE, PERFORMANCE_CLASS)

PLACES_DESCRIPTION = (
    "poi_laguna.json is a curated list of landscape places: falls, mountains, "
    "lakes and ponds, and rivers and landscape. The model counts all of them "
    "together. It does not include OpenStreetMap shops, restaurants, or transport."
)
GROUP_SCHEME = "stratified_group_kfold"
FALLBACK_SCHEME = "group_shuffle"


def market_features(
    grid: pd.DataFrame,
    listings: pd.DataFrame,
    places: pd.DataFrame,
    radius_km: float = config.NEIGHBORHOOD_RADIUS_KM,
) -> pd.DataFrame:
    """Features for every cell. In-cell performance is attached for labeling only."""
    assert radius_km > 0, f"neighborhood radius must be positive, got {radius_km}"
    assert grid[CELL_ID_COLUMN].is_unique, "market features repeat a cell"
    assert len(listings) > 0, "there are no listings to describe the surrounding market"
    assert len(places) > 0, "there are no listed places to measure distance against"
    assert {LONGITUDE_COLUMN, LATITUDE_COLUMN, "ttm_revenue", "ttm_occupancy"} <= set(listings.columns)

    cell_of_listing = _cell_of_each_listing(grid, listings)
    distances = haversine_km(
        grid[LONGITUDE_COLUMN].to_numpy()[:, None],
        grid[LATITUDE_COLUMN].to_numpy()[:, None],
        listings[LONGITUDE_COLUMN].to_numpy()[None, :],
        listings[LATITUDE_COLUMN].to_numpy()[None, :],
    )
    cell_ids = grid[CELL_ID_COLUMN].astype(str).to_numpy()
    inside = cell_of_listing[None, :] == cell_ids[:, None]
    surrounding = (distances <= radius_km) & ~inside

    revenue = listings["ttm_revenue"].to_numpy(dtype=float)
    occupancy = listings["ttm_occupancy"].to_numpy(dtype=float)
    surrounding_counts = surrounding.sum(axis=1).astype(int)
    in_cell_counts = inside.sum(axis=1).astype(int)

    place_distances = haversine_km(
        grid[LONGITUDE_COLUMN].to_numpy()[:, None],
        grid[LATITUDE_COLUMN].to_numpy()[:, None],
        places[LONGITUDE_COLUMN].to_numpy()[None, :],
        places[LATITUDE_COLUMN].to_numpy()[None, :],
    )
    nearest = place_distances.argmin(axis=1)
    rows = np.arange(len(grid))

    frame = pd.DataFrame({CELL_ID_COLUMN: cell_ids})
    frame[SURROUNDING_REVENUE] = _masked_mean(surrounding, revenue, surrounding_counts)
    frame[SURROUNDING_OCCUPANCY] = _masked_mean(surrounding, occupancy, surrounding_counts)
    frame[SURROUNDING_MEDIAN_REVENUE] = _masked_median(surrounding, revenue, surrounding_counts)
    frame[SURROUNDING_MEDIAN_OCCUPANCY] = _masked_median(surrounding, occupancy, surrounding_counts)
    frame[SURROUNDING_LISTINGS] = surrounding_counts
    frame[LISTED_PLACES_WITHIN_RADIUS] = (place_distances <= radius_km).sum(axis=1).astype(int)
    frame[DISTANCE_COLUMN] = place_distances[rows, nearest]
    frame[NEAREST_PLACE_COLUMN] = places["name"].to_numpy()[nearest]
    frame[LISTINGS_IN_CELL] = in_cell_counts
    frame[CELL_REVENUE] = _masked_mean(inside, revenue, in_cell_counts)
    frame[CELL_OCCUPANCY] = _masked_mean(inside, occupancy, in_cell_counts)
    frame[MUNICIPALITY_COLUMN] = _group_labels(grid, cell_of_listing, listings)
    empty_market = frame[SURROUNDING_LISTINGS] == 0
    assert frame.loc[empty_market, SURROUNDING_REVENUE].isna().all(), (
        "a cell with no surrounding listings was given a revenue of zero"
    )
    assert frame.loc[empty_market, SURROUNDING_OCCUPANCY].isna().all(), (
        "a cell with no surrounding listings was given an occupancy of zero"
    )
    assert frame.loc[empty_market, SURROUNDING_MEDIAN_REVENUE].isna().all(), (
        "a cell with no surrounding listings was given a median revenue of zero"
    )
    assert frame.loc[empty_market, SURROUNDING_MEDIAN_OCCUPANCY].isna().all(), (
        "a cell with no surrounding listings was given a median occupancy of zero"
    )
    assert set(CLASSIFIER_FEATURES).isdisjoint(LABEL_COLUMNS), (
        "a label column was listed as a classifier feature"
    )
    return frame


def label_performance(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Label cells that contain listings. Empty cells stay unlabeled."""
    labeled = features.loc[features[LISTINGS_IN_CELL] >= config.MIN_LISTINGS_FOR_LABEL].copy()
    assert len(labeled) > 0, "no cell contains enough listings to define historical performance"
    revenue_norm = _min_max(labeled[CELL_REVENUE])
    occupancy_norm = _min_max(labeled[CELL_OCCUPANCY])
    labeled[PERFORMANCE_SCORE] = (
        config.PERFORMANCE_REVENUE_WEIGHT * revenue_norm
        + config.PERFORMANCE_OCCUPANCY_WEIGHT * occupancy_norm
    )
    low_cut = float(np.percentile(labeled[PERFORMANCE_SCORE], config.PERFORMANCE_LOW_PERCENTILE))
    high_cut = float(np.percentile(labeled[PERFORMANCE_SCORE], config.PERFORMANCE_HIGH_PERCENTILE))
    assert low_cut < high_cut, (
        f"the performance percentiles are not separated ({low_cut} and {high_cut})"
    )
    classes = np.full(len(labeled), 1, dtype=int)
    scores = labeled[PERFORMANCE_SCORE].to_numpy(dtype=float)
    classes[scores <= low_cut] = 0
    classes[scores >= high_cut] = 2
    labeled[PERFORMANCE_CLASS] = classes
    counts = {label: int((classes == index).sum()) for index, label in enumerate(config.PERFORMANCE_CLASS_LABELS)}
    assert all(count > 0 for count in counts.values()), (
        f"a performance class is empty: {counts}"
    )
    assert set(classes.tolist()) == {0, 1, 2}, "a labeled cell was left without a performance class"
    report = {
        "normalization": (
            "min-max within labeled cells only: (value - minimum) / (maximum - minimum). "
            "Revenue and occupancy are both oriented so a larger value is higher performance."
        ),
        "performance_formula": (
            f"{config.PERFORMANCE_REVENUE_WEIGHT} * normalized cell mean revenue + "
            f"{config.PERFORMANCE_OCCUPANCY_WEIGHT} * normalized cell mean occupancy"
        ),
        "low_percentile": int(config.PERFORMANCE_LOW_PERCENTILE),
        "high_percentile": int(config.PERFORMANCE_HIGH_PERCENTILE),
        "low_cutoff": low_cut,
        "high_cutoff": high_cut,
        "rule": (
            "Low when the historical performance score is at or below the "
            f"{config.PERFORMANCE_LOW_PERCENTILE}th percentile; High when it is at or above the "
            f"{config.PERFORMANCE_HIGH_PERCENTILE}th percentile; Moderate strictly between them. "
            "Cells with no listing are not labeled."
        ),
        "min_listings_for_label": int(config.MIN_LISTINGS_FOR_LABEL),
        "labeled_cells": int(len(labeled)),
        "class_counts": counts,
        "class_percentages": {
            label: float(count / len(labeled)) for label, count in counts.items()
        },
        "cells_without_a_label": int((features[LISTINGS_IN_CELL] < config.MIN_LISTINGS_FOR_LABEL).sum()),
        "neighborhood_radius_km": float(config.NEIGHBORHOOD_RADIUS_KM),
        "places_description": PLACES_DESCRIPTION,
        "missing_market": (
            "surrounding mean revenue and occupancy stay missing when the surrounding "
            "listing count is zero. The forest accepts those missing values. They are not filled with zero. "
            "A true zero mean can occur only when surrounding listings exist and their mean is zero."
        ),
        "features": list(CLASSIFIER_FEATURES),
        "market_features": list(MARKET_FEATURES),
        "competition_features": list(COMPETITION_FEATURES),
        "tourist_features": list(TOURIST_FEATURES),
        "median_market_features": list(MEDIAN_MARKET_FEATURES),
        "excluded_from_features": [CELL_REVENUE, CELL_OCCUPANCY, LISTINGS_IN_CELL, PERFORMANCE_SCORE],
    }
    return labeled.reset_index(drop=True), report


def train_location_classifier(labeled: pd.DataFrame) -> dict[str, object]:
    """Score every municipality fold, then refit the same settings on every labeled cell."""
    features = labeled[list(CLASSIFIER_FEATURES)]
    target = labeled[PERFORMANCE_CLASS].astype(int)
    groups = labeled[MUNICIPALITY_COLUMN].astype(str)
    folds, scheme = _group_splits(features, target, groups)
    _assert_folds_hold_out_municipalities(groups, folds)
    if scheme.startswith(GROUP_SCHEME):
        covered = np.concatenate([test_index for _, test_index in folds])
        assert len(covered) == len(set(covered.tolist())) == len(features), (
            "a labeled cell is missing from the municipality folds or appears in more than one test fold"
        )

    fold_metrics = []
    majority_folds = []
    stratified_folds = []
    importance_folds = []
    impurity_folds = []
    evaluation_model = None
    for train_index, test_index in folds:
        features_train, features_test = features.iloc[train_index], features.iloc[test_index]
        target_train, target_test = target.iloc[train_index], target.iloc[test_index]
        evaluation_model = _forest()
        evaluation_model.fit(features_train, target_train)
        fold_metrics.append(_score_fold(evaluation_model, features_test, target_test))
        majority_folds.append(_baseline_fold("most_frequent", features_train, target_train, features_test, target_test))
        stratified_folds.append(_baseline_fold("stratified", features_train, target_train, features_test, target_test))
        importance_folds.append(_importance_table(evaluation_model, features_test, target_test))
        impurity_folds.append(
            {
                name: float(value)
                for name, value in zip(CLASSIFIER_FEATURES, evaluation_model.feature_importances_, strict=True)
            }
        )

    test_metrics = _aggregate_fold_metrics(fold_metrics)
    majority_metrics = _aggregate_fold_metrics(majority_folds)
    stratified_metrics = _aggregate_fold_metrics(stratified_folds)
    map_model = _forest()
    map_model.fit(features, target)
    assert list(map_model.classes_) == [0, 1, 2], "the performance model did not learn three classes"
    first_train, first_test = folds[0]
    return {
        "model": map_model,
        "evaluation_model": evaluation_model,
        "params": dict(config.CLASSIFIER_PARAMS),
        "class_weight": "balanced",
        "scheme": scheme,
        "features": list(CLASSIFIER_FEATURES),
        "market_features": list(MARKET_FEATURES),
        "competition_features": list(COMPETITION_FEATURES),
        "tourist_features": list(TOURIST_FEATURES),
        "test_metrics": test_metrics,
        "baseline_metrics": majority_metrics,
        "stratified_baseline_metrics": stratified_metrics,
        "macro_f1_above_majority_baseline": bool(
            test_metrics["macro_f1"] > majority_metrics["macro_f1"]
        ),
        "importance": _aggregate_importance(importance_folds),
        "importance_note": IMPORTANCE_NOTE,
        "impurity_importance": _aggregate_named_values(impurity_folds),
        "train_cells": int(len(first_train)),
        "test_cells": int(len(first_test)),
        "train_groups": int(groups.iloc[first_train].nunique()),
        "test_groups": int(groups.iloc[first_test].nunique()),
        "train_class_counts": _count_classes(target.iloc[first_train]),
        "test_class_counts": _count_classes(target.iloc[first_test]),
        "fold_sizes": [
            {"train_cells": int(len(train_index)), "test_cells": int(len(test_index))}
            for train_index, test_index in folds
        ],
        "cells_outside_training_feature_range": _outside_training_range(features, features.iloc[first_train]),
    }


def score_cells(model, features: pd.DataFrame) -> pd.DataFrame:
    probabilities = _class_probabilities(model, features[list(CLASSIFIER_FEATURES)])
    assert np.isfinite(probabilities).all(), "the performance model produced a missing probability"
    assert np.allclose(probabilities.sum(axis=1), 1.0), "class probabilities do not sum to 1"
    predicted = _class_from_probability(probabilities)
    scored = features[[CELL_ID_COLUMN, *CLASSIFIER_FEATURES, NEAREST_PLACE_COLUMN, LISTINGS_IN_CELL]].copy()
    for index, column in enumerate(PROBABILITY_COLUMNS):
        scored[column] = probabilities[:, index]
    scored[PREDICTED_CLASS] = [config.PERFORMANCE_CLASS_LABELS[int(label)] for label in predicted]
    return scored


def _forest(**overrides) -> RandomForestClassifier:
    params = dict(config.CLASSIFIER_PARAMS)
    params.update(overrides)
    return RandomForestClassifier(
        random_state=config.RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
        **params,
    )


def _score_fold(model, features: pd.DataFrame, target: pd.Series) -> dict[str, object]:
    probability = _class_probabilities(model, features)
    return _classification_metrics(target, probability, _class_from_probability(probability))


def _baseline_fold(strategy: str, features_train, target_train, features_test, target_test) -> dict[str, object]:
    baseline = DummyClassifier(strategy=strategy, random_state=config.RANDOM_STATE)
    baseline.fit(features_train, target_train)
    probability = _class_probabilities(baseline, features_test)
    predicted = (
        baseline.predict(features_test)
        if strategy == "stratified"
        else _class_from_probability(probability)
    )
    return _classification_metrics(target_test, probability, predicted)


def _mean_std(values: list[float]) -> dict[str, float | int | None]:
    usable = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not usable:
        return {"mean": None, "std": None, "n_folds": 0}
    std = float(np.std(usable, ddof=1)) if len(usable) > 1 else 0.0
    return {"mean": float(np.mean(usable)), "std": std, "n_folds": len(usable)}


def _aggregate_fold_metrics(fold_metrics: list[dict[str, object]]) -> dict[str, object]:
    keys = (
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "roc_auc_ovr_macro",
    )
    combined: dict[str, object] = {}
    cv: dict[str, object] = {}
    for key in keys:
        stats = _mean_std([fold.get(key) for fold in fold_metrics])
        combined[key] = stats["mean"]
        cv[key] = stats
    combined["cv"] = cv
    combined["n_folds"] = len(fold_metrics)
    combined["n"] = int(sum(int(fold["n"]) for fold in fold_metrics))
    combined["folds"] = [
        {key: fold.get(key) for key in (*keys, "n", "class_counts")}
        for fold in fold_metrics
    ]
    matrix = np.sum([np.asarray(fold["confusion_matrix"], dtype=float) for fold in fold_metrics], axis=0)
    combined["confusion_matrix"] = matrix.astype(int).tolist()
    return combined


def _aggregate_importance(tables: list[pd.DataFrame]) -> list[dict[str, object]]:
    features = list(tables[0]["feature"])
    rows = []
    for feature in features:
        values = [
            float(table.loc[table["feature"] == feature, "importance_mean"].iloc[0])
            for table in tables
        ]
        stats = _mean_std(values)
        rows.append(
            {
                "feature": feature,
                "importance_mean": stats["mean"],
                "importance_std": stats["std"],
                "fold_importances": values,
            }
        )
    rows.sort(key=lambda row: row["importance_mean"], reverse=True)
    return rows


def _aggregate_named_values(folds: list[dict[str, float]]) -> list[dict[str, object]]:
    rows = []
    for feature in folds[0]:
        stats = _mean_std([fold[feature] for fold in folds])
        rows.append({"feature": feature, "importance": stats["mean"], "importance_std": stats["std"]})
    rows.sort(key=lambda row: row["importance"], reverse=True)
    return rows


def _assert_folds_hold_out_municipalities(groups: pd.Series, folds: list[tuple[np.ndarray, np.ndarray]]) -> None:
    for train_index, test_index in folds:
        assert set(groups.iloc[train_index]).isdisjoint(set(groups.iloc[test_index])), (
            "a municipality appears in both the training and test side of a fold"
        )


def _min_max(values: pd.Series) -> pd.Series:
    number = values.astype(float)
    low = float(number.min())
    high = float(number.max())
    assert high > low, f"{values.name} has no range to normalize"
    return (number - low) / (high - low)


def _class_probabilities(model, features: pd.DataFrame) -> np.ndarray:
    present = [int(label) for label in model.classes_]
    raw = model.predict_proba(features)
    aligned = np.zeros((len(features), 3), dtype=float)
    for column, label in enumerate(present):
        aligned[:, label] = raw[:, column]
    return aligned


def _class_from_probability(probabilities: np.ndarray) -> np.ndarray:
    return probabilities.argmax(axis=1).astype(int)


def _count_classes(target: pd.Series) -> dict[str, int]:
    values = target.to_numpy(dtype=int)
    return {
        label: int((values == index).sum())
        for index, label in enumerate(config.PERFORMANCE_CLASS_LABELS)
    }


def _outside_training_range(all_features: pd.DataFrame, train_features: pd.DataFrame) -> int:
    outside = np.zeros(len(all_features), dtype=bool)
    for column in CLASSIFIER_FEATURES:
        observed = train_features[column].dropna()
        if len(observed) == 0:
            continue
        values = all_features[column]
        outside |= values.notna() & ((values < float(observed.min())) | (values > float(observed.max())))
    return int(outside.sum())


def _classification_metrics(observed, probabilities, predicted) -> dict[str, object]:
    observed_array = np.asarray(observed, dtype=int)
    predicted_array = np.asarray(predicted, dtype=int)
    labels = [0, 1, 2]
    names = list(config.PERFORMANCE_CLASS_LABELS)
    precision, recall, f1, _ = precision_recall_fscore_support(
        observed_array, predicted_array, labels=labels, zero_division=0
    )
    roc_auc = None
    per_class_auc: dict[str, float | None] = {}
    if len(set(observed_array)) > 1 and probabilities.shape[1] == 3:
        try:
            roc_auc = float(
                roc_auc_score(observed_array, probabilities, multi_class="ovr", average="macro")
            )
        except ValueError:
            roc_auc = None
        for index, name in enumerate(names):
            binary = (observed_array == index).astype(int)
            if binary.min() == binary.max():
                per_class_auc[name] = None
            else:
                per_class_auc[name] = float(roc_auc_score(binary, probabilities[:, index]))
    return {
        "accuracy": float(accuracy_score(observed_array, predicted_array)),
        "balanced_accuracy": float(balanced_accuracy_score(observed_array, predicted_array)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1_score(observed_array, predicted_array, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(observed_array, predicted_array, average="weighted", zero_division=0)),
        "roc_auc_ovr_macro": roc_auc,
        "roc_auc_ovr_per_class": per_class_auc,
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, name in enumerate(names)
        },
        "confusion_matrix": confusion_matrix(observed_array, predicted_array, labels=labels).tolist(),
        "class_counts": _count_classes(pd.Series(observed_array)),
        "n": int(len(observed_array)),
    }


def _importance_table(model, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    if target.nunique() < 2:
        return pd.DataFrame({"feature": list(features.columns), "importance_mean": 0.0, "importance_std": 0.0})
    result = permutation_importance(
        model,
        features,
        target,
        n_repeats=config.PERMUTATION_REPEATS,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        scoring="f1_macro",
    )
    table = pd.DataFrame(
        {
            "feature": features.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def _cell_of_each_listing(grid: pd.DataFrame, listings: pd.DataFrame) -> np.ndarray:
    import geopandas as gpd

    from str_suitability.config import GEOGRAPHIC_CRS

    if isinstance(grid, gpd.GeoDataFrame):
        cells = grid
    else:
        cells = gpd.GeoDataFrame(grid, geometry="geometry", crs=GEOGRAPHIC_CRS)
    points = gpd.GeoDataFrame(
        {CELL_ID_COLUMN: np.arange(len(listings))},
        geometry=gpd.points_from_xy(listings[LONGITUDE_COLUMN], listings[LATITUDE_COLUMN]),
        crs=GEOGRAPHIC_CRS,
    ).to_crs(cells.crs)
    joined = gpd.sjoin(
        points,
        cells[[CELL_ID_COLUMN, "geometry"]],
        predicate="within",
        how="left",
        lsuffix="listing",
        rsuffix="cell",
    )
    assert joined[f"{CELL_ID_COLUMN}_listing"].is_unique, "a listing fell inside more than one cell"
    joined = joined.sort_values(f"{CELL_ID_COLUMN}_listing")
    return joined[f"{CELL_ID_COLUMN}_cell"].fillna("").astype(str).to_numpy()


def _group_labels(grid: pd.DataFrame, cell_of_listing: np.ndarray, listings: pd.DataFrame) -> pd.Series:
    if MUNICIPALITY_COLUMN in grid.columns:
        groups = grid[MUNICIPALITY_COLUMN].fillna("unassigned").astype(str)
        return pd.Series(groups.to_numpy(), index=grid.index)
    listing_groups = (
        listings[MUNICIPALITY_COLUMN].astype(str)
        if MUNICIPALITY_COLUMN in listings.columns
        else pd.Series("unassigned", index=listings.index)
    )
    by_cell: dict[str, str] = {}
    for cell_id, municipality in zip(cell_of_listing, listing_groups, strict=True):
        if cell_id and cell_id not in by_cell:
            by_cell[cell_id] = municipality
    return pd.Series(
        [by_cell.get(str(cell_id), "unassigned") for cell_id in grid[CELL_ID_COLUMN]],
        index=grid.index,
    )


def _masked_mean(mask: np.ndarray, values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    totals = mask @ values
    means = np.full(len(counts), np.nan, dtype=float)
    covered = counts > 0
    means[covered] = totals[covered] / counts[covered]
    return means


def _masked_median(mask: np.ndarray, values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    medians = np.full(len(counts), np.nan, dtype=float)
    for index in np.flatnonzero(counts > 0):
        medians[index] = float(np.median(values[mask[index]]))
    return medians


def _usable_splits(target: pd.Series, groups: pd.Series) -> int:
    frame = pd.DataFrame({"y": target.to_numpy(), "g": groups.astype(str).to_numpy()})
    groups_per_class = frame.groupby("y")["g"].nunique()
    return int(min(config.CV_FOLDS, int(groups_per_class.min()), int(groups.nunique())))


def _group_splits(
    features: pd.DataFrame, target: pd.Series, groups: pd.Series
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """Every stratified municipality fold. A grouped shuffle is used only when a class has one municipality."""
    splits = _usable_splits(target, groups)
    if splits >= 2:
        splitter = StratifiedGroupKFold(
            n_splits=splits, shuffle=True, random_state=config.RANDOM_STATE
        )
        folds = [(train_index, test_index) for train_index, test_index in splitter.split(features, target, groups)]
        return folds, f"{GROUP_SCHEME}_{splits}"

    splitter = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    folds = [(train_index, test_index) for train_index, test_index in splitter.split(features, target, groups)]
    return folds, FALLBACK_SCHEME


def _holdout_split(
    features: pd.DataFrame, target: pd.Series, groups: pd.Series
) -> tuple[np.ndarray, np.ndarray, str]:
    """First municipality fold. The classifier itself evaluates every fold."""
    folds, scheme = _group_splits(features, target, groups)
    train_index, test_index = folds[0]
    return train_index, test_index, scheme
