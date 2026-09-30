"""Sensitivity checks around the pre-specified location model.

The map model stays 5 km, mean revenue and occupancy, and all three signals.
These comparisons describe that choice. They do not replace it when a score is higher.
"""

import numpy as np
import pandas as pd

from str_suitability import config
from str_suitability.features.road_distance import ROAD_DISTANCE_KM
from str_suitability.modeling.location_classifier import (
    CELL_ID_COLUMN,
    CLASSIFIER_FEATURES,
    COMPETITION_FEATURES,
    CV_METRIC_KEYS,
    MARKET_FEATURES,
    MEDIAN_MARKET_FEATURES,
    MUNICIPALITY_COLUMN,
    PERFORMANCE_CLASS,
    SURROUNDING_LISTINGS,
    TOURIST_FEATURES,
    _aggregate_fold_metrics,
    _forest,
    _group_splits,
    _mean_std,
    _score_fold,
    market_features,
)

SIGNAL_GROUPS = ("tourist", "market_competition", "all_signals")


def run_location_experiments(
    grid: pd.DataFrame,
    listings: pd.DataFrame,
    places: pd.DataFrame,
    labeled: pd.DataFrame,
    prespecified_metrics: dict[str, object],
) -> dict[str, object]:
    """Compare signals, radii, mean versus median, and leaf sizes on the same municipality folds."""
    folds, scheme = _group_splits(
        labeled[list(CLASSIFIER_FEATURES)],
        labeled[PERFORMANCE_CLASS].astype(int),
        labeled[MUNICIPALITY_COLUMN].astype(str),
    )
    assert prespecified_metrics["n_folds"] == len(folds), (
        "the pre-specified cross-validation and the experiment folds are different lengths"
    )
    by_radius = {
        float(config.NEIGHBORHOOD_RADIUS_KM): labeled,
    }
    for radius_km in config.EXPERIMENT_RADII_KM:
        if float(radius_km) in by_radius:
            continue
        by_radius[float(radius_km)] = _align_labels(
            market_features(grid, listings, places, radius_km=float(radius_km)),
            labeled,
        )

    signal_groups = [
        _cached_or_fit(
            name,
            labeled,
            _columns(name, "mean"),
            folds,
            prespecified_metrics if name == "all_signals" else None,
            radius_km=float(config.NEIGHBORHOOD_RADIUS_KM),
            market_statistic="mean" if name != "tourist" else "not_used",
        )
        for name in SIGNAL_GROUPS
    ]
    radii = [
        _cached_or_fit(
            f"{radius_km:g}km",
            by_radius[float(radius_km)],
            list(CLASSIFIER_FEATURES),
            folds,
            prespecified_metrics if float(radius_km) == float(config.NEIGHBORHOOD_RADIUS_KM) else None,
            radius_km=float(radius_km),
            market_statistic="mean",
            labeled_cells_with_no_surrounding_listings=int(
                (by_radius[float(radius_km)][SURROUNDING_LISTINGS] == 0).sum()
            ),
        )
        for radius_km in config.EXPERIMENT_RADII_KM
    ]
    market_statistics = [
        _cached_or_fit(
            statistic,
            labeled,
            _columns("all_signals", statistic),
            folds,
            prespecified_metrics if statistic == "mean" else None,
            radius_km=float(config.NEIGHBORHOOD_RADIUS_KM),
            market_statistic=statistic,
        )
        for statistic in ("mean", "median")
    ]
    return {
        "prespecified_model": {
            "radius_km": float(config.NEIGHBORHOOD_RADIUS_KM),
            "market_statistic": "mean",
            "signals": "all_signals",
            "features": list(CLASSIFIER_FEATURES),
            "min_samples_leaf": int(config.CLASSIFIER_PARAMS["min_samples_leaf"]),
            "replaced_by_experiments": False,
            "reason": (
                "The final model was specified before these comparisons. "
                "A higher experiment score does not replace it."
            ),
        },
        "validation": scheme,
        "n_folds": len(folds),
        "signal_groups": signal_groups,
        "radii": radii,
        "market_statistics": market_statistics,
        "leaf_sizes": _leaf_comparison(labeled, folds),
        "tourist_minus_market_competition": _paired_difference(
            prespecified_metrics["folds"],
            next(row["folds"] for row in signal_groups if row["name"] == "market_competition"),
        ),
        "reading": (
            "Tourist features add predictive information on this check when the all-signal "
            "model scores higher than market plus competition on the same municipality folds. "
            "The difference is the paired fold-by-fold change. It is not a causal effect."
        ),
    }


def _columns(signals: str, statistic: str) -> list[str]:
    market = {"mean": MARKET_FEATURES, "median": MEDIAN_MARKET_FEATURES}[statistic]
    if signals == "tourist":
        return list(TOURIST_FEATURES)
    if signals == "market_competition":
        return list(market + COMPETITION_FEATURES)
    if signals == "all_signals":
        return list(market + COMPETITION_FEATURES + TOURIST_FEATURES)
    raise AssertionError(f"unknown signal group {signals}")


def _align_labels(features: pd.DataFrame, labeled: pd.DataFrame) -> pd.DataFrame:
    order = labeled[CELL_ID_COLUMN].tolist()
    frame = features.set_index(CELL_ID_COLUMN).loc[order].reset_index()
    frame[PERFORMANCE_CLASS] = labeled[PERFORMANCE_CLASS].to_numpy()
    assert frame[CELL_ID_COLUMN].tolist() == order, "a radius feature table changed the labeled-cell order"
    return frame


def _cached_or_fit(name, frame, columns, folds, cached, **meta) -> dict[str, object]:
    if cached is None:
        cached = _evaluate(frame, columns, folds)
    else:
        assert list(columns) == list(CLASSIFIER_FEATURES) or meta.get("market_statistic") == "mean", (
            "a cached result was reused for a different feature set"
        )
    return {
        "name": name,
        "features": list(columns),
        "metrics": {key: cached["cv"][key] for key in CV_METRIC_KEYS},
        "folds": cached["folds"],
        **meta,
    }


def _evaluate(frame: pd.DataFrame, columns: list[str], folds, **overrides) -> dict[str, object]:
    target = frame[PERFORMANCE_CLASS].astype(int)
    fold_metrics = []
    for train_index, test_index in folds:
        model = _forest(**overrides)
        model.fit(frame.iloc[train_index][columns], target.iloc[train_index])
        fold_metrics.append(_score_fold(model, frame.iloc[test_index][columns], target.iloc[test_index]))
    return _aggregate_fold_metrics(fold_metrics)


def _leaf_comparison(frame: pd.DataFrame, folds) -> dict[str, object]:
    """Choose leaf sizes on inner municipality folds. Outer test municipalities are not used."""
    columns = list(CLASSIFIER_FEATURES)
    collected = {int(leaf): [] for leaf in config.EXPERIMENT_LEAF_SIZES}
    selections = []
    skipped = 0
    for train_index, _test_index in folds:
        inner = frame.iloc[train_index].reset_index(drop=True)
        inner_folds, scheme = _group_splits(
            inner[columns],
            inner[PERFORMANCE_CLASS].astype(int),
            inner[MUNICIPALITY_COLUMN].astype(str),
        )
        if not scheme.startswith("stratified_group_kfold"):
            skipped += 1
            continue
        scores = {}
        for leaf in config.EXPERIMENT_LEAF_SIZES:
            fold_scores = []
            for inner_train, inner_test in inner_folds:
                model = _forest(min_samples_leaf=int(leaf))
                model.fit(
                    inner.iloc[inner_train][columns],
                    inner.iloc[inner_train][PERFORMANCE_CLASS].astype(int),
                )
                fold_scores.append(
                    _score_fold(
                        model,
                        inner.iloc[inner_test][columns],
                        inner.iloc[inner_test][PERFORMANCE_CLASS].astype(int),
                    )["macro_f1"]
                )
            scores[int(leaf)] = float(np.mean(fold_scores))
            collected[int(leaf)].append(scores[int(leaf)])
        selections.append(max(scores, key=scores.get))

    leaves = []
    for leaf, values in collected.items():
        stats = _mean_std(values)
        leaves.append(
            {
                "min_samples_leaf": leaf,
                "inner_macro_f1_mean": stats["mean"],
                "inner_macro_f1_std": stats["std"],
                "outer_folds_compared": stats["n_folds"],
            }
        )
    counts: dict[str, int] = {}
    for leaf in selections:
        counts[str(leaf)] = counts.get(str(leaf), 0) + 1
    return {
        "selection": "inner stratified municipality folds on the outer training municipalities only",
        "outer_test_used_for_selection": False,
        "folds_skipped": skipped,
        "selected_leaf_counts": counts,
        "leaves": leaves,
        "final_model_min_samples_leaf": int(config.CLASSIFIER_PARAMS["min_samples_leaf"]),
        "final_model_changed": False,
    }


def _paired_difference(left_folds, right_folds) -> dict[str, object]:
    differences = {}
    for key in CV_METRIC_KEYS:
        paired = []
        for left, right in zip(left_folds, right_folds, strict=True):
            if left[key] is None or right[key] is None:
                continue
            paired.append(float(left[key]) - float(right[key]))
        differences[key] = _mean_std(paired)
    return {
        "comparison": "all signals minus market and competition, same municipality folds",
        "difference": differences,
    }


def compare_road_accessibility(labeled: pd.DataFrame) -> dict[str, object]:
    """Score the current five features against those features plus road distance.

    The comparison uses the same municipality folds and the same forest settings.
    It does not replace the five-feature map model.
    """
    assert ROAD_DISTANCE_KM in labeled.columns, "labeled cells have no road distance"
    assert set(CLASSIFIER_FEATURES) <= set(labeled.columns), "a current model feature is missing"
    base_columns = list(CLASSIFIER_FEATURES)
    road_columns = [*base_columns, ROAD_DISTANCE_KM]
    folds, scheme = _group_splits(
        labeled[base_columns],
        labeled[PERFORMANCE_CLASS].astype(int),
        labeled[MUNICIPALITY_COLUMN].astype(str),
    )
    model_a = _evaluate(labeled, base_columns, folds)
    model_b = _evaluate(labeled, road_columns, folds)
    return {
        "final_model_changed": False,
        "prespecified_features": base_columns,
        "candidate_features": road_columns,
        "validation": scheme,
        "n_folds": len(folds),
        "model_a_five_features": _compact(model_a),
        "model_b_with_road_distance": _compact(model_b),
        "model_b_minus_model_a": _paired_change(model_b["folds"], model_a["folds"]),
        "reading": _road_reading(model_b["folds"], model_a["folds"]),
    }


def _compact(metrics: dict[str, object]) -> dict[str, object]:
    return {
        "metrics": {key: metrics["cv"][key] for key in CV_METRIC_KEYS},
        "folds": metrics["folds"],
    }


def _paired_change(candidate_folds, current_folds) -> dict[str, object]:
    differences = {}
    for key in CV_METRIC_KEYS:
        paired = []
        for candidate, current in zip(candidate_folds, current_folds, strict=True):
            if candidate[key] is None or current[key] is None:
                continue
            paired.append(float(candidate[key]) - float(current[key]))
        differences[key] = _mean_std(paired)
    return differences


def _road_reading(candidate_folds, current_folds) -> str:
    """Require a gain on every held-out fold before calling the addition an improvement."""
    macro_gains = []
    auc_gains = []
    for candidate, current in zip(candidate_folds, current_folds, strict=True):
        if candidate["macro_f1"] is None or current["macro_f1"] is None:
            continue
        if candidate["roc_auc_ovr_macro"] is None or current["roc_auc_ovr_macro"] is None:
            continue
        macro_gains.append(float(candidate["macro_f1"]) - float(current["macro_f1"]))
        auc_gains.append(float(candidate["roc_auc_ovr_macro"]) - float(current["roc_auc_ovr_macro"]))
    consistent = bool(macro_gains) and all(gain > 0 for gain in macro_gains) and all(gain > 0 for gain in auc_gains)
    if consistent:
        return (
            "Adding distance to the nearest mapped road was associated with improved "
            "out-of-sample classification performance under the municipality-grouped validation. "
            "This is not evidence that road proximity causes Airbnb performance."
        )
    return (
        "Adding distance to the nearest mapped road did not improve the model's held-out "
        "classification performance under the tested configuration. "
        "The average score was higher, but the gain was not present on every municipality fold. "
        "The five-feature model was left unchanged."
    )
