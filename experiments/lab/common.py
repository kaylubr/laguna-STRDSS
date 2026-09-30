"""Shared, decoupled utilities for the experiment lab.

Validation splitting and metrics are reimplemented here rather than imported from
str_suitability.modeling.location_classifier's private helpers. A change to this file
cannot reach the production module, and a change to the production module cannot
silently change what an experiment measures.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.2
CLASS_LABELS = ("Low", "Moderate", "High")
RF_PARAMS = dict(
    n_estimators=200, max_depth=10, max_features="sqrt", min_samples_leaf=1, min_samples_split=2
)

# ---------------------------------------------------------------------------
# Validation splitters
# ---------------------------------------------------------------------------


def usable_splits(target: pd.Series, groups: pd.Series, folds: int = CV_FOLDS) -> int:
    frame = pd.DataFrame({"y": target.to_numpy(), "g": groups.astype(str).to_numpy()})
    groups_per_class = frame.groupby("y")["g"].nunique()
    return int(min(folds, int(groups_per_class.min()), int(groups.nunique())))


def grouped_kfold_splits(
    features: pd.DataFrame, target: pd.Series, groups: pd.Series, folds: int = CV_FOLDS
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """Municipality-grouped folds, matching the splitting rule used by the current
    thesis model (StratifiedGroupKFold, shuffled, random_state 42), so experiment
    results are comparable to the baseline on the same terms. Falls back to one
    grouped split if a class has fewer than two municipalities."""
    usable = usable_splits(target, groups, folds)
    if usable >= 2:
        splitter = StratifiedGroupKFold(n_splits=usable, shuffle=True, random_state=RANDOM_STATE)
        splits = [(tr, te) for tr, te in splitter.split(features, target, groups)]
        return splits, f"stratified_group_kfold_{usable}"
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    splits = [(tr, te) for tr, te in splitter.split(features, target, groups)]
    return splits, "group_shuffle"


def leave_one_municipality_out(groups: pd.Series) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """One fold per municipality: train on every other town, test on that one town.
    A degenerate test fold (few rows, one class) still gets a fold; metrics that need
    two classes come back as None for that town, and that is reported, not hidden."""
    index = np.arange(len(groups))
    folds = []
    for municipality in sorted(groups.unique()):
        mask = (groups == municipality).to_numpy()
        test_idx = index[mask]
        train_idx = index[~mask]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        folds.append((train_idx, test_idx, str(municipality)))
    return folds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def mean_std(values: list) -> dict:
    usable = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not usable:
        return {"mean": None, "std": None, "n": 0}
    std = float(np.std(usable, ddof=1)) if len(usable) > 1 else 0.0
    return {"mean": float(np.mean(usable)), "std": std, "n": len(usable)}


def classification_metrics(observed, probabilities, predicted) -> dict:
    """The same metric set the baseline model reports, computed independently."""
    observed_arr = np.asarray(observed, dtype=int)
    predicted_arr = np.asarray(predicted, dtype=int)
    labels = list(range(len(CLASS_LABELS)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        observed_arr, predicted_arr, labels=labels, zero_division=0
    )
    roc_auc = None
    if probabilities is not None and len(set(observed_arr.tolist())) > 1 and probabilities.shape[1] == len(labels):
        try:
            roc_auc = float(
                roc_auc_score(observed_arr, probabilities, multi_class="ovr", average="macro", labels=labels)
            )
        except ValueError:
            roc_auc = None
    return {
        "accuracy": float(accuracy_score(observed_arr, predicted_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(observed_arr, predicted_arr)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1_score(observed_arr, predicted_arr, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(observed_arr, predicted_arr, average="weighted", zero_division=0)),
        "roc_auc_ovr_macro": roc_auc,
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
            for i, name in enumerate(CLASS_LABELS)
        },
        "confusion_matrix": confusion_matrix(observed_arr, predicted_arr, labels=labels).tolist(),
        "n": int(len(observed_arr)),
        "class_counts": {name: int((observed_arr == i).sum()) for i, name in enumerate(CLASS_LABELS)},
    }


def regression_metrics(observed, predicted) -> dict:
    observed_arr = np.asarray(observed, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    return {
        "mae": float(mean_absolute_error(observed_arr, predicted_arr)),
        "rmse": float(np.sqrt(mean_squared_error(observed_arr, predicted_arr))),
        "r2": float(r2_score(observed_arr, predicted_arr)) if len(observed_arr) > 1 else None,
        "n": int(len(observed_arr)),
    }


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------


def class_probabilities(model, features: pd.DataFrame, n_classes: int = 3) -> np.ndarray:
    present = [int(c) for c in model.classes_]
    raw = model.predict_proba(features)
    aligned = np.zeros((len(features), n_classes), dtype=float)
    for column, label in enumerate(present):
        aligned[:, label] = raw[:, column]
    return aligned


def make_model(name: str, **overrides):
    """Returns (estimator, needs_imputation). Random Forest uses this project's
    scikit-learn's native NaN support, matching the current thesis model. The other
    models cannot take NaN, so they get a median imputer in front of them. That is a
    real preprocessing difference between models and is reported as one."""
    if name == "random_forest":
        params = dict(RF_PARAMS)
        params.update(overrides)
        model = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1, **params)
        return model, False
    if name == "logistic_regression":
        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE, **overrides)
        return model, True
    if name == "decision_tree":
        params = dict(max_depth=6)
        params.update(overrides)
        model = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced", **params)
        return model, True
    if name == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=RANDOM_STATE, **overrides)
        return model, True
    if name == "majority_baseline":
        return DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE), False
    if name == "stratified_baseline":
        return DummyClassifier(strategy="stratified", random_state=RANDOM_STATE), False
    raise AssertionError(f"unknown model {name}")


def fit_predict(model, needs_imputation: bool, features_train, target_train, features_test):
    if needs_imputation:
        pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("model", model)])
        pipeline.fit(features_train, target_train)
        proba = class_probabilities(pipeline, features_test)
        predicted = pipeline.predict(features_test)
        return proba, predicted
    model.fit(features_train, target_train)
    proba = class_probabilities(model, features_test)
    predicted = proba.argmax(axis=1)
    return proba, predicted


# ---------------------------------------------------------------------------
# Experiment evaluation
# ---------------------------------------------------------------------------


def evaluate_grouped(
    frame: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    group_col: str,
    model_name: str = "random_forest",
    model_overrides: dict | None = None,
    folds=None,
) -> dict:
    """Municipality-grouped cross-validation for one feature set and one model."""
    features = frame[feature_cols]
    target = frame[target_col].astype(int)
    groups = frame[group_col].astype(str)
    if folds is None:
        folds, scheme = grouped_kfold_splits(features, target, groups)
    else:
        scheme = "reused_folds"
    fold_metrics = []
    for train_idx, test_idx in folds:
        model, needs_imputation = make_model(model_name, **(model_overrides or {}))
        proba, predicted = fit_predict(
            model, needs_imputation, features.iloc[train_idx], target.iloc[train_idx], features.iloc[test_idx]
        )
        fold_metrics.append(classification_metrics(target.iloc[test_idx], proba, predicted))
    return _aggregate(fold_metrics, scheme)


def evaluate_lomo(
    frame: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    group_col: str,
    model_name: str = "random_forest",
    model_overrides: dict | None = None,
) -> dict:
    """One fold per municipality. Every municipality is held out exactly once."""
    features = frame[feature_cols]
    target = frame[target_col].astype(int)
    groups = frame[group_col].astype(str)
    folds = leave_one_municipality_out(groups)
    fold_metrics = []
    per_municipality = []
    for train_idx, test_idx, municipality in folds:
        model, needs_imputation = make_model(model_name, **(model_overrides or {}))
        proba, predicted = fit_predict(
            model, needs_imputation, features.iloc[train_idx], target.iloc[train_idx], features.iloc[test_idx]
        )
        metrics = classification_metrics(target.iloc[test_idx], proba, predicted)
        metrics["municipality"] = municipality
        fold_metrics.append(metrics)
        per_municipality.append(metrics)
    aggregated = _aggregate(fold_metrics, "leave_one_municipality_out")
    aggregated["per_municipality"] = per_municipality
    return aggregated


def _aggregate(fold_metrics: list, scheme: str) -> dict:
    keys = (
        "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
        "macro_f1", "weighted_f1", "roc_auc_ovr_macro",
    )
    combined: dict = {"scheme": scheme, "n_folds": len(fold_metrics)}
    cv: dict = {}
    for key in keys:
        stats = mean_std([fold.get(key) for fold in fold_metrics])
        combined[key] = stats["mean"]
        cv[key] = stats
    combined["cv"] = cv
    combined["n"] = int(sum(int(fold["n"]) for fold in fold_metrics)) if fold_metrics else 0
    combined["folds"] = fold_metrics
    if fold_metrics:
        matrix = np.sum([np.asarray(fold["confusion_matrix"], dtype=float) for fold in fold_metrics], axis=0)
        combined["confusion_matrix"] = matrix.astype(int).tolist()
    return combined


RESULTS_COLUMNS = (
    "experiment_id", "feature_group", "model", "grid_size", "poi_radius", "airbnb_radius",
    "target_definition", "n_features", "n_train", "n_test", "accuracy", "macro_precision",
    "macro_recall", "macro_f1", "weighted_f1", "high_precision", "high_recall",
    "validation_score", "mean_municipality_score", "std_municipality_score", "notes",
)


def high_class_metrics(grouped_metrics: dict) -> tuple[float | None, float | None]:
    per_class = grouped_metrics["folds"]
    precisions = [fold["per_class"]["High"]["precision"] for fold in per_class]
    recalls = [fold["per_class"]["High"]["recall"] for fold in per_class]
    return mean_std(precisions)["mean"], mean_std(recalls)["mean"]


def results_row(
    experiment_id: str,
    feature_group: str,
    model: str,
    grouped_metrics: dict,
    n_features: int,
    lomo_metrics: dict | None = None,
    grid_size: str = "1000m",
    poi_radius: str = "",
    airbnb_radius: str = "",
    target_definition: str = "low25_mid50_high25",
    notes: str = "",
) -> dict:
    high_precision, high_recall = high_class_metrics(grouped_metrics)
    n_test = sum(fold["n"] for fold in grouped_metrics["folds"])
    n_train = grouped_metrics.get("n_train")
    row = {
        "experiment_id": experiment_id,
        "feature_group": feature_group,
        "model": model,
        "grid_size": grid_size,
        "poi_radius": poi_radius,
        "airbnb_radius": airbnb_radius,
        "target_definition": target_definition,
        "n_features": n_features,
        "n_train": n_train if n_train is not None else "",
        "n_test": n_test,
        "accuracy": grouped_metrics.get("accuracy"),
        "macro_precision": grouped_metrics.get("macro_precision"),
        "macro_recall": grouped_metrics.get("macro_recall"),
        "macro_f1": grouped_metrics.get("macro_f1"),
        "weighted_f1": grouped_metrics.get("weighted_f1"),
        "high_precision": high_precision,
        "high_recall": high_recall,
        "validation_score": grouped_metrics.get("macro_f1"),
        "mean_municipality_score": lomo_metrics.get("macro_f1") if lomo_metrics else "",
        "std_municipality_score": lomo_metrics["cv"]["macro_f1"]["std"] if lomo_metrics else "",
        "notes": notes,
    }
    return row


def append_result_rows(results_csv: Path, rows: list) -> None:
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = results_csv.exists()
    with results_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_COLUMNS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_experiment(
    directory: Path,
    config: dict,
    grouped_metrics: dict,
    feature_cols: list,
    notes: str,
    lomo_metrics: dict | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    metrics_out = {"grouped_cv": _strip_folds(grouped_metrics)}
    if lomo_metrics is not None:
        metrics_out["leave_one_municipality_out"] = _strip_folds(lomo_metrics)
        metrics_out["per_municipality"] = [
            {k: v for k, v in fold.items() if k != "confusion_matrix"}
            for fold in lomo_metrics.get("per_municipality", [])
        ]
    (directory / "metrics.json").write_text(json.dumps(metrics_out, indent=2, default=str), encoding="utf-8")
    if "confusion_matrix" in grouped_metrics:
        matrix = grouped_metrics["confusion_matrix"]
        with (directory / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["", *CLASS_LABELS])
            for label, row in zip(CLASS_LABELS, matrix, strict=True):
                writer.writerow([label, *row])
    (directory / "feature_list.json").write_text(json.dumps(list(feature_cols), indent=2), encoding="utf-8")
    (directory / "notes.md").write_text(notes, encoding="utf-8")


def _strip_folds(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k not in {"folds", "per_municipality"}}


def paired_change(candidate_folds: list, baseline_folds: list, keys=None) -> dict:
    keys = keys or ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "roc_auc_ovr_macro")
    out = {}
    for key in keys:
        diffs = []
        for cand, base in zip(candidate_folds, baseline_folds, strict=True):
            if cand.get(key) is None or base.get(key) is None:
                continue
            diffs.append(float(cand[key]) - float(base[key]))
        out[key] = mean_std(diffs)
    return out
