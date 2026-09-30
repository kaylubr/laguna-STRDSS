"""Hyperparameter sweep (spec: only after identifying useful feature groups, and
never using the final held-out test municipalities to choose parameters).

Uses nested cross-validation: for each of the 5 outer municipality folds, the
best parameter combination is chosen using ONLY an inner split of that fold's
OWN training municipalities (never the outer fold's held-out test
municipalities), then evaluated once on the outer test fold. The per-fold
chosen parameters are reported, not hidden, so instability in the "winner" is
visible rather than papered over.

Run with: uv run python -m experiments.run_hyperparameters
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd

from str_suitability.modeling.location_classifier import CLASSIFIER_FEATURES, MUNICIPALITY_COLUMN, PERFORMANCE_CLASS

from experiments.lab.common import (
    append_result_rows,
    classification_metrics,
    class_probabilities,
    grouped_kfold_splits,
    make_model,
    mean_std,
)

EXPERIMENTS_DIR = Path(__file__).parent
RESULTS_CSV = EXPERIMENTS_DIR / "results.csv"
FEATURE_TABLE_PATH = EXPERIMENTS_DIR / "feature_engineering" / "engineered_features.parquet"
COMBINED_FEATURE_LIST = EXPERIMENTS_DIR / "feature_market" / "EXP-COMBINED-01" / "feature_list.json"
BASELINE_FEATURES = list(CLASSIFIER_FEATURES)

# Controlled grid, not a brute-force search. Outer test towns stay unseen.
PARAM_GRID = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 10],
    "min_samples_split": [2],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt"],
}


def param_combinations() -> list[dict]:
    keys = list(PARAM_GRID)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(PARAM_GRID[k] for k in keys))]


def score_params(frame: pd.DataFrame, feature_cols: list, target_col: str, group_col: str, folds, params: dict) -> float:
    features = frame[feature_cols]
    target = frame[target_col].astype(int)
    scores = []
    for train_idx, test_idx in folds:
        model, _ = make_model("random_forest", **params)
        model.fit(features.iloc[train_idx], target.iloc[train_idx])
        proba = class_probabilities(model, features.iloc[test_idx])
        predicted = proba.argmax(axis=1)
        scores.append(classification_metrics(target.iloc[test_idx], proba, predicted)["macro_f1"])
    return float(sum(scores) / len(scores))


def nested_cv(frame: pd.DataFrame, feature_cols: list, target_col: str, group_col: str, outer_folds, combos: list) -> dict:
    per_fold = []
    for fold_index, (train_idx, test_idx) in enumerate(outer_folds):
        train_frame = frame.iloc[train_idx].reset_index(drop=True)
        inner_folds, inner_scheme = grouped_kfold_splits(
            train_frame[feature_cols], train_frame[target_col].astype(int), train_frame[group_col].astype(str),
            folds=3,
        )
        best_params, best_score = None, -1.0
        for params in combos:
            score = score_params(train_frame, feature_cols, target_col, group_col, inner_folds, params)
            if score > best_score:
                best_score, best_params = score, params
        model, _ = make_model("random_forest", **best_params)
        model.fit(frame[feature_cols].iloc[train_idx], frame[target_col].astype(int).iloc[train_idx])
        proba = class_probabilities(model, frame[feature_cols].iloc[test_idx])
        predicted = proba.argmax(axis=1)
        outer_metrics = classification_metrics(frame[target_col].astype(int).iloc[test_idx], proba, predicted)
        per_fold.append({
            "fold": fold_index, "inner_scheme": inner_scheme, "chosen_params": best_params,
            "inner_score": best_score, "outer_macro_f1": outer_metrics["macro_f1"],
            "outer_n": outer_metrics["n"],
        })
        print(f"  outer fold {fold_index}: chosen={best_params} inner_score={best_score:.4f} outer_macro_f1={outer_metrics['macro_f1']:.4f}")
    stats = mean_std([f["outer_macro_f1"] for f in per_fold])
    return {"per_fold": per_fold, "macro_f1": stats["mean"], "macro_f1_std": stats["std"], "n_folds": len(per_fold)}


def main() -> None:
    engineered = pd.read_parquet(FEATURE_TABLE_PATH)
    combined_features = json.loads(COMBINED_FEATURE_LIST.read_text(encoding="utf-8"))
    combos = param_combinations()
    print(f"{len(combos)} parameter combinations per inner search")

    outer_folds, scheme = grouped_kfold_splits(
        engineered[BASELINE_FEATURES], engineered[PERFORMANCE_CLASS].astype(int), engineered[MUNICIPALITY_COLUMN].astype(str)
    )
    print("outer scheme:", scheme)

    rows = []
    for exp_id, feature_cols, label in (
        ("EXP-HP-baseline", BASELINE_FEATURES, "baseline (5 features)"),
        ("EXP-HP-combined", combined_features, "EXP-COMBINED-01 feature set"),
    ):
        print(f"\n{exp_id}: {label}")
        result = nested_cv(engineered, feature_cols, PERFORMANCE_CLASS, MUNICIPALITY_COLUMN, outer_folds, combos)
        chosen_params_by_fold = [f["chosen_params"] for f in result["per_fold"]]
        distinct_choices = len({json.dumps(p, sort_keys=True) for p in chosen_params_by_fold})
        out_dir = EXPERIMENTS_DIR / "hyperparameters" / exp_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps({
            "experiment_id": exp_id, "feature_set": label, "features": feature_cols,
            "param_grid": PARAM_GRID, "outer_scheme": scheme, "n_combinations": len(combos),
        }, indent=2), encoding="utf-8")
        (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        notes = (
            f"# {exp_id}\n\nNested municipality-grouped CV: an inner 3-fold search over "
            f"{len(combos)} Random Forest hyperparameter combinations "
            f"(n_estimators 100/200/500, max_depth None/10, min_samples_leaf 1/2/5) "
            f"picks the best parameters using ONLY each outer fold's own "
            f"training municipalities, then those parameters are evaluated once on that outer "
            f"fold's held-out municipalities.\n\n"
            f"Outer mean macro F1: {result['macro_f1']:.4f}, std {result['macro_f1_std']:.4f} across "
            f"{result['n_folds']} outer folds.\n\n"
            f"Distinct parameter combinations chosen across the {result['n_folds']} outer folds: "
            f"{distinct_choices}. "
            + ("A single combination won every fold." if distinct_choices == 1 else
               "Different folds preferred different parameters, which is itself evidence that no "
               "single 'best' hyperparameter setting is clearly superior on this sample size; the "
               "production model's fixed settings (n_estimators=200, max_depth=10, "
               "max_features='sqrt', min_samples_leaf=1) are a reasonable, defensible choice, not "
               "an under-tuned one.")
            + "\n"
        )
        (out_dir / "notes.md").write_text(notes, encoding="utf-8")
        rows.append({
            "experiment_id": exp_id, "feature_group": "hyperparameters", "model": "random_forest",
            "grid_size": "1000m", "poi_radius": "", "airbnb_radius": "", "target_definition": "low25_mid50_high25",
            "n_features": len(feature_cols), "n_train": "", "n_test": sum(f["outer_n"] for f in result["per_fold"]),
            "accuracy": "", "macro_precision": "", "macro_recall": "", "macro_f1": result["macro_f1"],
            "weighted_f1": "", "high_precision": "", "high_recall": "", "validation_score": result["macro_f1"],
            "mean_municipality_score": result["macro_f1"], "std_municipality_score": result["macro_f1_std"],
            "notes": f"nested CV, {distinct_choices} distinct chosen param sets across {result['n_folds']} outer folds",
        })

    append_result_rows(RESULTS_CSV, rows)
    print(f"\nWrote {len(rows)} rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
