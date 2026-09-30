"""Target-definition alternatives (spec group N).

Every alternative reuses the SAME underlying historical_performance_score the
production model computes; only the percentile cutoffs (or a collapse to two
classes) change. This is an experiment only; it does not change the thesis's
official 25/75 target, and it does not touch label_performance() or config.py.

Run with: uv run python -m experiments.run_target_definition
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from pathlib import Path

from str_suitability.modeling.location_classifier import (
    CLASSIFIER_FEATURES,
    MUNICIPALITY_COLUMN,
    PERFORMANCE_SCORE,
    label_performance,
    market_features,
)

from experiments.lab import data as lab_data
from experiments.lab.common import (
    RANDOM_STATE,
    append_result_rows,
    evaluate_grouped,
    evaluate_lomo,
    grouped_kfold_splits,
    leave_one_municipality_out,
    mean_std,
    results_row,
    save_experiment,
)
from experiments.lab.targets import alt_three_class_target, binary_high_target

EXPERIMENTS_DIR = Path(__file__).parent
RESULTS_CSV = EXPERIMENTS_DIR / "results.csv"
BASELINE_FEATURES = list(CLASSIFIER_FEATURES)


def _binary_fold_metrics(observed, proba_high, predicted) -> dict:
    observed_arr = np.asarray(observed, dtype=int)
    predicted_arr = np.asarray(predicted, dtype=int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        observed_arr, predicted_arr, labels=[0, 1], zero_division=0
    )
    roc_auc = None
    if len(set(observed_arr.tolist())) > 1:
        try:
            roc_auc = float(roc_auc_score(observed_arr, proba_high))
        except ValueError:
            roc_auc = None
    return {
        "accuracy": float(accuracy_score(observed_arr, predicted_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(observed_arr, predicted_arr)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1_score(observed_arr, predicted_arr, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(observed_arr, predicted_arr, average="weighted", zero_division=0)),
        "roc_auc": roc_auc,
        "high_precision": float(precision[1]),
        "high_recall": float(recall[1]),
        "confusion_matrix": confusion_matrix(observed_arr, predicted_arr, labels=[0, 1]).tolist(),
        "n": int(len(observed_arr)),
    }


def evaluate_binary(labeled: pd.DataFrame, feature_cols: list, target_col: str, group_col: str) -> dict:
    features = labeled[feature_cols]
    target = labeled[target_col].astype(int)
    groups = labeled[group_col].astype(str)
    folds, scheme = grouped_kfold_splits(features, target, groups)
    fold_metrics = []
    for train_idx, test_idx in folds:
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, max_features="sqrt", min_samples_leaf=1, min_samples_split=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        )
        model.fit(features.iloc[train_idx], target.iloc[train_idx])
        proba = model.predict_proba(features.iloc[test_idx])
        high_col = list(model.classes_).index(1) if 1 in model.classes_ else None
        proba_high = proba[:, high_col] if high_col is not None else np.zeros(len(test_idx))
        predicted = (proba_high >= 0.5).astype(int)
        fold_metrics.append(_binary_fold_metrics(target.iloc[test_idx], proba_high, predicted))
    combined = {"scheme": scheme, "n_folds": len(fold_metrics), "folds": fold_metrics}
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "roc_auc", "high_precision", "high_recall"):
        stats = mean_std([fold.get(key) for fold in fold_metrics])
        combined[key] = stats["mean"]
    combined["macro_precision"] = mean_std([f["macro_precision"] for f in fold_metrics])["mean"]
    combined["macro_recall"] = mean_std([f["macro_recall"] for f in fold_metrics])["mean"]
    combined["n"] = int(sum(f["n"] for f in fold_metrics))
    return combined


def evaluate_binary_lomo(labeled: pd.DataFrame, feature_cols: list, target_col: str, group_col: str) -> dict:
    groups = labeled[group_col].astype(str)
    folds = leave_one_municipality_out(groups)
    features = labeled[feature_cols]
    target = labeled[target_col].astype(int)
    per_municipality = []
    for train_idx, test_idx, municipality in folds:
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, max_features="sqrt", min_samples_leaf=1, min_samples_split=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        )
        model.fit(features.iloc[train_idx], target.iloc[train_idx])
        proba = model.predict_proba(features.iloc[test_idx])
        high_col = list(model.classes_).index(1) if 1 in model.classes_ else None
        proba_high = proba[:, high_col] if high_col is not None else np.zeros(len(test_idx))
        predicted = (proba_high >= 0.5).astype(int)
        metrics = _binary_fold_metrics(target.iloc[test_idx], proba_high, predicted)
        metrics["municipality"] = municipality
        per_municipality.append(metrics)
    stats = mean_std([f["macro_f1"] for f in per_municipality])
    return {"macro_f1": stats["mean"], "std": stats["std"], "n_folds": len(per_municipality), "per_municipality": per_municipality}


def main() -> None:
    grid = lab_data.load_grid()
    active_listings = lab_data.load_active_listings()
    places = lab_data.load_places_with_category()
    features = market_features(grid, active_listings, places)
    labeled, _ = label_performance(features)
    score = labeled[PERFORMANCE_SCORE]

    rows = []

    variants = [("EXP-N-20_60_20", 20.0, 80.0), ("EXP-N-30_40_30", 30.0, 70.0)]
    for exp_id, low_pct, high_pct in variants:
        alt_target_col = "alt_target"
        alt_labeled = labeled.copy()
        alt_labeled[alt_target_col] = alt_three_class_target(score, low_pct, high_pct)
        counts = alt_labeled[alt_target_col].value_counts().sort_index().to_dict()
        grouped = evaluate_grouped(alt_labeled, BASELINE_FEATURES, alt_target_col, MUNICIPALITY_COLUMN)
        lomo = evaluate_lomo(alt_labeled, BASELINE_FEATURES, alt_target_col, MUNICIPALITY_COLUMN)
        target_definition = f"low{int(low_pct)}_mid{int(high_pct - low_pct)}_high{int(100 - high_pct)}"
        cfg = {
            "experiment_id": exp_id, "low_percentile": low_pct, "high_percentile": high_pct,
            "class_counts": {"Low": counts.get(0, 0), "Moderate": counts.get(1, 0), "High": counts.get(2, 0)},
            "features": BASELINE_FEATURES,
        }
        notes = (
            f"# {exp_id}\n\nSame historical_performance_score as the production label, cut at the "
            f"{low_pct:.0f}th/{high_pct:.0f}th percentile instead of 25th/75th. Class counts: "
            f"{cfg['class_counts']}.\n\nGrouped-CV macro F1: {grouped['macro_f1']:.4f}. "
            f"LOMO mean macro F1: {lomo['macro_f1']:.4f}, std {lomo['cv']['macro_f1']['std']:.4f}.\n\n"
            "This is an experiment only; the official thesis target (25th/75th percentile, three "
            "classes) is unchanged in the production pipeline.\n"
        )
        save_experiment(EXPERIMENTS_DIR / "target_definition" / exp_id, cfg, grouped, BASELINE_FEATURES, notes, lomo_metrics=lomo)
        rows.append(results_row(
            exp_id, "target_definition", "random_forest", grouped, len(BASELINE_FEATURES), lomo_metrics=lomo,
            target_definition=target_definition, notes=f"class counts {cfg['class_counts']}",
        ))
        print(f"{exp_id:18s} macro_f1={grouped['macro_f1']:.4f} lomo_mean={lomo['macro_f1']:.4f} lomo_std={lomo['cv']['macro_f1']['std']:.4f} classes={cfg['class_counts']}")

    # Binary High vs Not High
    exp_id = "EXP-N-binary_high"
    binary_col = "is_high"
    binary_labeled = labeled.copy()
    binary_labeled[binary_col] = binary_high_target(score, 75.0)
    counts = binary_labeled[binary_col].value_counts().sort_index().to_dict()
    grouped = evaluate_binary(binary_labeled, BASELINE_FEATURES, binary_col, MUNICIPALITY_COLUMN)
    lomo = evaluate_binary_lomo(binary_labeled, BASELINE_FEATURES, binary_col, MUNICIPALITY_COLUMN)
    cfg = {
        "experiment_id": exp_id, "high_percentile": 75.0,
        "class_counts": {"NotHigh": counts.get(0, 0), "High": counts.get(1, 0)},
        "features": BASELINE_FEATURES,
    }
    notes = (
        f"# {exp_id}\n\nCollapses the three-class target to binary: top 25% by "
        f"historical_performance_score = High (1), everything else = Not High (0). Class counts: "
        f"{cfg['class_counts']}.\n\nGrouped-CV macro F1: {grouped['macro_f1']:.4f}, "
        f"High precision {grouped['high_precision']:.4f}, High recall {grouped['high_recall']:.4f}. "
        f"LOMO mean macro F1: {lomo['macro_f1']:.4f}, std {lomo['std']:.4f} across {lomo['n_folds']} "
        "municipality folds.\n\nThis metric is NOT directly comparable to the three-class macro F1 "
        "above (fewer classes makes the classification problem easier by construction); it answers a "
        "different question (\"is this square High or not\") than the three-class screener.\n"
    )
    (EXPERIMENTS_DIR / "target_definition" / exp_id).mkdir(parents=True, exist_ok=True)
    import json
    (EXPERIMENTS_DIR / "target_definition" / exp_id / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (EXPERIMENTS_DIR / "target_definition" / exp_id / "metrics.json").write_text(
        json.dumps({"grouped_cv": {k: v for k, v in grouped.items() if k != "folds"},
                    "leave_one_municipality_out": {k: v for k, v in lomo.items() if k != "per_municipality"}}, indent=2), encoding="utf-8")
    (EXPERIMENTS_DIR / "target_definition" / exp_id / "notes.md").write_text(notes, encoding="utf-8")
    rows.append({
        "experiment_id": exp_id, "feature_group": "target_definition", "model": "random_forest",
        "grid_size": "1000m", "poi_radius": "", "airbnb_radius": "", "target_definition": "binary_high_vs_not",
        "n_features": len(BASELINE_FEATURES), "n_train": "", "n_test": sum(f["n"] for f in grouped["folds"]),
        "accuracy": grouped["accuracy"], "macro_precision": grouped["macro_precision"], "macro_recall": grouped["macro_recall"],
        "macro_f1": grouped["macro_f1"], "weighted_f1": grouped["weighted_f1"], "high_precision": grouped["high_precision"],
        "high_recall": grouped["high_recall"], "validation_score": grouped["macro_f1"], "mean_municipality_score": lomo["macro_f1"],
        "std_municipality_score": lomo["std"], "notes": f"binary target, not comparable to 3-class macro F1; class counts {cfg['class_counts']}",
    })
    print(f"{exp_id:18s} macro_f1={grouped['macro_f1']:.4f} lomo_mean={lomo['macro_f1']:.4f} lomo_std={lomo['std']:.4f} classes={cfg['class_counts']}")

    append_result_rows(RESULTS_CSV, rows)
    print(f"Wrote {len(rows)} rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
