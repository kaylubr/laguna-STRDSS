"""Capture and save the fixed baseline (spec step 2/3).

Reproduces the production five-feature Random Forest classifier's validation
result using a separate implementation of the same splitting rule and metrics
(experiments/lab/common.py), so the saved baseline numbers are independently
verified, not copied from model_summary.json. Also loads model_summary.json and
reports it side by side so any difference is visible, not hidden.

Run with: uv run python experiments/run_baseline.py
"""

from __future__ import annotations

import json
from pathlib import Path

from str_suitability.modeling.location_classifier import (
    CLASSIFIER_FEATURES,
    MUNICIPALITY_COLUMN,
    PERFORMANCE_CLASS,
    label_performance,
    market_features,
)

from experiments.lab import data as lab_data
from experiments.lab.common import evaluate_grouped, save_experiment

EXPERIMENTS_DIR = Path(__file__).parent
BASELINE_DIR = EXPERIMENTS_DIR / "baseline"


def main() -> None:
    grid = lab_data.load_grid()
    active_listings = lab_data.load_active_listings()
    places = lab_data.load_places_with_category()  # longitude/latitude/name(/category); category unused here

    features = market_features(grid, active_listings, places)
    labeled, label_report = label_performance(features)

    feature_cols = list(CLASSIFIER_FEATURES)
    forest_metrics = evaluate_grouped(
        labeled, feature_cols, PERFORMANCE_CLASS, MUNICIPALITY_COLUMN, model_name="random_forest"
    )
    majority_metrics = evaluate_grouped(
        labeled, feature_cols, PERFORMANCE_CLASS, MUNICIPALITY_COLUMN, model_name="majority_baseline",
        folds=None,
    )
    stratified_metrics = evaluate_grouped(
        labeled, feature_cols, PERFORMANCE_CLASS, MUNICIPALITY_COLUMN, model_name="stratified_baseline",
    )

    high_index = 2
    # predicted High = column sum across confusion matrices (rows=true, cols=predicted)
    predicted_high = sum(
        sum(fold["confusion_matrix"][true][high_index] for true in range(3)) for fold in forest_metrics["folds"]
    )
    actual_high = sum(fold["class_counts"]["High"] for fold in forest_metrics["folds"])
    matched_high = sum(fold["confusion_matrix"][high_index][high_index] for fold in forest_metrics["folds"])

    production_summary = None
    try:
        production_summary = lab_data.load_model_summary().get("models", {}).get("location_success")
    except FileNotFoundError:
        pass

    config = {
        "model_type": "RandomForestClassifier",
        "hyperparameters": {
            "n_estimators": 200, "max_depth": 10, "max_features": "sqrt",
            "min_samples_leaf": 1, "min_samples_split": 2, "class_weight": "balanced",
            "random_state": 42,
        },
        "n_trees": 200,
        "features": feature_cols,
        "n_features": len(feature_cols),
        "target": PERFORMANCE_CLASS,
        "target_definition": "Low <= 25th pct, High >= 75th pct of historical_performance_score, else Moderate",
        "n_labeled_cells": int(len(labeled)),
        "n_municipalities": int(labeled[MUNICIPALITY_COLUMN].nunique()),
        "validation_scheme": forest_metrics["scheme"],
        "n_folds": forest_metrics["n_folds"],
        "class_counts": label_report["class_counts"],
    }

    reproduced = {
        "random_forest": {k: v for k, v in forest_metrics.items() if k not in {"folds"}},
        "majority_baseline": {k: v for k, v in majority_metrics.items() if k not in {"folds"}},
        "stratified_baseline": {k: v for k, v in stratified_metrics.items() if k not in {"folds"}},
        "high_calls": {
            "predicted_high": int(predicted_high),
            "actually_high": int(actual_high),
            "matched": int(matched_high),
        },
    }

    notes_lines = [
        "# Baseline (spec step 2/3)",
        "",
        "This reproduces the production 5-feature Random Forest classifier's",
        "municipality-grouped validation using a separate implementation of the",
        "same splitting rule and metrics (experiments/lab/common.py). It is not a",
        "copy of model_summary.json's numbers; it is a fresh fit-and-score.",
        "",
        f"Random Forest macro F1 (reproduced): {forest_metrics['macro_f1']:.4f}",
        f"Stratified baseline macro F1 (reproduced): {stratified_metrics['macro_f1']:.4f}",
        f"Majority baseline macro F1 (reproduced): {majority_metrics['macro_f1']:.4f}",
        f"Predicted High: {predicted_high}, actually High among those: {matched_high}, actual High total: {actual_high}",
        "",
    ]
    if production_summary is not None:
        prod_test = production_summary.get("test_metrics", {})
        prod_strat = production_summary.get("stratified_baseline_metrics", {})
        notes_lines += [
            "## From the current thesis model's own model_summary.json (not modified)",
            f"macro_f1: {prod_test.get('macro_f1')}",
            f"stratified_baseline macro_f1: {prod_strat.get('macro_f1')}",
            "",
            "## Comparison",
        ]
        diff = forest_metrics['macro_f1'] - float(prod_test.get('macro_f1', float('nan')))
        if abs(diff) < 0.01:
            notes_lines.append(
                f"The reproduced macro F1 matches model_summary.json within {abs(diff):.4f} "
                "(StratifiedGroupKFold with shuffle=True and a fixed random_state gives the "
                "same folds for the same input rows, so a small residual difference, if any, "
                "is expected floating-point / library-version noise, not a methodology error)."
            )
        else:
            notes_lines.append(
                f"The reproduced macro F1 differs from model_summary.json by {diff:+.4f}. "
                "Reporting the actual reproduced number rather than assuming it matches the "
                "user's expected ~0.314, per the instruction to verify, not assume."
            )
    notes = "\n".join(notes_lines)

    save_experiment(
        BASELINE_DIR, config, forest_metrics, feature_cols, notes, lomo_metrics=None,
    )
    (BASELINE_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "grouped_cv": {k: v for k, v in forest_metrics.items() if k != "folds"},
                "majority_baseline": {k: v for k, v in majority_metrics.items() if k != "folds"},
                "stratified_baseline": {k: v for k, v in stratified_metrics.items() if k != "folds"},
                "high_calls": reproduced["high_calls"],
                "production_model_summary_test_metrics": production_summary.get("test_metrics") if production_summary else None,
                "production_model_summary_stratified_baseline_metrics": production_summary.get("stratified_baseline_metrics") if production_summary else None,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("Random Forest macro F1:", round(forest_metrics["macro_f1"], 4))
    print("Stratified baseline macro F1:", round(stratified_metrics["macro_f1"], 4))
    print("Majority baseline macro F1:", round(majority_metrics["macro_f1"], 4))
    print("High calls: predicted", predicted_high, "matched", matched_high, "actual", actual_high)
    if production_summary is not None:
        print("model_summary.json macro_f1:", production_summary.get("test_metrics", {}).get("macro_f1"))
        print(
            "model_summary.json stratified baseline macro_f1:",
            production_summary.get("stratified_baseline_metrics", {}).get("macro_f1"),
        )
    print("Saved to", BASELINE_DIR)


if __name__ == "__main__":
    main()
