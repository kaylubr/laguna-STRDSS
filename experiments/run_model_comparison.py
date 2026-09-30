"""Model comparison (spec group P): Logistic Regression, Decision Tree, Random
Forest, Gradient Boosting, all on the identical baseline five features,
identical municipality-grouped folds, identical target, and identical
evaluation metrics, so the comparison is fair. No new dependency is added;
every model here is already in this project's scikit-learn install.

Run with: uv run python -m experiments.run_model_comparison
"""

from __future__ import annotations

from pathlib import Path

from str_suitability.modeling.location_classifier import CLASSIFIER_FEATURES, MUNICIPALITY_COLUMN, PERFORMANCE_CLASS

from experiments.lab import data as lab_data
from experiments.lab.common import (
    append_result_rows,
    evaluate_grouped,
    evaluate_lomo,
    grouped_kfold_splits,
    results_row,
    save_experiment,
)
from str_suitability.modeling.location_classifier import label_performance, market_features

EXPERIMENTS_DIR = Path(__file__).parent
RESULTS_CSV = EXPERIMENTS_DIR / "results.csv"
BASELINE_FEATURES = list(CLASSIFIER_FEATURES)

MODELS = ("random_forest", "logistic_regression", "decision_tree", "gradient_boosting")


def main() -> None:
    grid = lab_data.load_grid()
    active_listings = lab_data.load_active_listings()
    places = lab_data.load_places_with_category()
    features = market_features(grid, active_listings, places)
    labeled, _ = label_performance(features)

    folds, scheme = grouped_kfold_splits(
        labeled[BASELINE_FEATURES], labeled[PERFORMANCE_CLASS].astype(int), labeled[MUNICIPALITY_COLUMN].astype(str)
    )
    print("shared folds scheme:", scheme, "n_folds:", len(folds))

    rows = []
    for model_name in MODELS:
        exp_id = f"EXP-P-{model_name}"
        grouped = evaluate_grouped(
            labeled, BASELINE_FEATURES, PERFORMANCE_CLASS, MUNICIPALITY_COLUMN, model_name=model_name, folds=folds
        )
        lomo = evaluate_lomo(labeled, BASELINE_FEATURES, PERFORMANCE_CLASS, MUNICIPALITY_COLUMN, model_name=model_name)
        needs_imputation = model_name != "random_forest"
        cfg = {
            "experiment_id": exp_id, "model": model_name, "features": BASELINE_FEATURES,
            "preprocessing": "median imputation before fitting (model cannot take NaN natively)" if needs_imputation
                              else "none; scikit-learn RandomForestClassifier's native NaN support, same as production",
            "n_labeled_cells": int(len(labeled)),
        }
        notes = (
            f"# {exp_id}\n\nSame baseline 5 features, same target, same municipality-grouped folds "
            f"as every other model in this comparison.\n\n"
            f"Preprocessing: {cfg['preprocessing']}.\n\n"
            f"Grouped-CV macro F1: {grouped['macro_f1']:.4f}. "
            f"LOMO mean macro F1: {lomo['macro_f1']:.4f}, std {lomo['cv']['macro_f1']['std']:.4f}.\n"
        )
        save_experiment(EXPERIMENTS_DIR / "model_comparison" / exp_id, cfg, grouped, BASELINE_FEATURES, notes, lomo_metrics=lomo)
        rows.append(results_row(
            exp_id, "model_comparison", model_name, grouped, len(BASELINE_FEATURES), lomo_metrics=lomo,
            notes=cfg["preprocessing"],
        ))
        print(
            f"{model_name:22s} macro_f1={grouped['macro_f1']:.4f} lomo_mean={lomo['macro_f1']:.4f} "
            f"lomo_std={lomo['cv']['macro_f1']['std']:.4f}"
        )

    append_result_rows(RESULTS_CSV, rows)
    print(f"Wrote {len(rows)} rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
