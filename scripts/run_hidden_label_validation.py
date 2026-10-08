"""Predict the 220 training cells with their labels hidden.

Each cell is scored by a forest trained only on other municipalities.
The label is not one of the six features. Nothing is written over the
original dataset or the existing site-candidate outputs.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from str_suitability import config
from str_suitability.modeling.site_candidate import (
    CELL_ID_COLUMN,
    FEATURES,
    PRESENCE_LABEL,
    build_training_sample,
    evaluate_forest,
    load_cells,
)

OUT = config.PROJECT_ROOT / "results" / "hidden_label_validation"
PREVIOUS = {
    "roc_auc": 0.7274258287458627,
    "pr_auc": 0.7128459949005714,
    "macro_f1": 0.6561302663653507,
}


def _classification(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_listed": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall_listed": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "f1_listed": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "precision_empty": float(precision_score(labels, predictions, pos_label=0, zero_division=0)),
        "recall_empty": float(recall_score(labels, predictions, pos_label=0, zero_division=0)),
        "f1_empty": float(f1_score(labels, predictions, pos_label=0, zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


def main() -> None:
    if PRESENCE_LABEL in FEATURES:
        raise RuntimeError("the label is one of the model features")
    cells, _places = load_cells()
    sample = build_training_sample(cells, negative_seed=42)
    oof, report = evaluate_forest(sample)
    roc = float(report["metrics"]["roc_auc"]["mean"])
    pr = float(report["metrics"]["pr_auc"]["mean"])
    f1 = float(report["metrics"]["macro_f1"]["mean"])
    if abs(roc - PREVIOUS["roc_auc"]) > 1e-9 or abs(f1 - PREVIOUS["macro_f1"]) > 1e-9:
        raise SystemExit(
            "held-out forest did not match the saved run: "
            f"ROC-AUC {roc} vs {PREVIOUS['roc_auc']}, macro F1 {f1} vs {PREVIOUS['macro_f1']}"
        )

    thresholds = {int(row["fold"]): float(row["youden_threshold"]) for row in report["folds"]}
    checked = oof.copy()
    checked[CELL_ID_COLUMN] = checked[CELL_ID_COLUMN].astype(str)
    checked["fold"] = checked["fold"].astype(int)
    checked["original_label"] = np.where(checked[PRESENCE_LABEL].eq(1), "listed", "empty")
    checked["youden_threshold"] = checked["fold"].map(thresholds)
    checked["predicted_listed"] = checked["random_forest_raw"].ge(checked["youden_threshold"])
    checked["predicted_label"] = np.where(checked["predicted_listed"], "listed", "empty")
    checked["correct"] = checked["predicted_label"].eq(checked["original_label"])
    labels = checked[PRESENCE_LABEL].astype(int).to_numpy()
    predictions = checked["predicted_listed"].astype(int).to_numpy()
    overall = _classification(labels, predictions)
    listed_only = checked.loc[checked["original_label"].eq("listed")]
    listed_recovered = int(listed_only["predicted_label"].eq("listed").sum())

    OUT.mkdir(parents=True, exist_ok=True)
    checked[
        [
            CELL_ID_COLUMN,
            "municipality",
            "fold",
            "original_label",
            "random_forest_raw",
            "youden_threshold",
            "predicted_label",
            "correct",
        ]
    ].rename(columns={"random_forest_raw": "score"}).sort_values(
        ["fold", CELL_ID_COLUMN]
    ).to_csv(OUT / "cell_predictions.csv", index=False)

    summary = {
        "cells_tested": int(len(checked)),
        "listed_cells_tested": int(checked["original_label"].eq("listed").sum()),
        "empty_cells_tested": int(checked["original_label"].eq("empty").sum()),
        "rule": (
            "Each cell is scored by a 200-tree forest fit on the other municipalities. "
            "The six location features do not include the label. "
            "The predicted class uses that fold's training-side Youden cutoff."
        ),
        "held_out_roc_auc": roc,
        "held_out_pr_auc": pr,
        "held_out_macro_f1": f1,
        "previous_reported": PREVIOUS,
        "classification": overall,
        "listed_cells_predicted_listed": listed_recovered,
        "listed_cells_missed": int(len(listed_only) - listed_recovered),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
