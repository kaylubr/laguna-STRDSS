import numpy as np
import pandas as pd

from str_suitability.modeling.location_classifier import (
    CLASSIFIER_FEATURES,
    CLASS_COLUMN,
    PROBABILITY_COLUMNS,
)
from str_suitability.rural.classify import CELL_ID_COLUMN

DIAGRAM_PATH_NAME = "rural-cell-diagrams.md"


def classifier_tree_trace(model, values: list[float]) -> dict:
    """Walk the first tree for one square. op 0 is <=, 1 is >, 2 is a blank value."""
    tree = model.estimators_[0].tree_
    node = 0
    steps = []
    while tree.feature[node] >= 0:
        feature_index = int(tree.feature[node])
        threshold = float(tree.threshold[node])
        measurement = float(values[feature_index])
        shown_threshold = round(threshold, 4)
        if np.isnan(measurement):
            go_left = bool(tree.missing_go_to_left[node])
            steps.append([feature_index, 2, shown_threshold, None, 1 if go_left else 0])
            node = int(tree.children_left[node] if go_left else tree.children_right[node])
            continue
        go_left = measurement <= threshold
        steps.append([
            feature_index,
            0 if go_left else 1,
            shown_threshold,
            round(measurement, 4),
            1 if go_left else 0,
        ])
        node = int(tree.children_left[node] if go_left else tree.children_right[node])
    counts = np.asarray(tree.value[node][0], dtype=float)
    total = float(counts.sum()) or 1.0
    leaf = [round(float(share), 3) for share in counts / total]
    return {"steps": steps, "leaf": leaf}


def classifier_tree_path(model, values: list[float]) -> list[str]:
    """Walk the first tree. A missing surrounding mean follows that tree's missing branch."""
    tree = model.estimators_[0].tree_
    node = 0
    steps = []
    while tree.feature[node] >= 0:
        feature_index = int(tree.feature[node])
        feature = CLASSIFIER_FEATURES[feature_index]
        threshold = float(tree.threshold[node])
        measurement = float(values[feature_index])
        if np.isnan(measurement):
            go_left = bool(tree.missing_go_to_left[node])
            steps.append(f"{feature} missing -> {'left' if go_left else 'right'}")
            node = int(tree.children_left[node] if go_left else tree.children_right[node])
            continue
        if measurement <= threshold:
            steps.append(f"{feature} {measurement:.4f} <= {threshold:.4f}")
            node = int(tree.children_left[node])
        else:
            steps.append(f"{feature} {measurement:.4f} > {threshold:.4f}")
            node = int(tree.children_right[node])
    counts = np.asarray(tree.value[node][0], dtype=float)
    total = float(counts.sum()) or 1.0
    shares = counts / total
    names = ("Low", "Moderate", "High")
    text = ", ".join(f"P({name}) {share:.3f}" for name, share in zip(names, shares, strict=True))
    steps.append(f"this tree's class shares: {text}")
    return steps


def write_cell_diagrams(output: pd.DataFrame, model, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        "# Location-success tree paths",
        "",
        "The map color is the random forest's predicted Airbnb performance class.",
        "Each section walks the first tree only. The class is the highest of the averaged probabilities.",
        "A missing surrounding mean means no other active listing was within 5 km outside the cell.",
        "",
    ]
    for row in output.itertuples(index=False):
        sections.extend(_section(row, model))
    path.write_text("\n".join(sections), encoding="utf-8")


def _section(row, model) -> list[str]:
    values = []
    for column in CLASSIFIER_FEATURES:
        measurement = getattr(row, column)
        values.append(np.nan if pd.isna(measurement) else float(measurement))
    lines = [
        f"## {getattr(row, CELL_ID_COLUMN)}",
        "",
        f"- Predicted class: {getattr(row, CLASS_COLUMN)}",
        *[
            f"- {column}: {float(getattr(row, column)):.4f}"
            for column in PROBABILITY_COLUMNS
        ],
        "- First tree:",
        "",
        "```text",
        *classifier_tree_path(model, values),
        "```",
        "",
    ]
    return lines
