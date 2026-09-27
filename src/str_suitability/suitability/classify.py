import mapclassify
import pandas as pd

CLASS_COUNT = 5
SUITABILITY_CLASS_COLUMN = "suitability_class"
CLASSIFICATION_METHOD = (
    "Jenks Natural Breaks classification implemented using the Fisher-Jenks algorithm"
)

CLASS_LABELS = (
    "Very Low Suitability",
    "Low Suitability",
    "Moderate Suitability",
    "High Suitability",
    "Very High Suitability",
)

TRAFFIC_CLASS_LABELS = (
    "Low",
    "Moderate",
    "High",
)


def classify_suitability(scores: pd.Series, classes: int = CLASS_COUNT) -> tuple[pd.Series, dict]:
    assert classes == len(CLASS_LABELS), (
        f"{len(CLASS_LABELS)} class labels are defined but {classes} classes were requested"
    )
    values = scores.to_numpy(dtype=float)
    classifier = mapclassify.FisherJenks(values, k=classes)

    labels = pd.Series(
        [CLASS_LABELS[int(bin_index)] for bin_index in classifier.yb],
        index=scores.index,
        name=SUITABILITY_CLASS_COLUMN,
    )
    assert labels.nunique() == classes, (
        f"Jenks natural breaks produced {labels.nunique()} distinct classes, expected {classes}"
    )

    report = {
        "breaks": [float(edge) for edge in classifier.bins],
        "class_counts": {label: int((labels == label).sum()) for label in CLASS_LABELS},
    }
    return labels, report


def classify_traffic_light(scores: pd.Series) -> tuple[pd.Series, dict]:
    values = scores.to_numpy(dtype=float)
    classifier = mapclassify.FisherJenks(values, k=len(TRAFFIC_CLASS_LABELS))
    labels = pd.Series(
        [TRAFFIC_CLASS_LABELS[int(bin_index)] for bin_index in classifier.yb],
        index=scores.index,
        name=SUITABILITY_CLASS_COLUMN,
    )
    assert labels.nunique() == len(TRAFFIC_CLASS_LABELS), (
        f"Jenks natural breaks produced {labels.nunique()} distinct classes, "
        f"expected {len(TRAFFIC_CLASS_LABELS)}"
    )
    report = {
        "breaks": [float(edge) for edge in classifier.bins],
        "class_counts": {label: int((labels == label).sum()) for label in TRAFFIC_CLASS_LABELS},
        "class_labels": list(TRAFFIC_CLASS_LABELS),
    }
    return labels, report
