import pandas as pd

from str_suitability.pipeline import FEATURE_COLUMNS
from str_suitability.rural.classify import CELL_ID_COLUMN
from str_suitability.rural.site_score import (
    DISTANCE_COLUMN,
    LISTINGS_IN_CELL_COLUMN,
    LISTINGS_NEARBY_COLUMN,
    NEAREST_PLACE_COLUMN,
    NEARBY_OCCUPANCY_COLUMN,
    NEARBY_REVENUE_COLUMN,
)
from str_suitability.rural.suitability import RURAL_CLASS_COLUMN, RURAL_SCORE_COLUMN
from str_suitability.suitability.classify import TRAFFIC_CLASS_LABELS

DIAGRAM_PATH_NAME = "rural-cell-diagrams.md"


def tree_path(model, values: list[float], leaf_label: str) -> list[str]:
    tree = model.estimators_[0].tree_
    node = 0
    steps = []
    while tree.feature[node] >= 0:
        feature = FEATURE_COLUMNS[int(tree.feature[node])]
        threshold = float(tree.threshold[node])
        measurement = float(values[int(tree.feature[node])])
        if measurement <= threshold:
            steps.append(f"{feature} {measurement:.4f} <= {threshold:.4f}")
            node = int(tree.children_left[node])
        else:
            steps.append(f"{feature} {measurement:.4f} > {threshold:.4f}")
            node = int(tree.children_right[node])
    vote = float(tree.value[node][0][0])
    steps.append(f"leaf vote for {leaf_label}: {vote:.2f}")
    return steps


def revenue_tree_path(model, values: list[float]) -> list[str]:
    return tree_path(model, values, "trailing-twelve-month revenue")


def write_cell_diagrams(output: pd.DataFrame, revenue_model, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        "# Rural cell training diagrams",
        "",
        "Each rural cell is scored as tourist access + nearby market − local competition.",
        "The path below is the first tree in the rural revenue forest.",
        "Green is High, yellow is Moderate, and red is Low.",
        "",
    ]
    for row in output.itertuples(index=False):
        sections.extend(_section(row, revenue_model))
    path.write_text("\n".join(sections), encoding="utf-8")


def example_cells(output: pd.DataFrame) -> list[pd.Series]:
    chosen = []
    for label in TRAFFIC_CLASS_LABELS:
        matches = output.loc[output[RURAL_CLASS_COLUMN] == label].sort_values(RURAL_SCORE_COLUMN)
        if len(matches) == 0:
            continue
        if label == "Low":
            chosen.append(matches.iloc[0])
        elif label == "High":
            chosen.append(matches.iloc[-1])
        else:
            chosen.append(matches.iloc[len(matches) // 2])
    return chosen


def format_example(row: pd.Series, revenue_model) -> str:
    values = [float(row[column]) for column in FEATURE_COLUMNS]
    path = "\n".join(f"  {step}" for step in revenue_tree_path(revenue_model, values))
    return "\n".join(
        [
            f"{row[RURAL_CLASS_COLUMN]}  {row[CELL_ID_COLUMN]}",
            f"  score {float(row[RURAL_SCORE_COLUMN]):.4f}",
            f"  nearest place {row[NEAREST_PLACE_COLUMN]} at {float(row[DISTANCE_COLUMN]):.3f} km",
            f"  nearby mean revenue {float(row[NEARBY_REVENUE_COLUMN]):.2f}",
            f"  nearby mean occupancy {float(row[NEARBY_OCCUPANCY_COLUMN]):.5f}",
            f"  listings in cell {int(row[LISTINGS_IN_CELL_COLUMN])}, "
            f"within 5 km {int(row[LISTINGS_NEARBY_COLUMN])}",
            "  revenue tree:",
            path,
        ]
    )


def _section(row, revenue_model) -> list[str]:
    lines = [f"## {row.cell_id}", ""]
    if not bool(row.in_rural_analysis):
        lines.extend([str(row.rural_suitability_class), ""])
        return lines
    values = [float(getattr(row, column)) for column in FEATURE_COLUMNS]
    lines.extend(
        [
            f"- Class: {row.rural_suitability_class}",
            f"- Score: {float(row.rural_composite_suitability_score):.4f}",
            f"- Nearest listed place: {row.nearest_listed_tourist_place} "
            f"({float(row.distance_to_listed_tourist_place):.3f} km)",
            f"- Tourist access contribution: {float(row.tourist_access_contribution):.4f}",
            f"- Nearby mean revenue: {float(row.nearby_mean_revenue):.2f}",
            f"- Nearby mean occupancy: {float(row.nearby_mean_occupancy):.5f}",
            f"- Nearby market contribution: {float(row.nearby_market_contribution):.4f}",
            f"- Listings in the cell: {int(row.listings_in_cell)}",
            f"- Listings within 5 km: {int(row.listings_nearby)}",
            f"- Competition contribution: {float(row.local_competition_contribution):.4f}",
            "- Revenue tree path:",
            "",
            "```text",
            *revenue_tree_path(revenue_model, values),
            "```",
            "",
        ]
    )
    return lines
