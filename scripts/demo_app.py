"""Panel demo for the 1 km rural site-candidate forest.

Run from the project root:

    uv run streamlit run scripts/demo_app.py
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from str_suitability.modeling.site_candidate import (  # noqa: E402
    FEATURES,
    PRESENCE_LABEL,
    random_forest,
)

RURAL_DIR = ROOT / "data" / "processed" / "rural"
GRID_PATH = RURAL_DIR / "grid_suitability.parquet"
SCORES_PATH = RURAL_DIR / "site_candidate_v2_rf_scores.parquet"
SAMPLE_PATH = RURAL_DIR / "site_candidate_v2_training.parquet"
REPORT_PATH = RURAL_DIR / "site_candidate_v2_report.json"
ROBUST_PATH = ROOT / "results" / "site_candidate_robustness" / "robustness_report.json"
MAP_PATH = ROOT / "data" / "processed" / "frontend" / "index.html"

FEATURE_LABELS = {
    "listed_places_within_radius": "Landscape places within 5 km",
    "distance_to_listed_tourist_place": "Nearest landscape place (km)",
    "road_distance_km": "Nearest mapped road (km)",
    "distance_to_nearest_town_center_km": "Nearest town centre (km)",
    "distance_to_major_road_km": "Nearest major road (km)",
    "attraction_count_within_5km": "Falls and mountains within 5 km",
}
DISTANCE_FEATURES = (
    "road_distance_km",
    "distance_to_major_road_km",
    "distance_to_nearest_town_center_km",
    "distance_to_listed_tourist_place",
)
COUNT_FEATURES = (
    "listed_places_within_radius",
    "attraction_count_within_5km",
)


@st.cache_data
def load_grid() -> pd.DataFrame:
    frame = pd.read_parquet(GRID_PATH)
    if "geometry" in frame.columns:
        frame = frame.drop(columns=["geometry"])
    return frame


@st.cache_data
def load_cells() -> pd.DataFrame:
    frame = pd.read_parquet(SCORES_PATH).rename(columns={"listings_in_cell_descriptive": "listings_in_cell"})
    frame["cell_class"] = "rural"
    frame["is_empty"] = frame["listings_in_cell"].eq(0)
    return frame


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_parquet(SAMPLE_PATH)


@st.cache_data
def load_evaluation() -> dict:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    forest = report["metrics"]["v2_random_forest"]
    true_negative = false_positive = false_negative = true_positive = 0
    for fold in forest["folds"]:
        empty_row, listed_row = fold["confusion_matrix"]
        true_negative += empty_row[0]
        false_positive += empty_row[1]
        false_negative += listed_row[0]
        true_positive += listed_row[1]
    robust = json.loads(ROBUST_PATH.read_text(encoding="utf-8"))
    draws = []
    for draw in robust["draws"]:
        if draw.get("negative_sampling_seed") is None:
            continue
        metrics = draw["metrics"]
        draws.append(
            {
                "seed": draw["negative_sampling_seed"],
                "roc_auc": metrics["roc_auc"]["mean"],
                "pr_auc": metrics["pr_auc"]["mean"],
                "macro_f1": metrics["macro_f1"]["mean"],
            }
        )
    roc_values = [row["roc_auc"] for row in draws]
    return {
        "seed_42": {
            "roc_auc": forest["roc_auc"]["mean"],
            "pr_auc": forest["pr_auc"]["mean"],
            "macro_f1": forest["youden_macro_f1"]["mean"],
            "classification": {
                "true_negative": true_negative,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_positive": true_positive,
            },
            "permutation": [
                row for row in report["permutation_importance"] if row["model"] == "v2_random_forest"
            ],
        },
        "ten_seeds": {
            "draws": draws,
            "roc_auc": {
                "mean": float(np.mean(roc_values)),
                "min": float(min(roc_values)),
                "max": float(max(roc_values)),
            },
        },
    }


def _ready() -> bool:
    return GRID_PATH.exists() and SCORES_PATH.exists() and SAMPLE_PATH.exists() and MAP_PATH.exists()


def _bar(frame: pd.DataFrame, x: str, y: str, color: str | None = None) -> alt.Chart:
    chart = alt.Chart(frame).mark_bar().encode(
        x=alt.X(x, title=None),
        y=alt.Y(y, title=None),
        tooltip=list(frame.columns),
    )
    if color is not None:
        chart = chart.encode(color=alt.Color(color, title=None))
    return chart.properties(height=280)


def _page_map() -> None:
    st.subheader("1 km empty rural cells")
    st.caption(
        "This is the original 1 km map. Red is a lower score, gold is the middle, and green is a higher score. "
        "The score is similarity to listed rural cells, not a forecast of revenue."
    )
    if not MAP_PATH.exists():
        st.error("The 1 km map file is missing.")
        return
    st.iframe(
        MAP_PATH.read_text(encoding="utf-8"),
        height=760,
        alt="1 km rural site-candidate map",
    )


def _score_histograms(cells: pd.DataFrame) -> None:
    rural = cells.loc[cells["cell_class"].eq("rural") & cells["candidate_pattern_score"].notna()].copy()
    rural["class"] = np.where(rural["listings_in_cell"].ge(1), "Listed", "Empty")
    chart = (
        alt.Chart(rural)
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X("candidate_pattern_score:Q", bin=alt.Bin(maxbins=20), title="Similarity score"),
            y=alt.Y("count()", title="Cells"),
            color=alt.Color("class:N", title=None),
            xOffset="class:N",
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")
    st.caption("Listed cells sit further to the right. The two groups still overlap, which is why some empty cells score high and some listed cells score low.")


def _band_chart(cells: pd.DataFrame) -> None:
    empty = cells.loc[cells["is_empty"].fillna(False).astype(bool) & cells["display_band"].notna()]
    order = ["<0.40", "0.40 to <0.60", ">=0.60"]
    counts = empty["display_band"].value_counts().reindex(order).fillna(0).astype(int)
    frame = pd.DataFrame({"Band": order, "Empty rural cells": counts.to_numpy()})
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("Band:N", sort=order, title=None),
            y=alt.Y("Empty rural cells:Q", title=None),
            tooltip=["Band", "Empty rural cells"],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")


def _feature_boxes(sample: pd.DataFrame) -> None:
    long = sample.melt(
        id_vars=[PRESENCE_LABEL],
        value_vars=list(DISTANCE_FEATURES),
        var_name="feature",
        value_name="kilometres",
    )
    long["class"] = long[PRESENCE_LABEL].map({1: "Listed", 0: "Empty"})
    long["feature"] = long["feature"].map(FEATURE_LABELS)
    chart = (
        alt.Chart(long)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("class:N", title=None),
            y=alt.Y("kilometres:Q", title="Kilometres"),
            color=alt.Color("class:N", title=None),
            column=alt.Column("feature:N", title=None),
        )
        .properties(width=140, height=240)
    )
    st.altair_chart(chart, width="stretch")
    st.caption("On the 220-cell training sample, listed and empty cells differ most on the two road distances.")


def _importance_chart(evaluation: dict) -> None:
    rows = evaluation["seed_42"]["permutation"]
    frame = pd.DataFrame(rows)
    frame["Feature"] = frame["feature"].map(FEATURE_LABELS)
    frame = frame.rename(columns={"mean_auc_drop": "Mean ROC-AUC drop"})
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("Mean ROC-AUC drop:Q", title="Drop in ROC-AUC when this feature is shuffled"),
            y=alt.Y("Feature:N", sort="-x", title=None),
            tooltip=["Feature", "Mean ROC-AUC drop"],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, width="stretch")
    st.caption("A larger drop means the held-out ranking relied more on that feature. This is not a claim that the feature causes a listing.")


def _seed_chart(evaluation: dict) -> None:
    frame = pd.DataFrame(evaluation["ten_seeds"]["draws"])
    frame["Seed"] = frame["seed"].astype(str)
    frame["Seed 42"] = np.where(frame["seed"].eq(42), "Seed 42", "Other seed")
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("Seed:N", title="Negative-sample seed"),
            y=alt.Y("roc_auc:Q", title="Mean ROC-AUC", scale=alt.Scale(domain=[0.5, 0.85])),
            color=alt.Color("Seed 42:N", title=None),
            tooltip=["seed", "roc_auc", "pr_auc", "macro_f1"],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")
    ten = evaluation["ten_seeds"]["roc_auc"]
    st.caption(
        f"Ten different draws of the 110 empty cells. Mean ROC-AUC {ten['mean']:.3f}, "
        f"from {ten['min']:.3f} to {ten['max']:.3f}. Seed 42 is the draw used in the paper tables."
    )


def _municipality_chart(cells: pd.DataFrame) -> None:
    high = cells.loc[
        cells["cell_class"].eq("rural")
        & cells["listings_in_cell"].eq(0)
        & cells["candidate_pattern_score"].ge(0.60)
    ]
    counts = high.groupby("municipality").size().sort_values(ascending=False).head(12)
    frame = counts.rename("high_empty_cells").reset_index()
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("high_empty_cells:Q", title="Empty cells scoring 0.60 or above"),
            y=alt.Y("municipality:N", sort="-x", title=None),
            tooltip=["municipality", "high_empty_cells"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")


def _page_data() -> None:
    grid = load_grid()
    cells = load_cells()
    sample = load_sample()
    evaluation = load_evaluation() if REPORT_PATH.exists() and ROBUST_PATH.exists() else None
    rural = cells["cell_class"].eq("rural")
    listed = int((rural & cells["listings_in_cell"].ge(1)).sum())
    empty = int((rural & cells["listings_in_cell"].eq(0)).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grid cells", f"{len(grid):,}")
    c2.metric("Rural cells", f"{int(rural.sum()):,}")
    c3.metric("With a listing", f"{listed:,}")
    c4.metric("Empty rural", f"{empty:,}")

    st.subheader("Scores of listed and empty rural cells")
    _score_histograms(cells)
    left, right = st.columns(2)
    with left:
        st.subheader("Empty cells by map color")
        _band_chart(cells)
    with right:
        st.subheader("Where the high empty cells are")
        _municipality_chart(cells)

    st.subheader("The six features on the training sample")
    st.write(
        "Every rural cell has the same six location numbers. "
        "The label is only whether that cell already contains an active listing. "
        "Revenue and occupancy are not inputs."
    )
    _feature_boxes(sample)

    if evaluation is not None:
        st.subheader("Which feature the ranking uses")
        _importance_chart(evaluation)
        st.subheader("The same forest on ten empty-cell draws")
        _seed_chart(evaluation)

    show = cells.loc[rural, ["cell_id", "municipality", "listings_in_cell", *FEATURES, "candidate_pattern_score"]].copy()
    show = show.rename(columns={**FEATURE_LABELS, "candidate_pattern_score": "Similarity score", "listings_in_cell": "Active listings"})
    municipality = st.selectbox("Municipality", ["All", *sorted(show["municipality"].dropna().unique())])
    if municipality != "All":
        show = show.loc[show["municipality"].eq(municipality)]
    st.dataframe(show, width="stretch", height=360)


def _train_one_fold(sample: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    features = sample[list(FEATURES)]
    labels = sample[PRESENCE_LABEL].astype(int)
    groups = sample["municipality"].astype(str)
    train_index, test_index = next(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(features, labels, groups)
    )
    model = random_forest()
    model.fit(features.iloc[train_index], labels.iloc[train_index])
    class_index = list(model.classes_).index(1)
    held = sample.iloc[test_index].copy()
    held["score"] = model.predict_proba(features.iloc[test_index])[:, class_index]
    held["held_out_roc_auc"] = float(roc_auc_score(labels.iloc[test_index], held["score"]))
    held["train_rows"] = int(len(train_index))
    held["test_rows"] = int(len(test_index))
    return held, model


def _tree_shares(model, row: pd.Series) -> np.ndarray:
    features = row[list(FEATURES)].to_numpy(dtype=float).reshape(1, -1)
    shares = []
    for tree in model.estimators_:
        proba = tree.predict_proba(features)[0]
        classes = list(tree.classes_)
        shares.append(float(proba[classes.index(1)]) if 1 in classes else 0.0)
    return np.asarray(shares)


def _roc_chart(held: pd.DataFrame) -> None:
    fpr, tpr, _thresholds = roc_curve(held[PRESENCE_LABEL].astype(int), held["score"])
    curve = pd.DataFrame({"False positive rate": fpr, "True positive rate": tpr, "Line": "This fold"})
    chance = pd.DataFrame({"False positive rate": [0, 1], "True positive rate": [0, 1], "Line": "Chance"})
    chart = (
        alt.Chart(pd.concat([curve, chance], ignore_index=True))
        .mark_line()
        .encode(
            x=alt.X("False positive rate:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("True positive rate:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Line:N", title=None),
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")


def _leaf_share(estimator, node_id: int) -> tuple[int, int, float]:
    counts = estimator.tree_.value[node_id, 0]
    classes = list(estimator.classes_)
    total = float(counts.sum())
    listed = float(counts[classes.index(1)]) if 1 in classes and total else 0.0
    share = listed / total if total else 0.0
    return int(round(listed)), int(round(total)), float(share)



QUESTION_LABELS = {
    "listed_places_within_radius": "places within 5 km",
    "distance_to_listed_tourist_place": "landscape km",
    "road_distance_km": "nearest road km",
    "distance_to_nearest_town_center_km": "town centre km",
    "distance_to_major_road_km": "major road km",
    "attraction_count_within_5km": "falls and mountains",
}


def _question_text(name: str, threshold: float) -> str:
    label = QUESTION_LABELS[name]
    if name in COUNT_FEATURES:
        return f"Is {label} ≤ {int(np.floor(threshold))}?"
    return f"Is {label} ≤ {threshold:.2f}?"


def _band_word(share: float) -> tuple[str, str, str]:
    if share >= 0.60:
        return "HIGH", "#3dcf6e", "#ffffff"
    if share >= 0.40:
        return "Medium", "#ffe14a", "#1a1a1a"
    return "LOW", "#ff5d6c", "#ffffff"


def _path_nodes(estimator, values: np.ndarray) -> set[int]:
    tree = estimator.tree_
    node = 0
    seen = {0}
    while tree.feature[node] >= 0:
        feature_index = int(tree.feature[node])
        goes_left = float(values[feature_index]) <= float(tree.threshold[node])
        node = int(tree.children_left[node] if goes_left else tree.children_right[node])
        seen.add(node)
    return seen


def _shallow(estimator, node: int, depth: int) -> dict:
    tree = estimator.tree_
    _listed, _total, share = _leaf_share(estimator, node)
    if tree.feature[node] < 0 or depth >= 2:
        word, fill, ink = _band_word(share)
        return {"id": int(node), "leaf": True, "word": word, "fill": fill, "ink": ink}
    feature_index = int(tree.feature[node])
    name = FEATURES[feature_index]
    return {
        "id": int(node),
        "leaf": False,
        "text": _question_text(name, float(tree.threshold[node])),
        "yes": _shallow(estimator, int(tree.children_left[node]), depth + 1),
        "no": _shallow(estimator, int(tree.children_right[node]), depth + 1),
    }


def _pill(x: float, y: float, width: float, height: float, fill: str, text: str, ink: str, outlined: bool) -> str:
    stroke = ' stroke="#111" stroke-width="3"' if outlined else ""
    return (
        f'<rect x="{x - width / 2:.1f}" y="{y - height / 2:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="{height / 2:.1f}" fill="{fill}"{stroke}/>'
        f'<text x="{x:.1f}" y="{y + 5:.1f}" text-anchor="middle" font-size="15" font-weight="700" fill="{ink}">'
        f"{html.escape(text)}</text>"
    )


def _arrow(x1: float, y1: float, x2: float, y2: float) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="#111" stroke-width="2.2" marker-end="url(#arrow)"/>'
    )


def _draw_side(node: dict, x: float, path: set[int], parts: list[str]) -> None:
    if node["leaf"]:
        parts.append(_arrow(x, 196, x, 268))
        parts.append(_pill(x, 300, 120, 42, node["fill"], node["word"], node["ink"], node["id"] in path))
        return
    parts.append(_arrow(x, 196, x, 236))
    parts.append(
        f'<text x="{x:.1f}" y="258" text-anchor="middle" font-size="15" font-weight="700" fill="#111">'
        f"{html.escape(node['text'])}</text>"
    )
    yes_x, no_x = x - 115, x + 115
    parts.append(_arrow(x - 18, 272, yes_x, 318))
    parts.append(_arrow(x + 18, 272, no_x, 318))
    parts.append(_pill(yes_x, 336, 72, 32, "#3dcf6e", "YES", "#ffffff", node["yes"]["id"] in path))
    parts.append(_pill(no_x, 336, 72, 32, "#ff5d6c", "NO", "#ffffff", node["no"]["id"] in path))
    for child, child_x in ((node["yes"], yes_x), (node["no"], no_x)):
        parts.append(_arrow(child_x, 354, child_x, 400))
        parts.append(_pill(child_x, 428, 120, 42, child["fill"], child["word"], child["ink"], child["id"] in path))


def _tree_diagram(model, row: pd.Series) -> str:
    estimator = model.estimators_[0]
    values = row[list(FEATURES)].to_numpy(dtype=float)
    path = _path_nodes(estimator, values)
    root = _shallow(estimator, 0, 0)
    parts = [
        '<svg viewBox="0 0 980 490" width="100%" xmlns="http://www.w3.org/2000/svg" '
        'style="font-family:Arial,sans-serif;background:#f3f4f6">',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
        '<path d="M0,0 L7,3 L0,6 Z" fill="#111"/></marker></defs>',
        _pill(490, 36, 110, 40, "#3b6cff", "Tree 1", "#ffffff", False),
        f'<text x="490" y="108" text-anchor="middle" font-size="18" font-weight="700" fill="#111">'
        f"{html.escape(root['text'])}</text>",
        _arrow(470, 122, 275, 158),
        _arrow(510, 122, 705, 158),
        _pill(250, 178, 78, 34, "#3dcf6e", "YES", "#ffffff", root["yes"]["id"] in path),
        _pill(730, 178, 78, 34, "#ff5d6c", "NO", "#ffffff", root["no"]["id"] in path),
    ]
    _draw_side(root["yes"], 250, path, parts)
    _draw_side(root["no"], 730, path, parts)
    parts.append("</svg>")
    svg = "".join(parts)
    return f'<!DOCTYPE html><html><body style="margin:0;background:#f3f4f6">{svg}</body></html>'

def _show_trees(held: pd.DataFrame, model) -> None:
    st.subheader("What the 200 trees do with one cell")
    st.write(
        "Pick a held-out cell. The diagram is the first two splits of tree 1. "
        "Yes means the cell is at or under the cut. "
        "High is a listed share of 0.60 or more, medium is 0.40 up to 0.60, and low is below 0.40. "
        "The dark outline is the path of the cell you picked. "
        "The forest score below is the average of all 200 leaf shares."
    )
    labels = [
        f"{row.cell_id} · {row.municipality} · {'listed' if row.presence == 1 else 'empty'} · {row.score:.2f}"
        for row in held.itertuples(index=False)
    ]
    choice = st.selectbox("Held-out cell", labels)
    row = held.iloc[labels.index(choice)]
    shares = _tree_shares(model, row)
    st.iframe(
        _tree_diagram(model, row),
        height=520,
        alt="Tree 1 split diagram for the selected cell",
    )
    mostly_listed = int((shares >= 0.5).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Forest score", f"{float(row.score):.3f}")
    c2.metric("Trees whose leaf is at least half listed", f"{mostly_listed} of 200")
    c3.metric("Trees whose leaf is mostly empty", f"{200 - mostly_listed} of 200")
    bins = pd.cut(shares, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    counts = bins.value_counts().sort_index()
    frame = pd.DataFrame({"Leaf listed share": counts.index.astype(str), "Trees": counts.to_numpy()})
    st.altair_chart(_bar(frame, "Leaf listed share:N", "Trees:Q"), width="stretch")
    st.caption(
        f"The average of the 200 leaf shares is {shares.mean():.3f}, which is the forest score. "
        "A tree at 0.80 means four of five training cells in that leaf were listed."
    )


def _page_training() -> None:
    sample = load_sample()
    evaluation = load_evaluation() if REPORT_PATH.exists() and ROBUST_PATH.exists() else None
    listed = int(sample[PRESENCE_LABEL].sum())
    empty = int((sample[PRESENCE_LABEL] == 0).sum())
    st.subheader("How the forest is trained")
    st.markdown(
        f"""
1. Start with the rural cells that have a municipality.
2. Keep all **{listed}** cells that contain a listing, and draw **{empty}** empty cells with seed 42.
3. Hide whole municipalities. The forest never sees those towns while it is fit.
4. Fit **200 trees** on the remaining cells. Each tree looks at the six location numbers only.
        """
    )
    st.write(
        "Click the button to fit one held-out fold in front of the panel. "
        "That is one fifth of the test. The number to cite is the saved five-fold result, not this single fold."
    )
    if evaluation is not None:
        seed = evaluation["seed_42"]
        a, b, c = st.columns(3)
        a.metric("Five-fold ROC-AUC", f"{seed['roc_auc']:.3f}")
        b.metric("Five-fold PR-AUC", f"{seed['pr_auc']:.3f}")
        c.metric("Five-fold macro F1", f"{seed['macro_f1']:.3f}")
        matrix = seed["classification"]
        confusion = pd.DataFrame(
            {
                "Predicted empty": [matrix["true_negative"], matrix["false_negative"]],
                "Predicted listed": [matrix["false_positive"], matrix["true_positive"]],
            },
            index=["Actually empty", "Actually listed"],
        )
        st.dataframe(confusion, width="stretch")
        st.caption(
            f"On the {listed + empty} training cells: "
            f"{matrix['true_positive']} listings recovered, {matrix['false_negative']} missed, "
            f"{matrix['false_positive']} empty cells called listed, {matrix['true_negative']} empty cells called empty."
        )

    if st.button("Train one held-out fold", type="primary"):
        with st.spinner("Fitting 200 trees on the other municipalities..."):
            held, model = _train_one_fold(sample)
        st.session_state["demo_held"] = held
        st.session_state["demo_model"] = model
        st.session_state["demo_grid"] = "1km"

    held = st.session_state.get("demo_held")
    model = st.session_state.get("demo_model")
    if st.session_state.get("demo_grid") != "1km":
        held = None
        model = None
    if held is None or model is None:
        return
    towns = ", ".join(sorted(held["municipality"].astype(str).unique()))
    st.success(
        f"Held out {int(held['test_rows'].iloc[0])} cells in {towns}. "
        f"This fold's ROC-AUC is {float(held['held_out_roc_auc'].iloc[0]):.3f}. "
        f"The forest was fit on {int(held['train_rows'].iloc[0])} cells."
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Held-out scores")
        plot = held.copy()
        plot["class"] = plot[PRESENCE_LABEL].map({1: "Listed", 0: "Empty"})
        chart = (
            alt.Chart(plot)
            .mark_circle(size=60, opacity=0.8)
            .encode(
                x=alt.X("score:Q", title="Forest score"),
                y=alt.Y("municipality:N", title=None),
                color=alt.Color("class:N", title=None),
                tooltip=["cell_id", "municipality", "class", "score"],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, width="stretch")
    with right:
        st.subheader("This fold's ROC curve")
        _roc_chart(held)
    _show_trees(held, model)
    view = held[["cell_id", "municipality", PRESENCE_LABEL, "score"]].copy()
    view["actual"] = view[PRESENCE_LABEL].map({1: "listed", 0: "empty"})
    st.dataframe(
        view[["cell_id", "municipality", "actual", "score"]].sort_values("score", ascending=False),
        width="stretch",
        height=320,
    )


def main() -> None:
    st.set_page_config(page_title="Laguna 1 km site scores", layout="wide")
    st.title("Laguna rural cells, 1 km")
    st.caption("Six-feature random forest. A higher score means the cell looks more like rural cells that already have a listing.")
    if not _ready():
        st.error("The 1 km map or score tables are missing.")
        return
    page = st.sidebar.radio("Show", ["Map", "Data", "Training"])
    st.sidebar.write(
        "Show the map last. Start with Data so the panel sees the charts, "
        "then Training so they see one fold and the 200 trees, then the map."
    )
    if page == "Map":
        _page_map()
    elif page == "Data":
        _page_data()
    else:
        _page_training()


if __name__ == "__main__":
    main()
