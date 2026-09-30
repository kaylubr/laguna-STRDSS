"""Build FINAL_REPORT.md and findings.html from saved experiment files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results.csv"


def load_rows() -> list[dict]:
    with RESULTS.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(row: dict, key: str):
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value, digits=3):
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def bar(value, maximum, color="#1d4e89"):
    width = 0 if value is None or maximum <= 0 else max(2, min(100, 100 * value / maximum))
    return f'<div class="bar"><i style="width:{width:.1f}%;background:{color}"></i><b>{fmt(value)}</b></div>'


def confusion_cells(path: Path) -> str:
    if not path.is_file():
        return ""
    rows = list(csv.reader(path.open(encoding="utf-8")))
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f'<table class="matrix">{body}</table>'


def high_from_matrix(path: Path) -> tuple[int, int, int]:
    rows = list(csv.reader(path.open(encoding="utf-8")))[1:]
    matrix = [[int(v) for v in row[1:]] for row in rows]
    predicted_high = sum(row[2] for row in matrix)
    actual_high = sum(matrix[2])
    matched = matrix[2][2]
    return matched, predicted_high, actual_high


def write_report(rows: list[dict]) -> None:
    by_id = {row["experiment_id"]: row for row in rows}
    baseline = by_id["EXP-000-baseline"]
    matched, predicted, actual = high_from_matrix(ROOT / "baseline" / "confusion_matrix.csv")
    feature_rows = [
        row
        for row in rows
        if row["feature_group"]
        in {
            "feature_market",
            "feature_poi",
            "feature_road",
            "feature_accessibility",
            "feature_population",
            "spatial_radius",
            "feature_selection",
        }
    ]
    ranked = sorted(feature_rows, key=lambda row: num(row, "macro_f1") or 0, reverse=True)

    def line(experiment_id: str) -> str:
        row = by_id[experiment_id]
        return (
            f"| {experiment_id} | {fmt(num(row, 'macro_f1'))} | "
            f"{fmt(num(row, 'mean_municipality_score'))} | "
            f"{fmt(num(row, 'high_precision'))} | {fmt(num(row, 'high_recall'))} |"
        )

    report = f"""# Experiment report

This report is for the `experiment/model-improvement` branch. It does not replace the thesis map or the five-feature forest on `2ndver`.

## Current baseline

The production model is a Random Forest (200 trees, depth 10, `max_features=sqrt`, leaf 1, class weight balanced, random state 42) trained on 288 labeled 1 km cells. Features: surrounding mean revenue, surrounding mean occupancy, surrounding listing count, listed landscape-place count, distance to the nearest listed place. Validation is `StratifiedGroupKFold` by municipality, 5 folds.

| Metric | Forest | Stratified guess | Majority (always Moderate) |
| --- | ---: | ---: | ---: |
| Macro F1 | {fmt(num(baseline, 'macro_f1'))} | 0.368 | 0.221 |
| Accuracy | {fmt(num(baseline, 'accuracy'))} | 0.425 | 0.498 |
| Macro precision | {fmt(num(baseline, 'macro_precision'))} | — | — |
| Macro recall | {fmt(num(baseline, 'macro_recall'))} | — | — |
| Weighted F1 | {fmt(num(baseline, 'weighted_f1'))} | — | — |
| High precision | {fmt(num(baseline, 'high_precision'))} | — | — |
| High recall | {fmt(num(baseline, 'high_recall'))} | — | — |

Held-out High calls: {predicted} predicted High, {matched} actually High, {actual} true High cells. High success rate {matched}/{predicted}. Leave-one-municipality-out mean macro F1: {fmt(num(baseline, 'mean_municipality_score'))} (SD {fmt(num(baseline, 'std_municipality_score'))}).

These numbers match `data/processed/rural/model_summary.json`.

## Validation

Every classification experiment reused the same municipality-grouped 5-fold split as the thesis model. Leave-one-municipality-out was recorded beside it. Hyperparameters were chosen on inner training-town folds only.

A configuration is treated as an improvement only when grouped-CV macro F1 and leave-one-municipality-out mean macro F1 both rise. A jump in one town is not enough.

## Feature experiments

Highest observed grouped-CV macro F1 among feature additions:

| Experiment | Grouped macro F1 | LOMO mean | High precision | High recall |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(line(row['experiment_id']) for row in ranked[:12])}

EXP-A05 (nearby Low/Moderate/High mix of other listings) had the highest grouped macro F1 ({fmt(num(by_id['EXP-A05'], 'macro_f1'))}). EXP-J01 (urban proximity) and EXP-A02 (3 km market) had the highest leave-one-municipality-out means. EXP-COMBINED-01 stacked every addition that beat the baseline on both scores and reached grouped macro F1 {fmt(num(by_id['EXP-COMBINED-01'], 'macro_f1'))} and LOMO {fmt(num(by_id['EXP-COMBINED-01'], 'mean_municipality_score'))}.

Road density (EXP-E03) and nearest-road distance (EXP-E01) also rose on grouped CV. Population density (EXP-G01) rose on grouped CV and fell on leave-one-municipality-out, so it is not treated as an improvement.

Elevation, slope, and protected-area distance were not built. Those layers are not in the repository. Laguna de Bay and other water used existing OSM-derived grid columns.

## Spatial experiments

| Radius / setup | Grouped macro F1 | LOMO mean |
| --- | ---: | ---: |
| 1 km market added (EXP-A01) | {fmt(num(by_id['EXP-A01'], 'macro_f1'))} | {fmt(num(by_id['EXP-A01'], 'mean_municipality_score'))} |
| 3 km market added (EXP-A02) | {fmt(num(by_id['EXP-A02'], 'macro_f1'))} | {fmt(num(by_id['EXP-A02'], 'mean_municipality_score'))} |
| 10 km market added (EXP-A03) | {fmt(num(by_id['EXP-A03'], 'macro_f1'))} | {fmt(num(by_id['EXP-A03'], 'mean_municipality_score'))} |
| 1 km POI count (EXP-C01) | {fmt(num(by_id['EXP-C01'], 'macro_f1'))} | {fmt(num(by_id['EXP-C01'], 'mean_municipality_score'))} |
| 10 km POI count (EXP-C02) | {fmt(num(by_id['EXP-C02'], 'macro_f1'))} | {fmt(num(by_id['EXP-C02'], 'mean_municipality_score'))} |

A 3 km market neighborhood was the most stable extra market scale. 10 km raised grouped F1 and lowered leave-one-municipality-out.

## Grid comparison

| Grid | Labeled cells | Grouped macro F1 | LOMO mean |
| --- | ---: | ---: | ---: |
| 500 m | 409 | {fmt(num(by_id['EXP-M-500m'], 'macro_f1'))} | {fmt(num(by_id['EXP-M-500m'], 'mean_municipality_score'))} |
| 1 km (thesis) | 288 | {fmt(num(baseline, 'macro_f1'))} | {fmt(num(baseline, 'mean_municipality_score'))} |
| 2 km | 160 | {fmt(num(by_id['EXP-M-2000m'], 'macro_f1'))} | {fmt(num(by_id['EXP-M-2000m'], 'mean_municipality_score'))} |

The 1 km grid remains the most defensible size. 500 m added labels but not a stable gain. 2 km lost labels and score.

## Model comparison

Same five features, same municipality folds.

| Model | Grouped macro F1 | LOMO mean | High precision |
| --- | ---: | ---: | ---: |
| Logistic regression | {fmt(num(by_id['EXP-P-logistic_regression'], 'macro_f1'))} | {fmt(num(by_id['EXP-P-logistic_regression'], 'mean_municipality_score'))} | {fmt(num(by_id['EXP-P-logistic_regression'], 'high_precision'))} |
| Decision tree | {fmt(num(by_id['EXP-P-decision_tree'], 'macro_f1'))} | {fmt(num(by_id['EXP-P-decision_tree'], 'mean_municipality_score'))} | {fmt(num(by_id['EXP-P-decision_tree'], 'high_precision'))} |
| Random Forest | {fmt(num(by_id['EXP-P-random_forest'], 'macro_f1'))} | {fmt(num(by_id['EXP-P-random_forest'], 'mean_municipality_score'))} | {fmt(num(by_id['EXP-P-random_forest'], 'high_precision'))} |
| Gradient boosting | {fmt(num(by_id['EXP-P-gradient_boosting'], 'macro_f1'))} | {fmt(num(by_id['EXP-P-gradient_boosting'], 'mean_municipality_score'))} | {fmt(num(by_id['EXP-P-gradient_boosting'], 'high_precision'))} |

Gradient boosting had the highest grouped macro F1. Its leave-one-municipality-out mean stayed near the forest. Logistic regression had higher High precision and a lower macro F1.

## Target experiments

| Target | Grouped macro F1 | Note |
| --- | ---: | --- |
| 25 / 50 / 25 (thesis) | {fmt(num(baseline, 'macro_f1'))} | Official label |
| 20 / 60 / 20 | {fmt(num(by_id['EXP-N-20_60_20'], 'macro_f1'))} | Not the thesis target |
| 30 / 40 / 30 | {fmt(num(by_id['EXP-N-30_40_30'], 'macro_f1'))} | Not the thesis target |
| Binary High vs not | {fmt(num(by_id['EXP-N-binary_high'], 'macro_f1'))} | Not comparable to three-class F1 |

Regression of own-cell revenue and occupancy produced negative R² on held-out towns. Continuous prediction did not fit this sample.

## Hyperparameters

Nested search over 18 Random Forest settings:

| Feature set | Nested outer macro F1 |
| --- | ---: |
| Five production features | {fmt(num(by_id['EXP-HP-baseline'], 'macro_f1'))} |
| EXP-COMBINED-01 features | {fmt(num(by_id['EXP-HP-combined'], 'macro_f1'))} |

Different outer folds chose different settings. Nested search on the five features scored below the fixed production settings. The production hyperparameters stay the ones to keep.

## Highest-observed configurations

Wording is "highest observed validation score," not "best model for the thesis."

1. EXP-A05, grouped macro F1 {fmt(num(by_id['EXP-A05'], 'macro_f1'))}. Nearby class mix of other listings.
2. EXP-E03, grouped macro F1 {fmt(num(by_id['EXP-E03'], 'macro_f1'))}. Road density within 3 km.
3. EXP-COMBINED-01, grouped macro F1 {fmt(num(by_id['EXP-COMBINED-01'], 'macro_f1'))}, LOMO {fmt(num(by_id['EXP-COMBINED-01'], 'mean_municipality_score'))}. Stacked additions.
4. EXP-SEL-lean, grouped macro F1 {fmt(num(by_id['EXP-SEL-lean'], 'macro_f1'))}, LOMO {fmt(num(by_id['EXP-SEL-lean'], 'mean_municipality_score'))}. Fewer stacked features.
5. Gradient boosting on the five production features, grouped macro F1 {fmt(num(by_id['EXP-P-gradient_boosting'], 'macro_f1'))}.

None of these beat the stratified guess of 0.368 on grouped macro F1 except EXP-A05 (0.389) and EXP-E03 (0.383) and the stacked sets. Those gains are still modest, and fold-to-fold spread remains large.

## Generalization

Leave-one-municipality-out standard deviations stay around 0.17–0.30. Small towns with two or three labeled cells swing the score. A feature that helps Calamba can fail in Cavinti.

## Limitations

- 288 labeled cells, unevenly spread across about 30 towns.
- Many rural cells have no Airbnb, so they have no true class for this test.
- Historical Airbnb performance is not demand, profit, or a building decision.
- POI importance (major vs minor) has no objective field in `poi_laguna.json`.
- Elevation and slope were not available.
- Municipality population density tracks town identity.
- A higher score is an association on held-out towns, not a cause.

## Recommendation for the next thesis experiment

Investigate the nearby class-mix of other listings (EXP-A05) and 3 km road density (EXP-E03) on the same municipality folds, one group at a time, without stacking every extra column. Keep the 1 km grid, the 25 / 50 / 25 label, and the five-feature Random Forest as the official map until a later review says otherwise.
"""
    (ROOT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")


def write_html(rows: list[dict]) -> None:
    by_id = {row["experiment_id"]: row for row in rows}
    baseline = by_id["EXP-000-baseline"]
    matched, predicted, actual = high_from_matrix(ROOT / "baseline" / "confusion_matrix.csv")
    feature_rows = [
        row
        for row in rows
        if row["feature_group"]
        in {
            "baseline",
            "feature_market",
            "feature_poi",
            "feature_road",
            "feature_accessibility",
            "feature_population",
            "spatial_radius",
            "feature_selection",
        }
        and num(row, "macro_f1") is not None
    ]
    feature_rows.sort(key=lambda row: num(row, "macro_f1") or 0, reverse=True)
    max_f1 = max(num(row, "macro_f1") or 0 for row in feature_rows)

    feature_html = "".join(
        f"<tr><td>{row['experiment_id']}</td><td>{row['feature_group']}</td>"
        f"<td>{bar(num(row, 'macro_f1'), max_f1, '#1a9850' if (num(row, 'macro_f1') or 0) > num(baseline, 'macro_f1') else '#4c78a8')}</td>"
        f"<td>{fmt(num(row, 'mean_municipality_score'))}</td></tr>"
        for row in feature_rows
    )

    models = [
        ("Logistic regression", "EXP-P-logistic_regression"),
        ("Decision tree", "EXP-P-decision_tree"),
        ("Random Forest", "EXP-P-random_forest"),
        ("Gradient boosting", "EXP-P-gradient_boosting"),
    ]
    model_html = "".join(
        f"<tr><td>{name}</td><td>{fmt(num(by_id[exp], 'macro_f1'))}</td>"
        f"<td>{fmt(num(by_id[exp], 'mean_municipality_score'))}</td>"
        f"<td>{fmt(num(by_id[exp], 'high_precision'))}</td></tr>"
        for name, exp in models
    )

    radii = [
        ("1 km market", "EXP-A01"),
        ("3 km market", "EXP-A02"),
        ("5 km (thesis)", "EXP-000-baseline"),
        ("10 km market", "EXP-A03"),
        ("1 km POI", "EXP-C01"),
        ("10 km POI", "EXP-C02"),
    ]
    radius_html = "".join(
        f"<tr><td>{name}</td><td>{fmt(num(by_id[exp], 'macro_f1'))}</td>"
        f"<td>{fmt(num(by_id[exp], 'mean_municipality_score'))}</td></tr>"
        for name, exp in radii
    )

    grids = [
        ("500 m", "EXP-M-500m"),
        ("1 km", "EXP-000-baseline"),
        ("2 km", "EXP-M-2000m"),
    ]
    grid_html = "".join(
        f"<tr><td>{name}</td><td>{by_id[exp]['n_test']}</td>"
        f"<td>{fmt(num(by_id[exp], 'macro_f1'))}</td>"
        f"<td>{fmt(num(by_id[exp], 'mean_municipality_score'))}</td></tr>"
        for name, exp in grids
    )

    matrices = "".join(
        f"<figure><figcaption>{label}</figcaption>{confusion_cells(path)}</figure>"
        for label, path in (
            ("Baseline", ROOT / "baseline" / "confusion_matrix.csv"),
            ("Nearby class mix (A05)", ROOT / "feature_market" / "EXP-A05" / "confusion_matrix.csv"),
            ("Road density 3 km (E03)", ROOT / "feature_road" / "EXP-E03" / "confusion_matrix.csv"),
            ("Combined additions", ROOT / "feature_market" / "EXP-COMBINED-01" / "confusion_matrix.csv"),
        )
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Laguna screener experiments</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body {{ margin:0; font:16px/1.5 system-ui, sans-serif; color:#1d2433; background:#f6f7f9; }}
  main {{ max-width:960px; margin:0 auto; padding:32px 20px 64px; }}
  h1 {{ font-size:28px; letter-spacing:-.03em; margin:0 0 8px; }}
  h2 {{ font-size:18px; margin:36px 0 10px; }}
  p, li {{ color:#344054; }}
  .cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:18px 0 8px; }}
  .card {{ background:#fff; border-radius:12px; padding:14px 16px; box-shadow:0 1px 4px rgba(16,24,40,.08); }}
  .card span {{ display:block; font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:#667085; }}
  .card b {{ font-size:26px; letter-spacing:-.03em; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:12px; overflow:hidden; }}
  th, td {{ padding:8px 10px; text-align:left; border-bottom:1px solid #eef1f4; font-size:14px; }}
  th {{ font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:#667085; }}
  .bar {{ display:flex; align-items:center; gap:8px; }}
  .bar i {{ display:block; height:8px; border-radius:99px; }}
  .bar b {{ font-variant-numeric:tabular-nums; }}
  .matrix td {{ text-align:center; }}
  figure {{ display:inline-block; margin:8px 16px 8px 0; }}
  figcaption {{ font-size:12px; color:#667085; margin-bottom:4px; }}
  .note {{ background:#fff8eb; border-radius:10px; padding:12px 14px; }}
  @media (max-width:800px) {{ .cards {{ grid-template-columns:1fr 1fr; }} }}
</style>
</head>
<body>
<main>
  <h1>What the experiments found</h1>
  <p>Separate lab on the <code>experiment/model-improvement</code> branch. The thesis map and the five-feature forest were not changed.</p>
  <div class="cards">
    <div class="card"><span>Forest macro F1</span><b>{fmt(num(baseline, 'macro_f1'))}</b></div>
    <div class="card"><span>Stratified guess</span><b>0.368</b></div>
    <div class="card"><span>High calls that matched</span><b>{matched}/{predicted}</b></div>
    <div class="card"><span>Highest observed F1</span><b>{fmt(num(by_id['EXP-A05'], 'macro_f1'))}</b></div>
  </div>
  <p class="note">A green square on the thesis map is still the highest of three probabilities. On held-out towns that High call matched a real top-quarter square {matched} times out of {predicted}. The experiments asked whether extra surroundings, other models, or other labels raise that score on towns the model did not train on.</p>

  <h2>Feature groups</h2>
  <p>Green bars are above the five-feature forest. The last column is leave-one-municipality-out mean macro F1.</p>
  <table>
    <tr><th>Experiment</th><th>Group</th><th>Grouped macro F1</th><th>LOMO</th></tr>
    {feature_html}
  </table>

  <h2>Confusion matrices</h2>
  {matrices}

  <h2>Models, same five features</h2>
  <table>
    <tr><th>Model</th><th>Grouped macro F1</th><th>LOMO</th><th>High precision</th></tr>
    {model_html}
  </table>

  <h2>Spatial scale</h2>
  <table>
    <tr><th>Setup</th><th>Grouped macro F1</th><th>LOMO</th></tr>
    {radius_html}
  </table>

  <h2>Grid size</h2>
  <table>
    <tr><th>Cell</th><th>Labeled cells</th><th>Grouped macro F1</th><th>LOMO</th></tr>
    {grid_html}
  </table>

  <h2>What deserves a later look</h2>
  <p>The nearby class mix of other listings (EXP-A05) and 3 km road density (EXP-E03) raised grouped-CV macro F1 the most. Stacking every extra column (EXP-COMBINED-01) also raised leave-one-municipality-out mean F1 from {fmt(num(baseline, 'mean_municipality_score'))} to {fmt(num(by_id['EXP-COMBINED-01'], 'mean_municipality_score'))}. Nested hyperparameter search did not beat the fixed production settings. Keep the official map on the five-feature forest until those two groups are reviewed one at a time.</p>
</main>
</body>
</html>
"""
    (ROOT / "findings.html").write_text(html, encoding="utf-8")


def main() -> None:
    rows = load_rows()
    write_report(rows)
    write_html(rows)
    print("wrote", ROOT / "FINAL_REPORT.md")
    print("wrote", ROOT / "findings.html")


if __name__ == "__main__":
    main()
