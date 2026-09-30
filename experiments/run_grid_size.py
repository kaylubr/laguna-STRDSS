"""Grid-size sensitivity (spec group M): rebuild the grid at 500 m and 2 km
(config.SENSITIVITY_CELL_SIZES_M already anticipates exactly these two sizes)
and rerun the SAME five-feature model and validation. A grid size is only worth
recommending if performance is consistent across held-out municipalities, not
just higher on one split.

Run with: uv run python -m experiments.run_grid_size
"""

from __future__ import annotations

from pathlib import Path

from str_suitability import config
from str_suitability.modeling.location_classifier import CLASSIFIER_FEATURES, MUNICIPALITY_COLUMN, PERFORMANCE_CLASS

from experiments.lab import data as lab_data
from experiments.lab.common import append_result_rows, evaluate_grouped, evaluate_lomo, results_row, save_experiment
from experiments.lab.grid_size import build_alt_grid, labeled_frame_for_grid

EXPERIMENTS_DIR = Path(__file__).parent
RESULTS_CSV = EXPERIMENTS_DIR / "results.csv"
BASELINE_FEATURES = list(CLASSIFIER_FEATURES)


def run_grid_size(cell_size_m: int, active_listings, places) -> dict:
    grid, grid_report = build_alt_grid(cell_size_m)
    labeled, label_report = labeled_frame_for_grid(grid, active_listings, places)
    exp_id = f"EXP-M-{cell_size_m}m"

    grouped = evaluate_grouped(labeled, BASELINE_FEATURES, PERFORMANCE_CLASS, MUNICIPALITY_COLUMN)
    lomo = evaluate_lomo(labeled, BASELINE_FEATURES, PERFORMANCE_CLASS, MUNICIPALITY_COLUMN)

    cfg = {
        "experiment_id": exp_id,
        "cell_size_m": cell_size_m,
        "grid_cells": grid_report["cells"],
        "labeled_cells": label_report["labeled_cells"],
        "class_counts": label_report["class_counts"],
        "n_municipalities": int(labeled[MUNICIPALITY_COLUMN].nunique()),
        "features": BASELINE_FEATURES,
        "grid_report": grid_report,
    }
    notes = (
        f"# {exp_id}\n\n"
        f"Same five features and same model settings as the production 1 km grid, but the "
        f"grid itself is rebuilt at {cell_size_m} m using the generic build_grid() function "
        f"(str_suitability.spatial.grid), which config.py already anticipates via "
        f"SENSITIVITY_CELL_SIZES_M = {config.SENSITIVITY_CELL_SIZES_M}.\n\n"
        f"Grid cells: {grid_report['cells']}. Labeled cells: {label_report['labeled_cells']} "
        f"(vs. 288 at 1 km). Municipalities represented: {labeled[MUNICIPALITY_COLUMN].nunique()}.\n\n"
        f"Grouped-CV macro F1: {grouped['macro_f1']:.4f}. "
        f"LOMO mean macro F1: {lomo['macro_f1']:.4f}, std {lomo['cv']['macro_f1']['std']:.4f} "
        f"across {lomo['n_folds']} municipality folds.\n\n"
        "A smaller cell holds fewer listings per cell (sparser surrounding-market signal per "
        "cell but more cells), and a larger cell pools more listings into fewer, coarser "
        "cells. Both change the labeled sample size, which changes the difficulty of the "
        "validation problem itself, not just the model.\n"
    )
    save_experiment(EXPERIMENTS_DIR / "grid_size" / exp_id, cfg, grouped, BASELINE_FEATURES, notes, lomo_metrics=lomo)
    row = results_row(
        exp_id, "grid_size", "random_forest", grouped, len(BASELINE_FEATURES), lomo_metrics=lomo,
        grid_size=f"{cell_size_m}m",
        notes=f"{label_report['labeled_cells']} labeled cells at {cell_size_m}m vs 288 at 1000m.",
    )
    print(
        f"{exp_id:16s} cells={grid_report['cells']:5d} labeled={label_report['labeled_cells']:4d} "
        f"macro_f1={grouped['macro_f1']:.4f} lomo_mean={lomo['macro_f1']:.4f} lomo_std={lomo['cv']['macro_f1']['std']:.4f}"
    )
    return row


def main() -> None:
    active_listings = lab_data.load_active_listings()
    places = lab_data.load_places_with_category()
    rows = [run_grid_size(size, active_listings, places) for size in config.SENSITIVITY_CELL_SIZES_M]
    append_result_rows(RESULTS_CSV, rows)
    print(f"Wrote {len(rows)} rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
