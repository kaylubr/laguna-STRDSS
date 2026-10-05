"""Ten negative-sample draws and one all-empty run of the six-feature forest.

Uses the same forest, the same municipality-grouped outer split, the same
inner Youden cutoff, and the same 20 permutation repeats as
``evaluate_forest``. Writes a new results directory and does not replace the
locked site-candidate files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from str_suitability import config
from str_suitability.modeling.site_candidate import (
    CELL_ID_COLUMN,
    FEATURES,
    PRESENCE_LABEL,
    build_training_sample,
    evaluate_forest,
    load_cells,
)
from str_suitability.rural.classify import RURAL_CELL

SEEDS = (7, 13, 21, 42, 99, 101, 202, 303, 404, 505)
OUTPUT_DIR = config.PROJECT_ROOT / "results" / "site_candidate_robustness"


def full_population(cells: pd.DataFrame) -> pd.DataFrame:
    """Every named rural cell: 110 listed cells and all 931 empty cells."""
    rural = cells.loc[cells["cell_class"].eq(RURAL_CELL)].copy()
    eligible = rural.loc[rural["municipality"].notna() & rural["municipality"].astype(str).ne("")]
    sample = eligible.reset_index(drop=True)
    sample[PRESENCE_LABEL] = (sample["listings_in_cell"] >= 1).astype(int)
    return sample


def _draw_record(seed: int | None, population: str, report: dict[str, object]) -> dict[str, object]:
    return {
        "negative_sampling_seed": seed,
        "population": population,
        "training_rows": report["training_rows"],
        "training_positives": report["training_positives"],
        "training_negatives": report["training_negatives"],
        "validation_scheme": report["validation_scheme"],
        "permutation_repeats": report["permutation_repeats"],
        "folds": report["folds"],
        "metrics": report["metrics"],
        "permutation_importance": report["permutation_importance"],
    }


def _across_draws(records: list[dict[str, object]], metric: str) -> dict[str, float]:
    values = [float(record["metrics"][metric]["mean"]) for record in records]
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "values": values,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cells, _place_report = load_cells()
    draws = []
    for seed in SEEDS:
        sample = build_training_sample(cells, negative_seed=seed)
        _oof, report = evaluate_forest(sample)
        record = _draw_record(seed, "balanced_negative_sample", report)
        draws.append(record)
        print(
            f"seed {seed}: ROC-AUC {report['metrics']['roc_auc']['mean']:.6f} "
            f"PR-AUC {report['metrics']['pr_auc']['mean']:.6f}",
            flush=True,
        )

    population = full_population(cells)
    assert int((population[PRESENCE_LABEL] == 1).sum()) == 110
    assert int((population[PRESENCE_LABEL] == 0).sum()) == 931
    assert len(population) == 1041
    oof, full_report = evaluate_forest(population)
    assert len(oof) == 1041
    assert oof["random_forest_raw"].notna().all()
    assert set(oof[CELL_ID_COLUMN].astype(str)) == set(population[CELL_ID_COLUMN].astype(str))
    all_empty = _draw_record(None, "all_empty_cells", full_report)
    oof_path = OUTPUT_DIR / "all_empty_oof_predictions.parquet"
    oof.to_parquet(oof_path, index=False)

    roc = _across_draws(draws, "roc_auc")
    pr = _across_draws(draws, "pr_auc")
    all_roc = float(full_report["metrics"]["roc_auc"]["mean"])
    all_pr = float(full_report["metrics"]["pr_auc"]["mean"])
    summary = {
        "features": list(FEATURES),
        "forest_random_state": int(config.RANDOM_STATE),
        "outer_split_random_state": int(config.RANDOM_STATE),
        "seeds": list(SEEDS),
        "draws": draws,
        "roc_auc_across_draws": roc,
        "pr_auc_across_draws": pr,
        "all_empty": all_empty,
        "all_empty_against_draw_range": {
            "roc_auc_mean": all_roc,
            "pr_auc_mean": all_pr,
            "roc_auc_inside_ten_draw_range": bool(roc["min"] <= all_roc <= roc["max"]),
            "pr_auc_inside_ten_draw_range": bool(pr["min"] <= all_pr <= pr["max"]),
            "oof_rows": int(len(oof)),
            "oof_path": str(oof_path.relative_to(config.PROJECT_ROOT)).replace("\\", "/"),
        },
    }
    report_path = OUTPUT_DIR / "robustness_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"ten-draw ROC-AUC mean {roc['mean']:.6f} range {roc['min']:.6f} to {roc['max']:.6f}",
        flush=True,
    )
    print(
        f"ten-draw PR-AUC mean {pr['mean']:.6f} range {pr['min']:.6f} to {pr['max']:.6f}",
        flush=True,
    )
    print(
        f"all-empty ROC-AUC {all_roc:.6f} PR-AUC {all_pr:.6f} "
        f"oof rows {len(oof)}",
        flush=True,
    )
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
