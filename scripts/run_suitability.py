import json

from str_suitability import config
from str_suitability.suitability.hybrid_weights import rf_importance_weights
from str_suitability.suitability.inputs import load_scoring_input
from str_suitability.suitability.stage import compute_suitability


def read_model_summary(processed_dir) -> dict:
    return json.loads((processed_dir / "model_summary.json").read_text())


if __name__ == "__main__":
    grid = load_scoring_input(config.PROCESSED_DIR)
    model_summary = read_model_summary(config.PROCESSED_DIR)

    rfi = rf_importance_weights(
        [
            model_summary["revenue"]["importance"],
            model_summary["occupancy"]["importance"],
        ],
        list(config.COMPOSITE_INDICATOR_DIRECTIONS),
    )
    scored, report = compute_suitability(
        grid,
        directions=config.COMPOSITE_INDICATOR_DIRECTIONS,
        weighting=config.HYBRID_WEIGHTING,
        rfi_weights=rfi,
    )

    _, legacy = compute_suitability(grid)
    report["legacy_five_indicator_ewm"] = {
        "indicators": legacy["indicators"],
        "weights": legacy["weights"],
        "weight_sum": legacy["weight_sum"],
    }
    report["rfi_weights"] = {name: float(value) for name, value in rfi.items()}

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(config.PROCESSED_DIR / "grid_suitability.parquet", index=False)
    (config.PROCESSED_DIR / "suitability_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
