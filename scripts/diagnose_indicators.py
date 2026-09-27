import json

import geopandas as gpd
import numpy as np
import pandas as pd

from str_suitability import config
from str_suitability.features.accessibility import distance_to_nearest_km
from str_suitability.rural.classify import CELL_CLASS_COLUMN, CELL_ID_COLUMN, RURAL_CELL
from str_suitability.suitability.entropy_weights import (
    entropy_values,
    indicator_proportions,
    weights_from_normalized,
)
from str_suitability.suitability.hybrid_weights import bloc_weight, hybrid_weights, rf_importance_weights
from str_suitability.suitability.inputs import load_scoring_input
from str_suitability.suitability.normalize import (
    indicator_from_normalized,
    minimum_maximum_normalize,
    normalized_column,
)

DIAGNOSTICS_PATH = config.PROCESSED_DIR / "indicator_diagnostics.json"
RURAL_CLASSIFICATION_FILENAME = "grid_rural_classification.parquet"
PREDICTION_COLUMNS = ("predicted_revenue", "predicted_occupancy")


def load_rural_cells(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    classification = pd.read_parquet(config.RURAL_PROCESSED_DIR / RURAL_CLASSIFICATION_FILENAME)
    rural_ids = set(
        classification.loc[classification[CELL_CLASS_COLUMN] == RURAL_CELL, CELL_ID_COLUMN]
    )
    return grid[grid[CELL_ID_COLUMN].isin(rural_ids)].copy()


def per_indicator(frame: pd.DataFrame, directions: dict[str, str]) -> dict[str, dict]:
    normalized = minimum_maximum_normalize(frame, directions=directions)
    ewm = weights_from_normalized(normalized)
    entropies = entropy_values(indicator_proportions(normalized))
    records = {}
    for indicator in directions:
        values = pd.to_numeric(frame[indicator], errors="raise").astype(float)
        normalised = normalized[normalized_column(indicator)]
        records[indicator] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "zero_share": float((values == 0).mean()),
            "skew": float(values.skew()),
            "normalised_dispersion": float(normalised.std()),
            "entropy": float(entropies[normalized_column(indicator)]),
            "ewm_weight": float(ewm[normalized_column(indicator)]),
        }
    return records


def bloc_sum(records: dict[str, dict], members) -> float:
    return float(sum(records[name]["ewm_weight"] for name in members if name in records))


def rfi_report(model_summary: dict, indicators) -> dict[str, float]:
    revenue = {row["feature"]: float(row["importance_mean"]) for row in model_summary["revenue"]["importance"]}
    occupancy = {row["feature"]: float(row["importance_mean"]) for row in model_summary["occupancy"]["importance"]}
    return {
        indicator: {
            "revenue": revenue.get(indicator),
            "occupancy": occupancy.get(indicator),
        }
        for indicator in indicators
    }


def poblacion_direction(grid: pd.DataFrame) -> dict[str, object]:
    distances = pd.to_numeric(grid["distance_to_poblacion"], errors="raise").astype(float)
    result = {}
    for column in PREDICTION_COLUMNS:
        values = pd.to_numeric(grid[column], errors="raise").astype(float)
        result[f"corr_with_{column}"] = float(np.corrcoef(distances, values)[0, 1])
    correlations = [result[f"corr_with_{column}"] for column in PREDICTION_COLUMNS]
    mean_correlation = float(np.mean(correlations))
    result["mean_correlation"] = mean_correlation
    result["recommended_direction"] = "negative" if mean_correlation <= 0.0 else "positive"
    return result


def transform_scenarios(
    frame: pd.DataFrame, directions: dict[str, str], rfi: pd.Series
) -> dict[str, dict[str, float]]:
    bloc_columns = [normalized_column(name) for name in config.POI_BLOC_INDICATORS]

    def scenarios(candidate: pd.DataFrame) -> dict[str, float]:
        entropy = weights_from_normalized(
            minimum_maximum_normalize(candidate, directions=directions)
        )
        aligned = pd.Series(
            rfi.reindex([indicator_from_normalized(column) for column in entropy.index]).to_numpy(),
            index=entropy.index,
        )
        hybrid = hybrid_weights(entropy, aligned)
        return {
            "ewm_bloc_weight": bloc_weight(entropy, bloc_columns),
            "hybrid_bloc_weight": bloc_weight(hybrid, bloc_columns),
        }

    zero_inflated = frame.copy()
    for name in config.POI_BLOC_INDICATORS:
        zero_inflated[name] = np.log1p(zero_inflated[name].astype(float))
    all_indicators = frame.copy()
    for name in directions:
        all_indicators[name] = np.log1p(all_indicators[name].astype(float))

    return {
        "raw": scenarios(frame),
        "log1p_zero_inflated": scenarios(zero_inflated),
        "log1p_all": scenarios(all_indicators),
    }


def nearby_report(grid: gpd.GeoDataFrame, processed_dir) -> dict[str, object]:
    observations = pd.read_parquet(processed_dir / "training_observations.parquet")
    distances = distance_to_nearest_km(
        grid["longitude"].to_numpy(),
        grid["latitude"].to_numpy(),
        observations["longitude"].to_numpy(),
        observations["latitude"].to_numpy(),
    )
    series = pd.Series(distances)
    return {
        "median_km": float(series.median()),
        "p75_km": float(series.quantile(0.75)),
        "p90_km": float(series.quantile(0.90)),
        "share_within_1km": float((series <= 1.0).mean()),
        "share_within_5km": float((series <= 5.0).mean()),
        "share_within_10km": float((series <= 10.0).mean()),
    }


def main() -> dict[str, object]:
    directions = dict(config.COMPOSITE_INDICATOR_DIRECTIONS)
    grid = load_scoring_input(config.PROCESSED_DIR)
    rural = load_rural_cells(grid)
    model_summary = json.loads((config.PROCESSED_DIR / "model_summary.json").read_text())

    province_records = per_indicator(grid, directions)
    rural_records = per_indicator(rural, directions)
    rfi = rf_importance_weights(
        [
            model_summary["revenue"]["importance"],
            model_summary["occupancy"]["importance"],
        ],
        list(directions),
    )

    diagnostics = {
        "indicators": list(directions),
        "province": {
            "cells": int(len(grid)),
            "per_indicator": province_records,
            "poi_bloc_weight": bloc_sum(province_records, config.POI_BLOC_INDICATORS),
        },
        "rural": {
            "cells": int(len(rural)),
            "per_indicator": rural_records,
            "poi_bloc_weight": bloc_sum(rural_records, config.POI_BLOC_INDICATORS),
        },
        "water_split": {
            "corr_lake_vs_other": float(
                grid["distance_to_laguna_de_bay"].corr(grid["distance_to_other_water"])
            ),
            "cells_differing_over_1km": int(
                (grid["distance_to_laguna_de_bay"] - grid["distance_to_other_water"])
                .abs()
                .gt(1.0)
                .sum()
            ),
        },
        "poblacion": {
            "province": poblacion_direction(grid),
            "rural": poblacion_direction(rural),
        },
        "rfi": rfi_report(model_summary, directions),
        "transform_scenarios": transform_scenarios(grid, directions, rfi),
        "nearby_training": nearby_report(grid, config.PROCESSED_DIR),
        "zero_inflation": {
            indicator: {
                "province_zero_share": province_records[indicator]["zero_share"],
                "rural_zero_share": rural_records[indicator]["zero_share"],
            }
            for indicator in directions
        },
    }
    DIAGNOSTICS_PATH.write_text(json.dumps(diagnostics, indent=2, default=str))
    return diagnostics


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, default=str))
    print(f"\nartifacts: {DIAGNOSTICS_PATH}")
