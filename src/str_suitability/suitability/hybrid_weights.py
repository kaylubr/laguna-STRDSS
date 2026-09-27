import pandas as pd

from str_suitability.config import BLEND_RATIO
from str_suitability.suitability.entropy_weights import WEIGHT_SUM_TOLERANCE

RFI_WEIGHT_NAME = "rf_importance_weight"
EQUAL_WEIGHT_NAME = "equal_weight"
HYBRID_WEIGHT_NAME = "hybrid_weight"


def equal_weights(columns) -> pd.Series:
    columns = list(columns)
    assert len(columns) >= 1, "equal weighting needs at least one indicator"
    return pd.Series(1.0 / len(columns), index=columns, name=EQUAL_WEIGHT_NAME)


def _normalised_model_importance(
    rows: list[dict], indicators: list[str]
) -> pd.Series | None:
    values = pd.Series(
        {row["feature"]: max(0.0, float(row["importance_mean"])) for row in rows}
    )
    values = values.reindex(indicators).fillna(0.0)
    total = float(values.sum())
    if total <= 0.0:
        return None
    return values / total


def rf_importance_weights(
    importance_tables: list[list[dict]], indicators: list[str]
) -> pd.Series:
    indicators = list(indicators)
    columns = {}
    for position, rows in enumerate(importance_tables):
        normalised = _normalised_model_importance(rows, indicators)
        if normalised is not None:
            columns[f"model_{position}"] = normalised
    if not columns:
        return pd.Series(0.0, index=indicators, name=RFI_WEIGHT_NAME)

    combined = pd.concat(columns, axis=1).mean(axis=1)
    total = float(combined.sum())
    if total > 0.0:
        combined = combined / total
    return combined.rename(RFI_WEIGHT_NAME)


def hybrid_weights(
    entropy: pd.Series, rf_importance: pd.Series, blend_ratio: float = BLEND_RATIO
) -> pd.Series:
    assert 0.0 <= blend_ratio <= 1.0, f"blend ratio must lie in [0, 1], got {blend_ratio}"

    aligned = rf_importance.reindex(entropy.index)
    fallback = aligned.where(aligned.notna(), entropy)
    blended = blend_ratio * entropy + (1.0 - blend_ratio) * fallback
    total = float(blended.sum())
    assert total > 0.0, "the hybrid weights sum to zero"

    blended = blended / total
    assert abs(float(blended.sum()) - 1.0) <= WEIGHT_SUM_TOLERANCE, (
        f"hybrid weights sum to {blended.sum()}, expected 1"
    )
    assert (blended >= 0.0).all(), "hybrid weights must not be negative"
    return blended.rename(HYBRID_WEIGHT_NAME)


def bloc_weight(weights: pd.Series, members: list[str]) -> float:
    present = [name for name in members if name in weights.index]
    return float(weights[present].sum())
