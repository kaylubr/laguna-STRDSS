"""Random Forest similarity screening for rural STR candidate cells."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from str_suitability import config
from str_suitability.features.context import POBLACION_DISTANCE_COLUMN
from str_suitability.features.road_distance import (
    DISTANCE_CRS,
    GRID_ROAD_DISTANCE_PATH,
    LAGUNA_ROADS_PATH,
    ROAD_DISTANCE_KM,
    _all_mapped_roads,
    _assert_metre_crs,
)
from str_suitability.rural.classify import CELL_ID_COLUMN, RURAL_CELL
from str_suitability.rural.site_score import (
    DISTANCE_COLUMN,
    LATITUDE_COLUMN,
    LISTED_PLACES_WITHIN_RADIUS,
    LONGITUDE_COLUMN,
    NEAREST_PLACE_COLUMN,
)
from str_suitability.spatial.haversine import haversine_km

PRESENCE_LABEL = "presence"
TOWN_CENTER_KM = "distance_to_nearest_town_center_km"
MAJOR_ROAD_KM = "distance_to_major_road_km"
ATTRACTION_COUNT = "attraction_count_within_5km"
ATTRACTION_CATEGORIES = ("Falls", "Mountains")
FEATURES = (
    LISTED_PLACES_WITHIN_RADIUS,
    DISTANCE_COLUMN,
    ROAD_DISTANCE_KM,
    TOWN_CENTER_KM,
    MAJOR_ROAD_KM,
    ATTRACTION_COUNT,
)
SCORES_FILENAME = "site_candidate_scores.parquet"
TRAINING_FILENAME = "site_candidate_training.parquet"
OOF_FILENAME = "site_candidate_oof_predictions.parquet"
REPORT_FILENAME = "site_candidate_report.json"
FOLDS = 5
INNER_FOLDS = 3
PERMUTATION_REPEATS = 20
DISPLAY_LOW = 0.40
DISPLAY_HIGH = 0.60


def random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        **config.CLASSIFIER_PARAMS,
        random_state=config.RANDOM_STATE,
        class_weight="balanced",
        n_jobs=1,
    )


def load_places(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(records)
    frame[LONGITUDE_COLUMN] = pd.to_numeric(frame[LONGITUDE_COLUMN], errors="coerce")
    frame[LATITUDE_COLUMN] = pd.to_numeric(frame[LATITUDE_COLUMN], errors="coerce")
    located = frame.dropna(subset=[LONGITUDE_COLUMN, LATITUDE_COLUMN]).copy()
    ids = located.get("place_id", pd.Series("", index=located.index)).fillna("").astype(str).str.strip()
    identified = located.loc[ids.ne("")].copy()
    identified["_place_id"] = ids.loc[ids.ne("")].to_numpy()
    identified = identified.drop_duplicates("_place_id", keep="first")
    unnamed = located.loc[ids.eq("")]
    places = pd.concat([identified, unnamed], ignore_index=True)
    places["name"] = places.get("name", pd.Series("", index=places.index)).fillna("").astype(str)
    places.loc[places["name"].str.strip().eq(""), "name"] = "Unnamed place"
    return places, {
        "records": int(len(frame)),
        "without_coordinates": int(len(frame) - len(located)),
        "duplicate_place_ids_removed": int(ids.ne("").sum() - identified["_place_id"].nunique()),
        "places_used": int(len(places)),
    }


def _place_features(cells: pd.DataFrame, places: pd.DataFrame) -> pd.DataFrame:
    distances = haversine_km(
        cells[LONGITUDE_COLUMN].to_numpy()[:, None],
        cells[LATITUDE_COLUMN].to_numpy()[:, None],
        places[LONGITUDE_COLUMN].to_numpy()[None, :],
        places[LATITUDE_COLUMN].to_numpy()[None, :],
    )
    nearest = distances.argmin(axis=1)
    rows = np.arange(len(cells))
    result = cells.copy()
    result[LISTED_PLACES_WITHIN_RADIUS] = (
        distances <= config.NEIGHBORHOOD_RADIUS_KM
    ).sum(axis=1).astype(int)
    result[DISTANCE_COLUMN] = distances[rows, nearest]
    result[NEAREST_PLACE_COLUMN] = places["name"].to_numpy()[nearest]
    return result


def _attraction_count(cells: pd.DataFrame, places: pd.DataFrame) -> pd.Series:
    attractions = places.loc[places["category"].isin(ATTRACTION_CATEGORIES)]
    if attractions.empty:
        raise ValueError("the landscape-place file has no Falls or Mountains")
    distances = haversine_km(
        cells[LONGITUDE_COLUMN].to_numpy()[:, None],
        cells[LATITUDE_COLUMN].to_numpy()[:, None],
        attractions[LONGITUDE_COLUMN].to_numpy()[None, :],
        attractions[LATITUDE_COLUMN].to_numpy()[None, :],
    )
    return pd.Series(
        (distances <= config.NEIGHBORHOOD_RADIUS_KM).sum(axis=1).astype(int),
        index=cells.index,
        name=ATTRACTION_COUNT,
    )


def _major_road_distance(cells: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> pd.Series:
    if "road_class" not in roads.columns:
        raise ValueError(f"{LAGUNA_ROADS_PATH} has no road_class field")
    major = roads.loc[roads["road_class"].astype(str).eq("major")].copy()
    major = major.loc[major.geometry.notna() & ~major.geometry.is_empty]
    if major.empty:
        raise ValueError("the road network has no major roads")
    projected_cells = cells.to_crs(DISTANCE_CRS)
    projected_roads = _all_mapped_roads(major).to_crs(DISTANCE_CRS)
    _assert_metre_crs(projected_cells)
    _assert_metre_crs(projected_roads)
    centroids = gpd.GeoDataFrame(
        {CELL_ID_COLUMN: projected_cells[CELL_ID_COLUMN].astype(str).to_numpy()},
        geometry=projected_cells.geometry.centroid,
        crs=DISTANCE_CRS,
    )
    nearest = centroids.sjoin_nearest(
        projected_roads[["geometry"]], how="left", distance_col="major_road_distance_m"
    )
    nearest = nearest.sort_values("major_road_distance_m").drop_duplicates(CELL_ID_COLUMN)
    by_id = nearest.set_index(CELL_ID_COLUMN)["major_road_distance_m"] / 1000.0
    return pd.Series(
        cells[CELL_ID_COLUMN].astype(str).map(by_id).to_numpy(),
        index=cells.index,
        name=MAJOR_ROAD_KM,
    )


def build_feature_frame(
    cells: gpd.GeoDataFrame, places: pd.DataFrame, roads: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    if POBLACION_DISTANCE_COLUMN not in cells.columns:
        raise KeyError(f"grid has no {POBLACION_DISTANCE_COLUMN}")
    features = _place_features(cells, places)
    features[TOWN_CENTER_KM] = features[POBLACION_DISTANCE_COLUMN]
    features[ATTRACTION_COUNT] = _attraction_count(features, places)
    features[MAJOR_ROAD_KM] = _major_road_distance(features, roads)
    return features


def require_complete_features(frame: pd.DataFrame) -> None:
    for feature in FEATURES:
        assert frame[feature].notna().all(), f"{feature} has missing values"


def load_cells() -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    grid = gpd.read_parquet(config.RURAL_PROCESSED_DIR / "grid_suitability.parquet")
    grid[CELL_ID_COLUMN] = grid[CELL_ID_COLUMN].astype(str)
    places, place_report = load_places(config.LISTED_TOURIST_PLACES_PATH)
    grid = _place_features(grid, places)
    road_distances = pd.read_parquet(GRID_ROAD_DISTANCE_PATH)
    road_distances[CELL_ID_COLUMN] = road_distances[CELL_ID_COLUMN].astype(str)
    grid = grid.merge(
        road_distances[[CELL_ID_COLUMN, ROAD_DISTANCE_KM]],
        on=CELL_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    roads = gpd.read_file(LAGUNA_ROADS_PATH)
    cells = build_feature_frame(grid, places, roads)
    require_complete_features(cells)
    return cells, place_report


def build_training_sample(cells: pd.DataFrame, negative_seed: int | None = None) -> pd.DataFrame:
    rural = cells.loc[cells["cell_class"].eq(RURAL_CELL)].copy()
    eligible = rural.loc[rural["municipality"].notna() & rural["municipality"].astype(str).ne("")]
    positives = eligible.loc[eligible["listings_in_cell"] >= 1]
    negatives = eligible.loc[eligible["listings_in_cell"] == 0]
    assert len(positives) > 0 and len(negatives) >= len(positives)
    seed = config.RANDOM_STATE if negative_seed is None else int(negative_seed)
    sampled_negatives = negatives.sample(n=len(positives), random_state=seed)
    sample = pd.concat([positives, sampled_negatives], ignore_index=True)
    sample[PRESENCE_LABEL] = (sample["listings_in_cell"] >= 1).astype(int)
    assert int(sample[PRESENCE_LABEL].sum()) == int((sample[PRESENCE_LABEL] == 0).sum())
    require_complete_features(sample)
    return sample


def _positive_probability(model, features: pd.DataFrame) -> np.ndarray:
    class_index = list(model.classes_).index(1)
    return model.predict_proba(features[list(FEATURES)])[:, class_index]


def _youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    false_positive, true_positive, thresholds = roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    return float(thresholds[finite][np.argmax(true_positive[finite] - false_positive[finite])])


def _inner_oof(features: pd.DataFrame, labels: pd.Series, groups: pd.Series) -> np.ndarray:
    splitter = GroupKFold(n_splits=min(INNER_FOLDS, int(groups.nunique())))
    scores = np.full(len(features), np.nan)
    for train_index, test_index in splitter.split(features, labels, groups):
        model = random_forest()
        model.fit(features.iloc[train_index], labels.iloc[train_index])
        scores[test_index] = _positive_probability(model, features.iloc[test_index])
    assert np.isfinite(scores).all(), "inner municipality scores are incomplete"
    return scores


def _fold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (scores >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def _permutation_auc_drop(model, features: pd.DataFrame, labels: np.ndarray, seed: int) -> np.ndarray:
    baseline = float(roc_auc_score(labels, _positive_probability(model, features)))
    rng = np.random.default_rng(seed)
    drops = []
    for feature in FEATURES:
        values = features[feature].to_numpy().copy()
        repeated = []
        for _ in range(PERMUTATION_REPEATS):
            shuffled = features.copy()
            shuffled[feature] = rng.permutation(values)
            repeated.append(baseline - roc_auc_score(labels, _positive_probability(model, shuffled)))
        drops.append(float(np.mean(repeated)))
    return np.asarray(drops)


def _mean_sd(values: list[float]) -> dict[str, object]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "values": [float(value) for value in values],
    }


def evaluate_forest(sample: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    labels = sample[PRESENCE_LABEL].astype(int)
    groups = sample["municipality"].astype(str)
    features = sample[list(FEATURES)]
    splitter = StratifiedGroupKFold(
        n_splits=FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )
    folds = list(splitter.split(features, labels, groups))
    oof = sample[[CELL_ID_COLUMN, "municipality", PRESENCE_LABEL]].copy()
    oof["fold"] = np.nan
    oof["random_forest_raw"] = np.nan
    fold_rows = []
    importance_rows = []
    for fold_id, (train_index, test_index) in enumerate(folds, start=1):
        train_groups = groups.iloc[train_index]
        test_groups = groups.iloc[test_index]
        assert set(train_groups).isdisjoint(test_groups)
        train_features = features.iloc[train_index]
        train_labels = labels.iloc[train_index]
        inner = _inner_oof(train_features, train_labels, train_groups)
        threshold = _youden_threshold(train_labels.to_numpy(), inner)
        model = random_forest()
        model.fit(train_features, train_labels)
        test_features = features.iloc[test_index]
        scores = _positive_probability(model, test_features)
        observed = labels.iloc[test_index].to_numpy()
        oof.loc[test_index, "fold"] = fold_id
        oof.loc[test_index, "random_forest_raw"] = scores
        fold_rows.append(
            {
                "fold": fold_id,
                "n": int(len(test_index)),
                "youden_threshold": threshold,
                **_fold_metrics(observed, scores, threshold),
            }
        )
        importance_rows.append(
            _permutation_auc_drop(model, test_features, observed, config.RANDOM_STATE + fold_id)
        )
    importance = np.vstack(importance_rows)
    importance_summary = [
        {
            "feature": feature,
            "mean_auc_drop": float(importance[:, index].mean()),
            "std_auc_drop": float(importance[:, index].std(ddof=1)),
            "fold_values": importance[:, index].astype(float).tolist(),
        }
        for index, feature in enumerate(FEATURES)
    ]
    report = {
        "model": "random_forest",
        "features": list(FEATURES),
        "training_rows": int(len(sample)),
        "training_positives": int(labels.sum()),
        "training_negatives": int((labels == 0).sum()),
        "validation_scheme": f"stratified_group_kfold_{FOLDS}",
        "folds": fold_rows,
        "metrics": {
            name: _mean_sd([float(row[name]) for row in fold_rows])
            for name in fold_rows[0]
            if name not in {"fold", "n", "youden_threshold"}
        },
        "permutation_importance": importance_summary,
        "permutation_repeats": PERMUTATION_REPEATS,
    }
    return oof, report


def score_rural_cells(sample: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    rural = cells.loc[
        cells["cell_class"].eq(RURAL_CELL)
        & cells["municipality"].notna()
        & cells["municipality"].astype(str).ne("")
    ].copy()
    model = random_forest()
    model.fit(sample[list(FEATURES)], sample[PRESENCE_LABEL].astype(int))
    scores = _positive_probability(model, rural)
    bands = np.where(
        scores < DISPLAY_LOW,
        "<0.40",
        np.where(scores < DISPLAY_HIGH, "0.40 to <0.60", ">=0.60"),
    )
    result = rural[[CELL_ID_COLUMN, "municipality", "listings_in_cell", *FEATURES]].copy()
    result["candidate_pattern_score"] = scores
    result["is_empty"] = result["listings_in_cell"].eq(0)
    result["in_training_sample"] = result[CELL_ID_COLUMN].isin(set(sample[CELL_ID_COLUMN]))
    result["display_band"] = bands
    result["model_name"] = "random_forest"
    return result


def run_site_candidate(output_dir: Path | None = None) -> dict[str, object]:
    output_dir = config.RURAL_PROCESSED_DIR if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cells, place_report = load_cells()
    sample = build_training_sample(cells)
    oof, report = evaluate_forest(sample)
    scores = score_rural_cells(sample, cells)
    report["place_source"] = place_report
    report["population"] = {
        "rural_cells_scored": int(len(scores)),
        "empty_rural_cells": int(scores["is_empty"].sum()),
        "listed_rural_cells": int((~scores["is_empty"]).sum()),
        "negative_sampling_seed": int(config.RANDOM_STATE),
    }
    report["interpretation"] = (
        "Higher raw Random Forest scores indicate greater similarity to rural cells with an active STR. "
        "Scores are a screening signal, not a calibrated population probability or feasibility assessment."
    )
    sample[[CELL_ID_COLUMN, "municipality", PRESENCE_LABEL, *FEATURES]].to_parquet(
        output_dir / TRAINING_FILENAME, index=False
    )
    oof.to_parquet(output_dir / OOF_FILENAME, index=False)
    scores.to_parquet(output_dir / SCORES_FILENAME, index=False)
    (output_dir / REPORT_FILENAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report