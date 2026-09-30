"""Screen for rural squares that look like places where an Airbnb already exists.

The examples are rural squares with at least one active listing. The comparison
squares are a random sample of rural squares with none, the same count, seed 42.
The forest sees listed landscape places within 5 km, distance to the nearest
one, and straight-line road distance. Nearby revenue, occupancy, and listing
count are not inputs.
"""

import json

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from str_suitability import config
from str_suitability.features.road_distance import GRID_ROAD_DISTANCE_PATH, ROAD_DISTANCE_KM
from str_suitability.modeling.location_classifier import (
    LISTINGS_IN_CELL,
    SURROUNDING_LISTINGS,
    SURROUNDING_OCCUPANCY,
    SURROUNDING_REVENUE,
    _cell_of_each_listing,
    _group_splits,
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

PRESENCE_FEATURES = (
    LISTED_PLACES_WITHIN_RADIUS,
    DISTANCE_COLUMN,
    ROAD_DISTANCE_KM,
)
EXCLUDED_MARKET_FEATURES = (
    SURROUNDING_REVENUE,
    SURROUNDING_OCCUPANCY,
    SURROUNDING_LISTINGS,
)
PRESENCE_LABEL = "presence"
PRESENCE_CLASSES = ("Not the pattern", "Looks listed")
MIN_LISTINGS_FOR_EARNINGS = 3
SCORES_FILENAME = "presence_scores.parquet"
TRAINING_FILENAME = "presence_training.parquet"
REPORT_FILENAME = "presence_screen.json"


def presence_forest() -> RandomForestClassifier:
    params = dict(config.CLASSIFIER_PARAMS)
    return RandomForestClassifier(
        random_state=config.RANDOM_STATE,
        class_weight="balanced",
        n_jobs=1,
        **params,
    )


def select_training_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Balance listed rural squares with a random sample of unlisted rural squares."""
    assert set(PRESENCE_FEATURES).isdisjoint(EXCLUDED_MARKET_FEATURES)
    required = {CELL_ID_COLUMN, "cell_class", "municipality", LISTINGS_IN_CELL, *PRESENCE_FEATURES}
    missing = required - set(frame.columns)
    assert not missing, f"the presence frame is missing {sorted(missing)}"
    rural = frame.loc[frame["cell_class"] == RURAL_CELL].copy()
    eligible = rural.loc[rural["municipality"].notna() & (rural["municipality"].astype(str) != "")]
    for column in PRESENCE_FEATURES:
        assert eligible[column].notna().all(), f"{column} is blank on a rural training square"
    listed = eligible.loc[eligible[LISTINGS_IN_CELL] >= 1]
    pool = eligible.loc[eligible[LISTINGS_IN_CELL] == 0]
    assert len(listed) > 0, "no rural square with a municipality contains an Airbnb"
    assert len(pool) >= len(listed), "there are fewer empty rural squares than listed ones"
    unlisted = pool.sample(n=len(listed), random_state=config.RANDOM_STATE)
    sample = pd.concat([listed, unlisted], ignore_index=True)
    sample[PRESENCE_LABEL] = (sample[LISTINGS_IN_CELL] >= 1).astype(int)
    report = {
        "listed_rural_squares": int(len(listed)),
        "unlisted_rural_squares_available": int(len(pool)),
        "unlisted_rural_squares_sampled": int(len(unlisted)),
        "rural_squares_without_municipality": int(len(rural) - len(eligible)),
    }
    assert int(sample[PRESENCE_LABEL].sum()) == int((sample[PRESENCE_LABEL] == 0).sum())
    return sample, report


def _positive_probability(model, features: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(features)
    column = list(model.classes_).index(1)
    return raw[:, column]


def _binary_metrics(observed, probability, predicted) -> dict[str, object]:
    observed_array = np.asarray(observed, dtype=int)
    predicted_array = np.asarray(predicted, dtype=int)
    labels = [0, 1]
    matrix = confusion_matrix(observed_array, predicted_array, labels=labels)
    auc = None
    if len(set(observed_array.tolist())) > 1 and np.isfinite(probability).all():
        auc = float(roc_auc_score(observed_array, probability))
    return {
        "accuracy": float(accuracy_score(observed_array, predicted_array)),
        "macro_f1": float(f1_score(observed_array, predicted_array, average="macro", zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": matrix.tolist(),
        "n": int(len(observed_array)),
    }


def _mean_std(values: list[float | None]) -> dict[str, float | int | None]:
    usable = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not usable:
        return {"mean": None, "std": None, "n_folds": 0}
    std = float(np.std(usable, ddof=1)) if len(usable) > 1 else 0.0
    return {"mean": float(np.mean(usable)), "std": std, "n_folds": len(usable)}


def _fit_metrics(model_factory, features, target, groups, strategy: str | None) -> dict[str, object]:
    folds, scheme = _group_splits(features, target, groups)
    fold_metrics = []
    for train_index, test_index in folds:
        if strategy is None:
            model = model_factory()
            model.fit(features.iloc[train_index], target.iloc[train_index])
            probability = _positive_probability(model, features.iloc[test_index])
            predicted = (probability >= 0.5).astype(int)
        else:
            model = DummyClassifier(strategy=strategy, random_state=config.RANDOM_STATE)
            model.fit(features.iloc[train_index], target.iloc[train_index])
            probability = _positive_probability(model, features.iloc[test_index])
            predicted = model.predict(features.iloc[test_index]).astype(int)
        fold_metrics.append(_binary_metrics(target.iloc[test_index], probability, predicted))
    combined = {
        key: _mean_std([fold[key] for fold in fold_metrics])
        for key in ("accuracy", "macro_f1", "roc_auc")
    }
    combined["scheme"] = scheme
    combined["folds"] = fold_metrics
    combined["confusion_matrix"] = np.sum(
        [np.asarray(fold["confusion_matrix"], dtype=float) for fold in fold_metrics], axis=0
    ).astype(int).tolist()
    return combined


def evaluate_presence(sample: pd.DataFrame) -> dict[str, object]:
    features = sample[list(PRESENCE_FEATURES)]
    target = sample[PRESENCE_LABEL].astype(int)
    groups = sample["municipality"].astype(str)
    forest = _fit_metrics(presence_forest, features, target, groups, None)
    stratified = _fit_metrics(presence_forest, features, target, groups, "stratified")
    majority = _fit_metrics(presence_forest, features, target, groups, "most_frequent")
    forest_auc = forest["roc_auc"]["mean"]
    chance_auc = stratified["roc_auc"]["mean"]
    separates = (
        forest_auc is not None
        and chance_auc is not None
        and forest_auc > chance_auc
        and forest_auc > 0.5
    )
    return {
        "question": "Does this rural square look like rural squares where an Airbnb already exists?",
        "features": list(PRESENCE_FEATURES),
        "excluded_from_the_model": list(EXCLUDED_MARKET_FEATURES),
        "forest": forest,
        "stratified_baseline": stratified,
        "majority_baseline": majority,
        "separates_better_than_random": bool(separates),
    }


def score_all_cells(sample: pd.DataFrame, cells: pd.DataFrame) -> tuple[pd.DataFrame, RandomForestClassifier]:
    model = presence_forest()
    model.fit(sample[list(PRESENCE_FEATURES)], sample[PRESENCE_LABEL].astype(int))
    probability = _positive_probability(model, cells[list(PRESENCE_FEATURES)])
    scored = cells[[CELL_ID_COLUMN, LISTED_PLACES_WITHIN_RADIUS, DISTANCE_COLUMN, NEAREST_PLACE_COLUMN]].copy()
    scored[CELL_ID_COLUMN] = scored[CELL_ID_COLUMN].astype(str)
    scored["presence_probability"] = probability
    scored["presence_class"] = np.where(probability >= 0.5, PRESENCE_CLASSES[1], PRESENCE_CLASSES[0])
    return scored, model


def earnings_per_bedroom(listings: pd.DataFrame) -> pd.DataFrame:
    """Mean trailing-twelve-month revenue per bedroom. Bedrooms of zero are left out."""
    usable = listings.loc[
        listings["bedrooms"].notna()
        & (listings["bedrooms"] > 0)
        & listings["ttm_revenue"].notna()
        & listings[CELL_ID_COLUMN].astype(str).ne("")
    ].copy()
    usable["earnings_per_bedroom"] = usable["ttm_revenue"] / usable["bedrooms"]
    grouped = usable.groupby(CELL_ID_COLUMN, as_index=False).agg(
        listings_with_bedrooms=(CELL_ID_COLUMN, "size"),
        earnings_per_bedroom=("earnings_per_bedroom", "mean"),
    )
    return grouped.loc[grouped["listings_with_bedrooms"] >= MIN_LISTINGS_FOR_EARNINGS].copy()


def relate_earnings(cells: pd.DataFrame, per_bedroom: pd.DataFrame) -> dict[str, object]:
    """Side check: do the same three surroundings track earnings per bedroom?"""
    rural = cells.loc[cells["cell_class"] == RURAL_CELL, [CELL_ID_COLUMN, "municipality", *PRESENCE_FEATURES]]
    merged = per_bedroom.merge(rural, on=CELL_ID_COLUMN, how="inner")
    merged = merged.loc[merged["municipality"].notna()].copy()
    correlations = []
    for column in PRESENCE_FEATURES:
        pair = merged[[column, "earnings_per_bedroom"]].dropna()
        coefficient = None
        if len(pair) >= 3 and pair[column].nunique() > 1:
            coefficient = float(pair[column].corr(pair["earnings_per_bedroom"], method="spearman"))
        correlations.append({"feature": column, "spearman": coefficient, "n": int(len(pair))})
    regression = _earnings_regression(merged)
    return {
        "rule": (
            "Rural squares with at least three active listings that have a bedroom count. "
            "The cell value is the mean of trailing-twelve-month revenue divided by bedrooms."
        ),
        "squares": int(len(merged)),
        "correlations": correlations,
        "grouped_regression": regression,
    }


def _earnings_regression(frame: pd.DataFrame) -> dict[str, object]:
    if frame["municipality"].nunique() < 2 or len(frame) < 10:
        return {"folds": 0, "r2_mean": None, "baseline_r2_mean": None}
    groups = frame["municipality"].astype(str)
    splits = min(5, int(groups.nunique()))
    folder = GroupKFold(n_splits=splits)
    features = frame[list(PRESENCE_FEATURES)]
    target = frame["earnings_per_bedroom"].to_numpy(dtype=float)
    model_scores = []
    baseline_scores = []
    for train_index, test_index in folder.split(features, target, groups):
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=1,
            random_state=config.RANDOM_STATE,
            n_jobs=1,
        )
        model.fit(features.iloc[train_index], target[train_index])
        predicted = model.predict(features.iloc[test_index])
        observed = target[test_index]
        model_scores.append(_r2(observed, predicted))
        baseline_scores.append(_r2(observed, np.full(len(test_index), target[train_index].mean())))
    return {
        "folds": splits,
        "r2_mean": float(np.mean(model_scores)),
        "baseline_r2_mean": float(np.mean(baseline_scores)),
    }


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    total = float(np.sum((observed - observed.mean()) ** 2))
    if total == 0:
        return 0.0
    residual = float(np.sum((observed - predicted) ** 2))
    return 1.0 - residual / total


def listed_places_for_training(path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Places with coordinates. Repeated place ids are kept once."""
    records = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(records)
    frame[LONGITUDE_COLUMN] = pd.to_numeric(frame.get(LONGITUDE_COLUMN), errors="coerce")
    frame[LATITUDE_COLUMN] = pd.to_numeric(frame.get(LATITUDE_COLUMN), errors="coerce")
    located = frame.dropna(subset=[LONGITUDE_COLUMN, LATITUDE_COLUMN]).copy()
    place_id = located.get("place_id", pd.Series("", index=located.index)).fillna("").astype(str).str.strip()
    identified = located.loc[place_id.ne("")].copy()
    identified["_place_id"] = place_id.loc[place_id.ne("")].to_numpy()
    identified = identified.drop_duplicates("_place_id", keep="first")
    unnamed = located.loc[place_id.eq("")]
    places = pd.concat([identified, unnamed], ignore_index=True)
    places["name"] = places.get("name", pd.Series("", index=places.index)).fillna("").astype(str)
    places.loc[places["name"].str.strip().eq(""), "name"] = "Unnamed place"
    report = {
        "records": int(len(frame)),
        "without_coordinates": int(len(frame) - len(located)),
        "duplicate_place_ids_removed": int(place_id.ne("").sum() - identified["_place_id"].nunique()),
        "places_used": int(len(places)),
    }
    return places[[LONGITUDE_COLUMN, LATITUDE_COLUMN, "name"]].copy(), report


def refresh_listed_places(cells: pd.DataFrame, places: pd.DataFrame) -> pd.DataFrame:
    """Replace stored place count and distance with the current place list."""
    assert len(places) > 0, "there are no listed places with coordinates"
    place_distances = haversine_km(
        cells[LONGITUDE_COLUMN].to_numpy()[:, None],
        cells[LATITUDE_COLUMN].to_numpy()[:, None],
        places[LONGITUDE_COLUMN].to_numpy()[None, :],
        places[LATITUDE_COLUMN].to_numpy()[None, :],
    )
    nearest = place_distances.argmin(axis=1)
    rows = np.arange(len(cells))
    refreshed = cells.copy()
    refreshed[LISTED_PLACES_WITHIN_RADIUS] = (place_distances <= config.NEIGHBORHOOD_RADIUS_KM).sum(axis=1).astype(int)
    refreshed[DISTANCE_COLUMN] = place_distances[rows, nearest]
    refreshed[NEAREST_PLACE_COLUMN] = places["name"].to_numpy()[nearest]
    return refreshed


def _bedrooms_by_listing(path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        {"listing_id": str(item["listing_id"]), "bedrooms": item.get("bedrooms")}
        for item in payload["listings"]
    ]
    return pd.DataFrame(rows)


def load_presence_frame() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    cells = gpd.read_parquet(config.RURAL_PROCESSED_DIR / "grid_suitability.parquet")
    places, place_report = listed_places_for_training(config.LISTED_TOURIST_PLACES_PATH)
    cells = refresh_listed_places(cells, places)
    cells.attrs["place_report"] = place_report
    roads = pd.read_parquet(GRID_ROAD_DISTANCE_PATH)
    roads[CELL_ID_COLUMN] = roads[CELL_ID_COLUMN].astype(str)
    cells[CELL_ID_COLUMN] = cells[CELL_ID_COLUMN].astype(str)
    merged = cells.merge(roads[[CELL_ID_COLUMN, ROAD_DISTANCE_KM]], on=CELL_ID_COLUMN, how="left", validate="one_to_one")
    assert merged[ROAD_DISTANCE_KM].notna().all(), "a square has no road distance"
    merged.attrs["place_report"] = place_report
    targets = pd.read_parquet(config.INTERIM_DIR / "airroi_targets.parquet")
    active = targets.loc[targets["in_training_population"]].copy()
    active["listing_id"] = active["listing_id"].astype(str)
    bedrooms = _bedrooms_by_listing(config.AIRROI_LISTINGS_PATH)
    active = active.merge(bedrooms, on="listing_id", how="left", validate="one_to_one")
    active[CELL_ID_COLUMN] = _cell_of_each_listing(merged, active)
    return merged, active


def run_presence_screen(output_dir=None) -> dict[str, object]:
    output_dir = config.RURAL_PROCESSED_DIR if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cells, listings = load_presence_frame()
    sample, sample_report = select_training_rows(cells)
    sample_report["places"] = cells.attrs.get("place_report", {})
    evaluation = evaluate_presence(sample)
    scored, _model = score_all_cells(sample, cells)
    per_bedroom = earnings_per_bedroom(listings)
    earnings = relate_earnings(cells, per_bedroom)
    report = {
        **sample_report,
        **evaluation,
        "earnings_per_bedroom": earnings,
        "classes": list(PRESENCE_CLASSES),
        "probability_rule": "Looks listed when the probability is at least 0.5.",
    }
    scored.to_parquet(output_dir / SCORES_FILENAME, index=False)
    sample[[CELL_ID_COLUMN, "municipality", PRESENCE_LABEL, *PRESENCE_FEATURES]].to_parquet(
        output_dir / TRAINING_FILENAME, index=False
    )
    (output_dir / REPORT_FILENAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
