import geopandas as gpd
import pandas as pd

from str_suitability.rural.classify import CELL_CLASS_COLUMN, CELL_ID_COLUMN, RURAL_CELL

GRID_FILENAME = "grid_features.parquet"
RURAL_CLASSIFICATION_FILENAME = "grid_rural_classification.parquet"
DENSITY_PREFIX = "poi_density_"
COUNT_PREFIX = "poi_count_"


def rural_cell_ids(rural_dir) -> pd.Index:
    classification = pd.read_parquet(rural_dir / RURAL_CLASSIFICATION_FILENAME)
    selected = classification.loc[classification[CELL_CLASS_COLUMN] == RURAL_CELL, CELL_ID_COLUMN]
    assert len(selected) > 0, "the rural classification holds no rural cell"
    return pd.Index(selected)


def read_grid(processed_dir) -> gpd.GeoDataFrame:
    return gpd.read_parquet(processed_dir / GRID_FILENAME)


def rural_rows(grid: gpd.GeoDataFrame, rural_ids: pd.Index) -> gpd.GeoDataFrame:
    subset = grid[grid[CELL_ID_COLUMN].isin(set(rural_ids))].copy()
    assert len(subset) == len(rural_ids), "the grid does not cover every rural cell"
    return gpd.GeoDataFrame(subset, geometry="geometry", crs=grid.crs)


def pois_inside_grid(grid: gpd.GeoDataFrame, pois: gpd.GeoDataFrame) -> pd.Series:
    inside = gpd.sjoin(
        pois.to_crs(grid.crs)[["osm_id", "geometry"]],
        grid[[CELL_ID_COLUMN, "geometry"]],
        predicate="within",
        how="inner",
    )
    return pois.index.isin(inside.index)


def category_density_distributions(frame: gpd.GeoDataFrame, categories) -> pd.DataFrame:
    records = []
    for category in categories:
        density = frame[f"{DENSITY_PREFIX}{category}"].astype(float).to_numpy()
        count = frame[f"{COUNT_PREFIX}{category}"].astype(float).to_numpy()
        without = density == 0
        records.append(
            {
                "category": category,
                "pois_in_rural_cells": int(count.sum()),
                "cells": int(len(density)),
                "cells_with_poi": int((~without).sum()),
                "cells_without_poi": int(without.sum()),
                "share_cells_without_poi": float(without.mean()),
                "mean_density": float(density.mean()),
                "median_density": float(pd.Series(density).median()),
                "p90_density": float(pd.Series(density).quantile(0.9)),
                "max_density": float(density.max()),
            }
        )
    return pd.DataFrame(records)


def tag_group_density(
    grid: gpd.GeoDataFrame, pois: gpd.GeoDataFrame, masks: dict[str, pd.Series]
) -> pd.DataFrame:
    frame = pois.to_crs(grid.crs)[["osm_id", "geometry"]].copy()
    for name, mask in masks.items():
        frame[name] = mask.astype(int).to_numpy()

    joined = gpd.sjoin(frame, grid[[CELL_ID_COLUMN, "geometry"]], predicate="within", how="inner")
    area_km2 = (grid["cell_area_m2"] / 1_000_000).to_numpy()

    records = {CELL_ID_COLUMN: grid[CELL_ID_COLUMN].to_numpy()}
    for name in masks:
        counts = joined.groupby(CELL_ID_COLUMN)[name].sum()
        aligned = counts.reindex(grid[CELL_ID_COLUMN], fill_value=0).astype(float).to_numpy()
        records[f"{COUNT_PREFIX}{name}"] = aligned.astype(int)
        records[f"{DENSITY_PREFIX}{name}"] = aligned / area_km2
    return pd.DataFrame(records)
