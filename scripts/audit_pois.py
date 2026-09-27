import json

import geopandas as gpd
import pandas as pd

from str_suitability import config
from str_suitability.audit.density import (
    category_density_distributions,
    pois_inside_grid,
    read_grid,
    rural_cell_ids,
    rural_rows,
)
from str_suitability.audit.poi_tags import (
    CATCH_ALL_COLUMN,
    TAG_KEY_COLUMN,
    TAG_VALUE_COLUMN,
    catch_all_breakdown,
    missing_universe_keys,
    tag_inventory,
)
from str_suitability.audit.weights import (
    POI_CATEGORY_NAMES,
    build_rural_scoring_frame,
    ewm_scenarios,
)

FEATURE_DIR = config.PROJECT_ROOT / "assets" / "osm"
AUDIT_DIR = config.PROCESSED_DIR / "poi_audit"
POI_PATH = FEATURE_DIR / "laguna_pois.parquet"
REQUIRED_TAG_COLUMNS = {TAG_KEY_COLUMN, TAG_VALUE_COLUMN, CATCH_ALL_COLUMN}


def write_json(path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def main() -> dict[str, object]:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    pois = gpd.read_parquet(POI_PATH)
    grid = read_grid(config.PROCESSED_DIR)
    rural_ids = rural_cell_ids(config.RURAL_PROCESSED_DIR)

    summary = {"pois": int(len(pois))}
    if REQUIRED_TAG_COLUMNS <= set(pois.columns):
        inventory = tag_inventory(pois)
        inventory.to_csv(AUDIT_DIR / "tag_counts.csv", index=False)
        catch_all, catch_all_summary = catch_all_breakdown(pois)
        catch_all.to_csv(AUDIT_DIR / "catch_all_breakdown.csv", index=False)
        totals = inventory.groupby("category")["count"].sum().sort_values(ascending=False)
        summary.update(
            {
                "category_totals": {name: int(value) for name, value in totals.items()},
                "category_shares": {
                    name: float(value / len(pois)) for name, value in totals.items()
                },
                "catch_all": catch_all_summary,
                "universe_keys_absent": missing_universe_keys(pois),
                "relevance_tier_totals": {
                    name: int(value)
                    for name, value in inventory.groupby("relevance_tier")["count"].sum().items()
                },
            }
        )
    else:
        summary["tag_inventory"] = (
            "unavailable: the POI extract carries no raw tag columns "
            f"({sorted(REQUIRED_TAG_COLUMNS)})"
        )

    inside = pois_inside_grid(grid, pois)
    distributions = category_density_distributions(rural_rows(grid, rural_ids), POI_CATEGORY_NAMES)
    distributions.to_csv(AUDIT_DIR / "category_density_distributions.csv", index=False)

    scoring = build_rural_scoring_frame(config.PROCESSED_DIR, config.RURAL_PROCESSED_DIR)
    scenarios, scenario_report = ewm_scenarios(scoring)
    scenarios.to_csv(AUDIT_DIR / "ewm_scenarios.csv", index=False)

    summary.update(
        {
            "pois_inside_grid": int(inside.sum()),
            "pois_outside_grid": int((~inside).sum()),
            "rural_cells": int(len(rural_ids)),
            "ewm": scenario_report,
        }
    )
    write_json(AUDIT_DIR / "summary.json", summary)
    return summary


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, default=str))
    print(f"\nartifacts: {AUDIT_DIR}")
