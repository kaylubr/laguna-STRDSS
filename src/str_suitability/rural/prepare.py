import shutil
import subprocess
from pathlib import Path

import pandas as pd

from str_suitability import config
from str_suitability.pipeline import run_pipeline
from str_suitability.preprocess.clean_airroi import build_target_dataset_from_paths
from str_suitability.rural.classify import classify_listings_by_barangay
from str_suitability.rural.psa import (
    classify_barangay_polygons,
    load_barangay_classification,
    load_barangay_polygons,
)

SOURCE_CANDIDATES = (
    config.PROJECT_ROOT.parent.parent / "airroi",
    config.PROJECT_ROOT.parent / "airroi",
)

SOURCE_FILES = {
    config.AIRROI_LISTINGS_PATH: ("data", "airroi", "laguna_listings.json"),
    config.AIRROI_HISTORY_DIR: ("data", "airroi", "history"),
    config.PSA_PATH: ("data", "airroi", "PSA.json"),
    config.BARANGAY_POLYGONS_PATH: ("data", "laguna.geojson"),
    config.BARANGAY_CLASSIFICATION_PATH: ("data", "barangay_with_classification.json"),
}


def ensure_rural_inputs() -> None:
    _link_source_files()
    _write_targets()
    _write_listing_classification()
    grid_path = config.PROCESSED_DIR / "grid_features.parquet"
    observations_path = config.PROCESSED_DIR / "training_observations.parquet"
    if not grid_path.exists() or not observations_path.exists():
        run_pipeline()


def _link_source_files() -> None:
    source_root = _source_root()
    for destination, parts in SOURCE_FILES.items():
        if destination.exists():
            continue
        source = source_root.joinpath(*parts)
        assert source.exists(), f"missing {destination.name}, and {source} is not available either"
        destination.parent.mkdir(parents=True, exist_ok=True)
        _link(source, destination)


def _source_root() -> Path:
    for candidate in SOURCE_CANDIDATES:
        if (candidate / "data" / "airroi" / "laguna_listings.json").exists():
            return candidate
    searched = ", ".join(str(candidate) for candidate in SOURCE_CANDIDATES)
    raise FileNotFoundError(
        "the rural run needs the AirROI listings, PSA population, and barangay files, "
        f"and none were found under data/ or in {searched}"
    )


def _link(source: Path, destination: Path) -> None:
    try:
        destination.symlink_to(source, target_is_directory=source.is_dir())
        return
    except OSError:
        pass
    if source.is_dir():
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(destination), str(source)],
            check=True,
        )
        return
    shutil.copy2(source, destination)


def _write_targets() -> None:
    path = config.INTERIM_DIR / "airroi_targets.parquet"
    if path.exists():
        return
    targets = build_target_dataset_from_paths(
        config.AIRROI_LISTINGS_PATH,
        config.AIRROI_HISTORY_DIR,
        (config.TTM_WINDOW_START, config.TTM_WINDOW_END),
    )
    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(path, index=False)


def _write_listing_classification() -> None:
    if config.RURAL_URBAN_LISTINGS_PATH.exists():
        return
    targets = pd.read_parquet(config.INTERIM_DIR / "airroi_targets.parquet")
    barangays = classify_barangay_polygons(
        load_barangay_polygons(config.BARANGAY_POLYGONS_PATH),
        load_barangay_classification(config.BARANGAY_CLASSIFICATION_PATH),
    )
    classified = classify_listings_by_barangay(targets, barangays)
    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    classified.to_csv(config.RURAL_URBAN_LISTINGS_PATH, index=False)
