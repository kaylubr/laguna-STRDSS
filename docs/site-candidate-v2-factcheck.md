# site_candidate_v2 — Chapter 3 fact-check and supplementary results

Branch: `experiment/model-improvement2`. Every claim below is grounded in the repository's
code, data files, or stored reports, with file paths and line numbers. Nothing in the stored
Part 1 results was changed, rerun, or tuned.

## Part 1 — Fact check

### 1. Repo state

- Latest commit: `a77ec46a684fec61baf67751489726381773aa99` — *"Add a rural presence screen and
  keep site-candidate v2 as a separate layer."* (33 files, +5961/−346). It committed the v2
  sources `src/str_suitability/modeling/site_candidate_v2.py`,
  `scripts/run_site_candidate_v2.py`, `tests/test_site_candidate_v2.py`, plus `presence.py`,
  `tests/test_presence.py`, `scripts/run_presence_screen.py`.
- Uncommitted: one path only, `data/README.md`, and it is a line-ending-only change (LF→CRLF):
  `git diff --ignore-cr-at-eol` and `--ignore-all-space` are empty, and `file` reports
  "UTF-8 text, with CRLF line terminators". No v2 content changed.
- Untracked v2 files: none. The v2 run artifacts under `data/processed/rural/` are gitignored by
  `.gitignore:41-43` (`data/*`, `!data/README.md`).
- v2 random-forest numbers (`data/processed/rural/site_candidate_v2_report.json`,
  `metrics.v2_random_forest`):
  - mean ROC-AUC `0.7274258287458627` (sample SD `0.09218070036883673`), lines 818-830;
  - per-fold ROC-AUC (lines 823-829): `0.5833333333333334`, `0.7361111111111112`,
    `0.7595307917888563`, `0.7208333333333333`, `0.8373205741626794`;
  - macro F1 = Youden-cut macro F1 `0.6561302663653507` (SD `0.05589913167754812`), lines 867-878
    (the report stores no plain 0.5-cut macro F1 for the six-feature RF);
  - fold gains over the three-feature baseline (`matched_deltas.v2_random_forest.delta_auc_fold`,
    lines 1080-1086): `0.051767676767676796`, `0.1435185185185186`, `0.029325513196480912`,
    `0.14374999999999993`, `0.09808612440191389`; mean `0.09328956657691803`.
  - Stored selection: `selected_v2_model = "v2_logistic_regression"` (line 5), because logistic
    mean ROC-AUC `0.7459168081494058` > RF `0.7274258287458627`.

### 2. Distances — how each of the six features is computed

Cell reference point for the three Haversine features: `longitude`/`latitude` = the cell's
`representative_point`, built in EPSG:32651 then reprojected to EPSG:4326
(`src/str_suitability/spatial/grid.py:74-83`). The two road features instead use the polygon
`geometry.centroid` in EPSG:32651 — a slightly different reference point.

- `listed_places_within_radius` — Haversine count within 5 km (`presence.py:292-307`, count line
  304); radius `config.NEIGHBORHOOD_RADIUS_KM = 5.0` (`config.py:37-38`); source
  `data/poi_laguna.json` (all deduped places); `spatial/haversine.py:6-24`, `EARTH_RADIUS_KM = 6371.0`.
- `distance_to_listed_tourist_place` — minimum Haversine (`presence.py:305`); same place universe.
- `road_distance_km` — precomputed projected nearest: `features/road_distance.py:29-64`
  (`sjoin_nearest` in EPSG:32651 from `geometry.centroid`, metres/1000; `_assert_metre_crs`
  enforces 32651 at lines 107-112); all mapped roads, `road_class` not a filter (lines 100-104);
  read/merged at `site_candidate_v2.py:449-457` from `data/processed/grid_road_distance.parquet`
  (builder `scripts/build_grid_road_distance.py`), derived from `data/laguna_roads.geojson`.
- `distance_to_nearest_town_center_km` — copy of `distance_to_poblacion`
  (`site_candidate_v2.py:469`); minimum Haversine in `features/context.py:116-127` (via
  `features/accessibility.py:10-28`); anchors = barangay names matching `poblaci[oó]n|\(pob`
  (`context.py:8`) plus municipality-centroid fallback; source barangay layer `data/laguna.geojson`.
- `distance_to_major_road_km` — `site_candidate_v2.py:238-274`; filters `road_class == "major"`
  (line 251), `sjoin_nearest` in EPSG:32651 from `geometry.centroid`, metres/1000; source
  `data/laguna_roads.geojson`.
- `attraction_count_within_5km` — Haversine count within 5 km of `category in ("Falls",
  "Mountains")` (`site_candidate_v2.py:68,223-235`); source `data/poi_laguna.json`.

Summary: three Haversine (lon/lat) features, two projected straight-line (EPSG:32651) road
features, one copy of a Haversine feature. EPSG:32651 is confirmed for the projected pair.

### 3. Active-listing rule

- Decisive code `src/str_suitability/preprocess/clean_airroi.py:23-46`:
  `is_dormant = (ttm_revenue.fillna(0)==0) & (ttm_occupancy.fillna(0)==0)` (lines 24-26);
  `window_revenue` = sum of monthly revenue over the TTM window (line 32);
  `in_training_population = ~is_dormant & (window_revenue > 0)` (lines 41-44).
- Dormant = TTM revenue and TTM occupancy both exactly 0 (nulls→0). Corroborated = at least one
  positive month in the window. There is no review-recency, occupancy, or revenue-magnitude
  threshold. TTM window `2025-08`…`2026-07` (`config.py:29-30`).
- Duplicates: no listing dedup step — uniqueness asserted (`clean_airroi.py:57-59`);
  `data/README.md:25` says all 2,080 have unique `listing_id`. (The only `listing_id`
  `drop_duplicates` is `rural/classify.py:191`, asserted row-preserving.)
- `listings_in_cell`: `gpd.sjoin` of listing points (EPSG:4326) to grid polygons (EPSG:32651),
  `predicate="within"` (`rural/site_score.py:247-266`; count at 215-237; mirrored in
  `location_classifier.py:578-588`; used at `presence.py:335`).
- Counts: raw **2,080** → after duplicate removal **2,080** (0 removed) → dormant excluded
  **732** + uncorroborated excluded **71** → active **1,277** → off-grid **46** → fitted
  **1,231** (288 cells). Sources: `data/laguna_listings.json` metadata; `docs/adr/0006-*.md:13-29`;
  `data/processed/model_summary.json:9-13`.
- Snapshot: retrieved **2026-08-18T14:14:15Z** (`data/laguna_listings.json`; `data/README.md:16`).

### 4. Landscape places list (`data/poi_laguna.json`)

- **159 records**; categories (verified by grep): **Falls 59, Mountains 17, Lakes & Ponds 19,
  Rivers & Landscape 64**.
- **39 without coordinates**, **120 located**; **16 blank `place_id`** (kept); **9 duplicate
  non-empty `place_id`s removed** → **111 used** (159−39−9). Matches `presence_screen.json` and
  the report `places` block (lines 18-23).
- Attraction subset = Falls + Mountains; the count restricts the 111-place universe to those two
  categories (`site_candidate_v2.py:68,223-235`). Other categories remain in the universe for the
  two baseline features.
- Dedupe: `site_candidate_v2.py:177-200` and `presence.py:268-289`; rule "first row of each
  non-empty `place_id`; blank `place_id` rows kept".
- Provenance: **NOT FOUND.** The file is git-ignored (`.gitignore:41-43`), so no commit/author/date;
  no build/download script exists; `data/README.md` never mentions it; docs call it a
  "curated"/"locally generated" landscape list; the code's own registry marks
  `url/license/retrieval_date = unknown`, `origin = locally generated`
  (`site_candidate_v2.py:814-842`; `data/processed/rural/site_candidate_v2_source_registry.csv`).
  Only hints: `coordinate_source` ∈ original/google_places/osm_nominatim/unresolved. Docs
  reference an older 172-place version (`model-pipeline.txt:54-55`).

### 5. Data vintages

- OSM roads (`data/laguna_roads.geojson`): source recorded as an OpenStreetMap/Geofabrik extract
  (`docs/road-accessibility.md:9`; `features/road_distance.py:3-4`; `build_grid_road_distance.py:46-49`).
  Extraction date/URL **NOT FOUND**.
- PSA (`data/PSA.json`): **PSA 2024 Census of Population (POPCEN), reference date 2024-07-01**,
  provincial total 3,687,345 (`data/PSA.json:1-7`; `data/README.md:71-85`).
- Boundaries: `assets/boundaries/*` retrieved **2026-09-16** from the OSM API (relation 1503483),
  30 municipalities (`assets/boundaries/laguna_boundaries.metadata.json:2-6`; `data/README.md:87-102`).
  `data/laguna.geojson` (barangay polygons) carries in-file `valid_on: 2025-02-13`, `version v03`;
  source/URL/retrieval date **NOT FOUND**. `data/barangay_with_classification.json` = PSA PSGC API,
  release **Q2_2024**, count 681; retrieval date **NOT FOUND** (`reference.py:1-10`; ADR 0016:12-14).
- `road_class == "major"`: the assigning code is **NOT FOUND**; the field ships pre-classified.
  Effective mapping derived from the data values = `{motorway, motorway_link, trunk, trunk_link,
  primary, primary_link, secondary, secondary_link, tertiary, tertiary_link}` (3,956 of 56,776
  features). Consumer filter `site_candidate_v2.py:251`.

### 6. Grid

- Cell **1,000 m**, CRS **EPSG:32651** (`config.py:20-25`). Built over the derived land boundary =
  province polygon (`assets/boundaries/laguna_province.geojson`) minus water
  (`assets/osm/laguna_water.geojson`) — `spatial/grid.py:14-21`, `pipeline.py:63-68`.
- Water exclusion: retain `land_coverage >= 0.5` (`grid.py:57-72`; `MIN_LAND_COVERAGE=0.5`,
  `config.py:26`; ADR 0004:29-32). Inclusive boundary — exactly 50% land is kept.
- Total retained cells **1,764** (`data/processed/rural/model_summary.json:3`;
  `grid_road_distance_summary.json:2`; ADR 0004:39).
- Rural/urban/unclassified = **1,041 / 693 / 30** (`data/processed/rural/model_summary.json:19-26`;
  `docs/technical-summary.md:40`; ADR 0016:31-32). Classification by overlay with PSA-classified
  barangay polygons, rural share ≥ 0.5; unclassified when no barangay overlaps
  (`rural/classify.py:61-95,128-131`; `RURAL_AREA_SHARE_THRESHOLD=0.5`, `config.py:35`).
- 500 m / 2 km sensitivity was run **only for the market/performance model**
  (`experiments/grid_size/EXP-M-500m` grid_cells 7029; `EXP-M-2000m` 441;
  `experiments/results.csv:25-26`; `experiments/FINAL_REPORT.md:66-74`). It was **never run for the
  rural presence screen or site_candidate_v2** — docs say unimplemented on this branch
  (ADR 0016:69-72; `docs/CODE-TOUR.md:191-193`; `docs/DEFENSE-QA.md:166-182`). For v2: **NOT FOUND**.
- Unclassified cells in v2: **excluded from both training and scoring** — training filter
  `presence.py:74-75`, scoring filter `site_candidate_v2.py:1064-1068` and `1104-1105`, stated at
  `site_candidate_v2.py:1145`. (Contrast: the baseline presence screen trains without them but
  scores all 1,764 cells — `presence.py:179-187`.)

### 7. Negative sampling

- The draw happens once at `presence.py:82`:
  `pool.sample(n=len(listed), random_state=config.RANDOM_STATE)`, with `RANDOM_STATE = 42`
  (`config.py:61`). v2 reads the locked sample, asserts 110/110, and only records the seed
  (`site_candidate_v2.py:475-496,1143`).
- Never repeated with a different seed — **NOT FOUND**. The only non-42 `random_state` values are
  unrelated input shuffles in `tests/test_suitability_inputs.py:59,69,75`.

### 8. Saved outputs

- No v2 random-forest full-grid score file existed before this work — **NOT FOUND**. The only v2
  full-grid file was `data/processed/rural/site_candidate_v2_scores.parquet`, whose `model_name` is
  solely `"v2_logistic_regression"`: `_fit_and_score` (`site_candidate_v2.py:1058-1092`) fits only
  the model returned by `_selected_v2_model_name` (1050-1055) = logistic, written at line 1207.
  RF values lived only in the 220-row `site_candidate_v2_oof_predictions.parquet` and the
  report/CSV metrics. The map reads `site_candidate_v2_scores.parquet` and hard-asserts
  `model_name == {"v2_logistic_regression"}` (`scripts/build_visualisation.py:222-224`).

## Part 2 — v2 random-forest full-grid scores

Settings are unchanged from the stored model: `n_estimators=200`, `max_depth=10`,
`max_features="sqrt"`, `min_samples_leaf=1`, `min_samples_split=2` (`config.py:89-95`) plus
`class_weight="balanced"`, `random_state=42`, `n_jobs=1` (`presence.py:52-58`). The forest is fit on
all 220 locked rows and scores all 1,041 rural-with-municipality cells.

- New file: `data/processed/rural/site_candidate_v2_rf_scores.parquet` (1,041 rows,
  `model_name = "v2_random_forest"`, same 20-column layout as the logistic score grid).
- `data/processed/rural/site_candidate_v2_report.json` gains a `thesis_model` field:
  `{ "name": "v2_random_forest", "stored_selection_kept": "v2_logistic_regression", ... }`.
  `selected_v2_model`, `adopted_for_map`, and every metric are left exactly as stored.
- No logistic output was overwritten: `site_candidate_v2_scores.parquet` md5 is unchanged
  (`e1002d0ad83f1126697d39173ef28ce9`) before and after the run.
- Code: `src/str_suitability/modeling/site_candidate_v2_rf.py`,
  `scripts/build_site_candidate_v2_rf_scores.py`, tests in `tests/test_site_candidate_v2.py`.

## Part 3 — seed robustness

Ten seeds redraw the 110 empty rural squares; for each seed the six-feature v2 random forest and
the three-feature baseline forest are scored with the same municipality-grouped 5-fold scheme
(`_group_splits` → `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`). All seeds are
reported, including unfavorable ones; no seed is selected as the winner.

- Files: `data/processed/rural/site_candidate_v2_seed_robustness.csv` and `.md`.
- Code: `src/str_suitability/modeling/site_candidate_v2_seed_robustness.py`,
  `scripts/run_site_candidate_v2_seed_robustness.py`.
- Seed 42 reproduces the stored result exactly (v2 `0.727426` vs stored `0.7274258287458627`;
  baseline `0.634136`), confirming the reimplementation matches the stored method.

| seed | v2 ROC-AUC | baseline ROC-AUC | gain | folds with positive gain |
| --- | --- | --- | --- | --- |
| 42 | 0.727 | 0.634 | 0.093 | 5 of 5 |
| 7 | 0.621 | 0.549 | 0.072 | 5 of 5 |
| 13 | 0.704 | 0.687 | 0.016 | 3 of 5 |
| 21 | 0.703 | 0.633 | 0.070 | 4 of 5 |
| 99 | 0.737 | 0.668 | 0.068 | 3 of 5 |
| 101 | 0.763 | 0.676 | 0.087 | 5 of 5 |
| 202 | 0.712 | 0.630 | 0.082 | 5 of 5 |
| 303 | 0.632 | 0.549 | 0.083 | 5 of 5 |
| 404 | 0.709 | 0.608 | 0.102 | 5 of 5 |
| 505 | 0.639 | 0.597 | 0.043 | 4 of 5 |

- v2 ROC-AUC across seeds: mean **0.695**, range **0.621–0.763**.
- Mean gain over baseline across seeds: mean **0.072**, range **0.016–0.102**.
- Every seed has a positive mean gain; the weakest is seed 13 (0.016, 3 of 5 folds positive).

## Verification

- `uv run pytest -q`: **235 passed** (24 warnings, pre-existing).
- Logistic score parquet md5 `e1002d0ad83f1126697d39173ef28ce9` unchanged by both runs.

## Could not verify

- `poi_laguna.json` provenance (compiler, upstream sources, URLs, criteria, date) — NOT FOUND.
- `laguna_roads.geojson` extraction date/source URL — NOT FOUND; the code assigning `road_class`
  major — NOT FOUND (mapping derived from data values only).
- `data/laguna.geojson` source/URL/retrieval date — NOT FOUND (only in-file `valid_on 2025-02-13`/v03).
- `barangay_with_classification.json` retrieval date — NOT FOUND (only release Q2_2024).
- Any 500 m/2 km run of `presence.py`/`site_candidate_v2.py` — NOT FOUND.
- Any negative-draw repeat with a seed other than 42 in the stored pipeline — NOT FOUND (this
  report's Part 3 adds a deliberate 10-seed robustness check as a separate output).
- Full stored `summarize_target_dataset` output — not persisted; the 732/71 numbers come from
  ADR 0006/docs, not a stored report.
