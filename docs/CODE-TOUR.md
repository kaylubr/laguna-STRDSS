# Code tour

A guided path through this build, for revision before a defense. Companion file:
`docs/DEFENSE-QA.md`.

## What the system does

Ten stages, in the order they run:

1. Preprocess each source separately — Group 2 below
2. Build the validated target population — `preprocess/clean_airroi.py`
3. Build the grid — `spatial/grid.py`
4. Assign rasters to spatial units — `spatial/join.py`
5. Engineer the feature groups — `features/`
6. Train two Random Forests — `modeling/train.py`
7. Evaluate and rank features — `modeling/evaluate.py`, `modeling/importance.py`
8. Predict for every cell — `pipeline.py`
9. Normalise and weight five indicators — `suitability/`
10. Classify into five classes — `suitability/classify.py`

## The one exercise that matters

Trace a **single grid cell** from raw data to its class, out loud. If you can narrate that, you know
the code.

```
data/laguna_listings.json ─► ingest/load_airroi.py ─► preprocess/clean_airroi.py ─► 1,231 listings
data/PSA.json ─────────────► preprocess/clean_psa.py ─┐
POIs/water from the PBF ───► ingest/load_osm_pbf.py ──┼─► spatial/join.py ─► cell attributes
assets/boundaries/* ───────► ingest/load_osm.py ──────┘
                             spatial/grid.py ───► 1,764 cells
                             features/*.py ─────► 10 predictors
                             modeling/*.py ─────► predicted revenue + occupancy
                             suitability/*.py ──► weights ─► score ─► class
```

---

## Reading order

Dependency order. Group 1 has no dependencies; each later group builds on those before it.

### Group 0 — orientation
`README.md` (the ten stages) · `CONTEXT.md` (the glossary — this is the vocabulary a panelist will
use) · `data/README.md` (what the raw data is and what is known to be wrong with it) · `docs/adr/`.

### Group 1 — vocabulary, no dependencies
- **`config.py`** — every constant in one place: EPSG:32651 projected and EPSG:4326 geographic, the
  1km primary grid, `MIN_LAND_COVERAGE = 0.5`, the TTM window, `TEST_SIZE`, `CV_FOLDS`,
  `RANDOM_STATE`, `PARAM_GRID`, the two targets, and the five suitability indicators with their
  directions. Any "what value did you use" question is answered here.
- **`taxonomy.py`** — the mechanical OSM tag→category mapping: six POI categories with declared
  precedence, the tourist-attraction whitelist, and a transport-facility definition deliberately
  *narrower* than the transportation POI category, so a bus stop is a POI but not a facility.
- **`reference.py`** — the PSA-sourced PSGC codes for Pila and Victoria, which OSM omits. Small, but
  it is the difference between 28 and 30 municipalities receiving demographics.

### Group 2 — getting data in
- **`ingest/load_airroi.py`** — reads the flattened listings JSON and the monthly history panel.
- **`preprocess/clean_airroi.py`** — **the most-asked-about file.** Turns 2,080 listings into 1,231:
  flags the 732 dormant (revenue *and* occupancy exactly zero) and the 71 whose positive TTM revenue
  no month in the window corroborates, reporting the validated 1,277. Seven validation assertions
  live here.
- **`preprocess/clean_psa.py`** — loads population, normalises municipality names, reconciles them
  against the boundary snapshot, fills the two missing PSGC codes, attaches codes to PSA rows.

### Group 3 — geometry
- **`ingest/load_osm.py`** — builds Laguna and its 30 municipalities from OSM API relation geometry,
  assembling rings from member ways; validates the 30 against PSA.
- **`ingest/load_osm_pbf.py`** — parses the Geofabrik extract into POIs and water polygons,
  bounding-box filtered, driven by `taxonomy.py`; fetches relation-based water geometry by ID.
- **`spatial/haversine.py`** — the great-circle distance exactly as Chapter 3 specifies, plus a
  chunked nearest-neighbour routine.
- **`spatial/grid.py`** — derives the land boundary (province minus water), generates 1km cells
  anchored on integer UTM kilometres, retains cells at ≥50% land coverage, and provides
  `assert_one_row_per_cell`, which runs after every join.
- **`spatial/join.py`** — assigns each cell to the municipality it overlaps **most**, joins
  demographics, assigns listings to their containing cell, normalises PSGC codes.

### Group 4 — features
- **`features/poi.py`** — per-category POI counts and densities (count ÷ cell area).
- **`features/accessibility.py`** — Haversine distances to the nearest transport facility and the
  nearest tourist attraction, plus attraction count and density.

### Group 5 — models
- **`modeling/train.py`** — the 80:20 split and the cross-validated grid search.
- **`modeling/evaluate.py`** — MAE, RMSE, R² and the untrimmed target distribution.
- **`modeling/importance.py`** — permutation importance on the held-out test set.

### Group 6 — scoring
- **`suitability/validate.py`** — pre-EWM assertions: one row per cell, all five indicators present,
  none incomplete; plus the 0–1 range check.
- **`suitability/normalize.py`** — min-max with the direction applied per indicator.
- **`suitability/entropy_weights.py`** — proportions, entropy (`k = 1/ln(n)`), diversification,
  weights, with the weight-sum assertion.
- **`suitability/composite_score.py`** — `S = Σ wⱼzⱼ`.
- **`suitability/classify.py`** — `FisherJenks` into five classes with the fixed labels.
- **`suitability/inputs.py`** — the **keyed** join of grid cells to the prediction files.
- **`suitability/stage.py`** — runs the steps in order and returns the report.

### Group 7 — orchestration and entry points
- **`pipeline.py`** — runs everything end to end into `data/processed/`.
- **`rural/`** — the rural analytical domain (ADR 0016): `psa.py` (PSA `urban_rural`, joined on PSGC),
  `classify.py` (majority-area cell class, listing labels, the three population levels),
  `training.py`, `prediction.py`, `suitability.py`, `pipeline.py`. Writes `data/processed/rural/`.
- **`scripts/`** — `osm_snapshot.py` (PBF extraction), `run_suitability.py` (scoring stage),
  `run_rural.py` (rural branch), `diagnose_indicators.py` (indicator diagnostics gate),
  `audit_pois.py` (POI/weight stress test), `build_visualisation.py` + `visualisation_template.html`
  (scratch viewer).

---

## The ADR index — your answer key

`docs/adr/`, nineteen files, each recording one decision Chapter 3 leaves open. Read them all; they
are short, and together they are the "why" layer over the "what".

| ADR | The question it answers |
|---|---|
| 0001 | Why 1km and not finer? Coordinates are jittered by up to ~150m |
| 0002 | Why train on listings but predict per grid cell? |
| 0003 | Why is population constant within a municipality? |
| 0004 | Which CRS, and which cells count as land? |
| 0005 | Why are property characteristics excluded? |
| 0006 | Why 1,231 listings and not 2,080? |
| 0007 | Why is there no outlier trimming? |
| 0008 | Why the OSM API for boundaries instead of Overpass? |
| 0009 | Why a local PBF extract, and why join on PSGC code? |
| 0010 | Where the hyperparameters and the 80:20 sampling came from |
| 0011 | How a cell straddling two municipalities is assigned |
| 0012 | What a predicted cell value actually means |
| 0013 | Why municipality identity and listing density stay out |
| 0014 | Why `FisherJenks` rather than `NaturalBreaks` |
| 0015 | How the prediction-to-cell join is guaranteed correct |
| 0016 | What makes a listing rural, what makes a cell rural, and why the two differ |
| 0017 | What the denominator of a cell's barangay share is |
| 0018 | Why the two rural definitions are allowed to disagree |
| 0019 | Why the composite indicator set was expanded, and why weighting is now EWM blended with RFI |

---

## Numbers to know cold

| Quantity | Value |
|---|---|
| Listings retrieved | 2,080 |
| Dormant (revenue and occupancy both zero) | 732 |
| Uncorroborated positive TTM revenue | 71 |
| **Validated target population** | **1,277** |
| Outside the retained grid | 46 |
| **Fitted listings** | **1,231** |
| Cells containing a listing | 288 of 1,764 |
| Grid cells | 1,764 |
| Administrative extent vs derived land | 2,108.4 km² vs 1,753.3 km² |
| Water inside the province | 355.1 km², of which Laguna de Bay 330.0 |
| Municipalities | 30, all with demographics |
| Cells straddling municipal boundaries | 508, up to 4 each |
| POIs total / counted in a cell / outside | 18,280 / 17,967 / 313 |
| Tourist attractions / transport facilities | 370 / 143 |
| Revenue model, test — grouped CV (primary) | MAE 277,723 · RMSE 483,881 · **R² 0.2322** (n 152) |
| Revenue model, test — random split (before) | **R² 0.2390** (n 247) |
| Occupancy model, test — grouped CV (primary) | MAE 0.0981 · RMSE 0.1381 · **R² −0.0688** (n 152) |
| Occupancy model, test — random split (before) | **R² 0.0322** (n 247) |
| Rural revenue, grouped vs random | **0.0035 vs 0.524** — the 0.524 was leakage |
| Chosen hyperparameters | revenue: 200 trees, depth 10 · occupancy: 200 trees, depth 10 |
| Composite weighting | hybrid EWM + RFI over 13 indicators (ADR 0019) |
| POI-bloc weight, province | EWM 0.970 · **hybrid 0.686** · equal 0.462 |
| POI-bloc weight, rural | EWM 0.971 · **hybrid 0.585** |
| Largest single hybrid weight | province 0.229 (`poi_density_recreation`) · rural 0.160 (`poi_density_tourist_attraction`) |
| Composite score range | 0.0314 – 0.6006 |
| Class counts | 140 / 394 / 627 / 542 / 61 |
| Test suite | 202 passing |

## Commands worth knowing

```bash
uv sync                                  # install the environment
uv run pytest -q                         # the test suite
uv run python scripts/run_suitability.py # scoring stage only
uv run python scripts/run_rural.py       # rural analytical domain
uv run python -c "from str_suitability.pipeline import run_pipeline; run_pipeline()"
```

Outputs land in `data/processed/`: `grid_suitability.parquet` (one row per cell, carrying the score
and class), `model_summary.json`, `suitability_summary.json`, `training_observations.parquet`.

---

## What is **not** built — say these before you are asked

1. **Sensitivity runs at 500m and 2km.** Chapter 3 states the analysis was repeated at those
   resolutions; only 1km exists. The grid builder accepts other cell sizes, so this is work not yet
   done rather than work that cannot be done.
2. **The six spatial outputs.** No maps are produced as part of the build.
3. **Chapter 3 is not synchronised with the code.** It still says 1,928 cells and ~64 per
   municipality, that extreme values are screened for outliers, and that the extent is the
   administrative boundary. The full reconciliation is in the audit.
4. **The occupancy model barely predicts** (grouped test R² −0.069), and only 288 of 1,764 cells have
   any observation behind them.
5. **`docs/thesis-chr3.md` is malformed.** Its newlines were replaced by non-breaking spaces, so it
   is one 27 KB line, and its 29 equation images are not in the repository at all.
6. **The POI audit's raw-tag analytics cannot run on the shipped extract.** `scripts/audit_pois.py`
   needs `poi_tag_key` / `poi_tag_value` / `is_catch_all` columns that `laguna_pois.parquet` does not
   carry (the committed extractor never emitted them), so it runs the category-based EWM stress test
   only and reports the tag inventory as unavailable. Regenerating the extract from the local PBF
   would risk the study's POI counts.
7. **The two predictions are diluted in the hybrid.** They carry no permutation importance, so the
   blend leaves them at roughly 0.003 each; the composite leans on the spatial indicators. Recorded
   in ADR 0019, not hidden.
