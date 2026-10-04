# Data — provenance and schema

The raw datasets are **not committed to Git** (`data/` is ignored). This file records what was
retrieved, when, in what shape, and what is known to be wrong with it, so the pipeline stays
reproducible without redistributing the payload.

## AirROI — short-term-rental listings

| Property | Value |
| --- | --- |
| Source | AirROI API, `/listings/search/polygon` |
| Province | Laguna, Philippines |
| Retrieved | 2026-08-18T14:14:15Z (`laguna_listings.json`), 2026-08-18T14:24Z (`history/`), 2026-08-17T13:20Z (`search_responses/`) |
| Currency | Native (PHP) |
| Records | 2,080 listings, all with unique `listing_id` |
| Municipalities covered | 29 of 30 — Pakil returned no listings |
| Coordinates | No nulls, no zeros; extent 13.99322–14.56415 N, 121.01127–121.59980 E |

### Files

- `laguna_listings.json` — 19MB. `metadata` plus a `listings` array of 2,080 flattened records,
  each retaining the full API payload under `raw_listing`.
- `municipality_listing_counts.json` — per-municipality listing counts.
- `search_responses/<municipality>/page_NNNN.json` — 17MB of raw paginated polygon-search
  responses, one directory per municipality, each with `metadata.json` recording the PSGC code
  and whether paging completed.
- `history/<listing_id>.json` — 15MB, one monthly panel per listing: `date`, `occupancy`,
  `average_daily_rate`, `rev_par`, `revenue`, `min_nights`. 50,068 monthly rows covering
  2021-08 to 2026-07.

### Flattened listing schema

Identity and location: `listing_id`, `psgc_code`, `municipality`, `airroi_search_municipality`,
`latitude`, `longitude`, `spatial_join_municipality`, `spatial_join_psgc_code`.

Property: `property_type`, `room_type`, `bedrooms`, `bathrooms`, `accommodates`, `min_nights`.

Reputation: `rating`, `number_of_reviews`, `host`, `superhost`.

Targets: `ttm_revenue`, `ttm_occupancy`, `ttm_avg_rate`, `ttm_revpar`,
`ttm_adjusted_occupancy`, `ttm_adjusted_revpar`, `ttm_available_days`, `ttm_days_reserved`,
`ttm_avg_min_nights`, `ttm_avg_length_of_stay`, and `l90d_*` equivalents.

Provenance: `currency`, `retrieved_at`, `airroi_endpoint`, `raw_listing`.

### Known data-quality issues

1. **Coordinates are privacy-jittered** by up to roughly 150m. This sets the minimum sensible
   grid resolution and is the reason it is fixed at 1km.
2. **`ttm_revenue = ttm_occupancy = 0` for 732 listings (35.2%).** These were verified against
   the monthly panel as *genuine dormancy*, not a reporting gap: the panel agrees with the TTM
   field in every case, with zero contradictions in either direction. 573 of the 616 listings
   with no monthly rows inside the trailing-twelve-month window have earlier revenue and panels
   that simply stop. A positive `ttm_avg_rate` or historical reviews co-occurring with zero
   revenue is not contradictory — the rate is a list price, and reviews may preut it preserves research history and evidence for earlier findings, so keep or archive it if you may need to cite or reproduce those experiments.date dormancy.
3. **`l90d_*` is sparser than `ttm_*`** — 1,049 records (50.4%) hold zeros, against 732 for TTM.
4. **The trailing-twelve-month window is 2025-08 to 2026-07**, established empirically: summing
   panel revenue over those twelve months reproduces `ttm_revenue` with a median ratio of 0.9969
   across the 686 listings whose window is complete.
5. **The panel omits months entirely rather than reporting them as zero** — 16,466 of 50,068
   rows are genuine zero-revenue months, so a missing month is missing data, not a zero.
6. **Panels are near-always contiguous** (1,737 of 1,933 partial panels), consistent with a
   listing's history beginning when it first became active.
7. **`raw_listing.location_info.region` is unreliable** — it contains Laguna, Calabarzon, Rizal,
   Batangas, Bulacan, Cavite, Metro Manila and Pangasinan across the same province-wide set, and
   `locality` does not match municipality names. Municipal attribution comes from the per-municipality
   polygon search, so `spatial_join_municipality` restates that polygon rather than a true
   administrative-boundary join; it must be recomputed against real boundaries.
8. **Property attributes are incomplete**: `bedrooms` is null for 300 records, `accommodates` for 169.

## PSA — population

| Property | Value |
| --- | --- |
| Source | PSA 2024 Census of Population (POPCEN) |
| Reference date | 2024-07-01 |
| Records | 30 municipalities/cities |
| Provincial total | 3,687,345 |

Municipal-level population counts only. **No barangay breakdown, no PSGC codes, and no
geometries.** Filename is `PSA.json` (not `data/raw/psa/`); its own `notes` field records that
Siniloan and Victoria were added from the same census.

Consequence for the study: the demographic indicator is constant across the cells of each
municipality, and cannot be refined without acquiring barangay-level data and boundaries.

## OSM — boundaries retrieved, features outstanding

Administrative boundaries are snapshotted under `assets/boundaries/` (committed, since they are
small and reproducibility depends on them). They were retrieved from the OSM API on
2026-09-16, province relation 1503483, covering the province and its 30 municipalities.

| Property | Value |
| --- | --- |
| Source | OpenStreetMap API, `api.openstreetmap.org/api/0.6` relation geometry |
| Files | `laguna_province.geojson`, `laguna_municipalities.geojson`, `laguna_boundaries.metadata.json` |
| Municipalities | 30, all reconciling exactly with the 30 PSA names |
| Administrative area | 2,108.4 km², tiled exactly by the 30 municipalities |
| Land area | roughly 1,918 km² — the difference is Laguna de Bay inside the boundary |

The municipal relations carry `ref` (PSGC code) and `population` (2024-07-01) tags matching the
PSA table, which supplies the PSGC codes `PSA.json` itself does not hold.

Still to retrieve: POIs, tourism features, roads, and the water polygons needed to derive the land
boundary. Overpass was unreachable (HTTP 504 on every attempt), so these require either a recovered
Overpass or a local parse of a Geofabrik extract — see `docs/adr/0008-osm-extraction-route.md`.
