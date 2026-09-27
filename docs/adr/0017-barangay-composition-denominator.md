---
Status: accepted
---

# Barangay composition of a grid cell: shares over the cell area, with an explicit uncovered residual

Chapter 3 scores each grid cell but does not say how a cell that overlaps several barangays is
described, and it does not say what the denominator of a "share of the cell" is. Two earlier
decisions touch the question without settling it. **ADR 0011** answers the analogous municipality
question by largest intersection area, retaining the overlap share as a diagnostic. **ADR 0016**
classifies a cell rural or urban by majority area — but takes the shares over the **barangay-covered**
area, so that a cell falling half in a coverage gap is not pushed toward urban, and it persists only
the dominant barangay. Neither produces a per-barangay breakdown, and the two decisions therefore
leave the composition of a cell unstated. This ADR records it.

**No cell is assigned to a barangay, and none is assigned to its dominant barangay.** The
composition table is one row per `(cell, barangay)` intersection: 4,819 rows over the 1,764 cells.
The largest overlap is retained as a diagnostic column only, because ADR 0016 already carries it as
the rural labels' anchor and a reader needs it to reconcile the two artifacts. It never produces a
share. The reason is measured rather than assumed: **1,393 of the 1,764 cells (79.0%) overlap more
than one barangay**, and one San Pablo City cell overlaps **17**, where the largest single barangay
holds **15.1%** of the cell. Assigning that cell to its largest overlap would misdescribe 85% of it.

**The denominator is the cell's own clipped land area.** ADR 0004 clips every retained cell to the
derived land boundary, so a coastal or boundary cell's area in the dataset is smaller than the
nominal 1 km² — the San Pablo cell above measures 633,064 m². `barangay_percentage` is therefore
`intersection_area_m2 / projected_area_m2 × 100`, the same quantity ADR 0011's municipality share
uses. Dividing by the nominal 1,000,000 m² instead would report a coastal cell as poorly covered by
barangays for a reason that has nothing to do with barangays, and dividing by the covered area would
hide the gaps. Both alternative denominators are **emitted alongside** so the choice can be audited:
`covered_area_share` and `rural_share_of_covered` divide by the barangay-covered area and reproduce
ADR 0016's `rural_area_share` exactly (see below).

**The residual is explicit and is never redistributed.** Shares sum to 100% only where barangays tile
the cell. The area no barangay covers is carried as `uncovered_area_m2` / `uncovered_coverage` rather
than being normalised away, so a reader always sees whether a share is large because the barangay is
large or because the rest of the cell is empty. On the current data: **1,516 cells are wholly covered**
to within floating-point noise, **218 carry a genuine coverage gap**, and **30 cells no PSA barangay
overlaps at all** — the same 30 ADR 0016 calls unclassified, which remain in the grid as the rural
domain requires. The gap totals 53.9 km² against 1,706.5 km² of retained grid (3.2%). The non-PSA
*Sierra Madre Mountains Forest Land* polygon is kept as **57 rows labelled `unclassified`**, not
dropped: an unclassified area is a statement about the PSA's coverage, not about land use.

**Overlaps and precision fail loudly rather than silently.** Barangay polygons that overlap each
other would double-count area, so a cell whose intersections sum above its own area by more than
1e-9 of that area raises instead of producing shares above 100%. Both layers are checked for validity
and repaired with `make_valid` where needed — on the current data neither layer holds an invalid
geometry. All measurement is in **EPSG:32651**, where the area is true. Intersections smaller than
1 m² are **counted and reported as slivers** (2 of them) rather than filtered out, because a silent
area threshold is indistinguishable from a silent assignment rule; the smallest retained share is
2.4e-05% of its cell.

## Consequences

The composition lives in `data/processed/barangay_grid_composition.parquet` (tidy, one row per cell
and barangay: `cell_id`, `barangay_psgc`, `barangay_name`, `polygon_name`, `municipality`,
`municipality_psgc`, `urban_rural`, `intersection_area_m2`, `grid_area_m2`, `barangay_percentage`,
`covered_area_share`) and `data/processed/grid_coverage.parquet` (one row per cell: area by class,
the four grid-area shares, the two covered-area shares, `barangay_count` and the dominant-barangay
diagnostic). `scripts/build_barangay_composition.py` writes both plus a summary JSON, and reads the
rural outputs without modifying them.

**The covered-area shares reproduce ADR 0016 exactly.** Across the 1,734 cells a barangay overlaps,
`rural_share_of_covered` differs from the persisted `rural_area_share` by at most **1.1e-16**, and
`urban_share_of_covered` from `urban_area_share` by at most **2.2e-16** — floating-point identity, no
cell diverging beyond 1e-9. This is the check that the new table and the rural rule are the same
overlay rather than two similar ones.

**The two share families are not interchangeable, and the difference is material.** Taking the
majority-area rule with the grid-area denominator would call **1,025** cells rural against ADR 0016's
**1,041** — 16 cells move, because a cell whose uncovered remainder is large can be majority-rural of
what is covered and not majority-rural of the cell. `rural_coverage` therefore must not be read as
the rural class; the class remains ADR 0016's, computed over the covered area.

Barangay names are joined on PSGC code and never on name, per ADR 0016, and the tidy table shows why:
the San Pablo cell's rows carry PSA names such as `Barangay IV-A` against polygon names such as
`Barangay IV-A (Pob.)`. `barangay_name` is the PSA `area_name`, `polygon_name` is the OSM `adm4_name`,
and both are carried so the two can be compared. One artifact of floating-point division is recorded
rather than hidden: the largest `barangay_percentage` reads 100.00000000000009.

A review of the reader-facing files confirmed this branch changes no model input: the grid, the
barangay polygons, the PSA classification and the persisted rural classification are byte-identical
before and after the run.
