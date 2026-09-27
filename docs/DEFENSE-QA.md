# Defense question drill

A self-test. Read the question, answer it aloud, then check. Every answer names where the evidence
lives. Companion file: `docs/CODE-TOUR.md`.

---

## Data

**1. You started with 2,080 listings and fitted on 1,231. Where did the rest go?**
732 were dormant — trailing-twelve-month revenue *and* occupancy both exactly zero, meaning the
property was not operating, and Chapter 3 excludes those because a zero there reflects the
property's status rather than the result of its operations. A further 71 report positive revenue
that no month inside the August 2025 to July 2026 window corroborates; Chapter 3 asks for such
inconsistencies to be corrected where possible, and for these it is not possible, so their targets
are treated as missing. That leaves the validated population of 1,277. Then each listing is placed
in its containing cell and 46 fall outside the retained grid — 17 jittered onto water, 26 in cells
dropped by the majority-land rule, 3 outside the province. `preprocess/clean_airroi.py`, ADR 0006.

**2. What makes a listing "active"?**
Non-zero trailing-twelve-month revenue *and* occupancy, plus corroboration of that revenue from the
monthly panel. That is Chapter 3's own definition.

**3. Why did two municipalities need codes from PSA rather than OSM?**
The OSM relations for Pila and Victoria carry no `ref` tag, so a code join would have dropped them
entirely. They are filled from an explicit, documented PSA PSGC mapping — not inferred from the
neighbouring codes in the sequence. All 30 now carry a code and a population.
`reference.py`, ADR 0011.

**4. Is OSM under-mapping rural areas, which would bias your map?**
I tested it. Across the 30 municipalities, POIs per km² correlate with population density at +0.832,
but POIs per 1,000 *people* correlate at −0.204 and attractions per 100,000 people at −0.465. Rural
municipalities have **more** amenities and attractions per person, not fewer, so there is no evidence
of systematic under-mapping. The urban gradient comes from density being measured per unit area.

**5. You have 313 POIs that count for nothing.**
They fall outside the retained grid — on majority-water cells or beyond it — so they contribute to
no cell's density. 17,967 of 18,280 contribute; the viewer marks the excluded ones hollow.

## Method

**6. Why 1km and not something finer?**
Airbnb jitters coordinates by up to about 150 metres, so a finer cell would resolve the jitter rather
than geography. It is a data-imposed floor, not a precision choice. ADR 0001.

**7. Why no bedrooms, guest capacity or property type as predictors?**
Chapter 3 requires predictors to be location and demographic "rather than specific property
characteristics", and that is exactly what lets the trained models be applied to cells that contain
no listing. Property attributes would make grid-level prediction impossible for the 1,476 cells with
no observation. The cost is a lower R², deliberately accepted. ADR 0005.

**8. Why Haversine rather than projected distances?**
Chapter 3 specifies the Haversine formula for straight-line distances between two points and
restricts accessibility to point-based locations — the nearest transport facility and the nearest
tourist attraction. No road-based distance is specified, so none is computed. EPSG:32651 is used for
the grid itself; `spatial/haversine.py` for the distances.

**9. A cell can overlap several municipalities. How was it assigned?**
By largest overlap: a polygon overlay, and the cell takes the municipality it intersects most.
508 of 1,764 cells straddle a boundary, some intersecting four. A "wholly within" rule was tried
first and left 522 cells with no municipality at all and therefore no demographic value.
ADR 0011.

**10. Chapter 3 says extreme values are screened for outliers. You did not screen them.**
Correct, and it is deliberate. Revenue reaches 8.3 million against a median of 141,767, and the
observations in that tail are the high performers that define the top suitability class. An IQR
fence would delete them and would improve RMSE and R² only because the hardest observations had been
removed. No trimming is applied and the distribution is reported instead. ADR 0007.

**11. Which coordinate system, and why?**
EPSG:32651, UTM zone 51N, for all metric work, with the grid anchored on integer UTM kilometres so
500m and 2km cells nest inside the 1km grid. Chapter 3 says only "a common projected coordinate
system", so the choice is recorded in ADR 0004.

**12. Chapter 3 says 1,928 cells. Your code produces 1,764.**
The administrative extent is 2,108.4 km², but 355 km² of it is water — Laguna de Bay alone accounts
for 330 km² — so the derived land boundary is 1,753.3 km². The grid retains cells with at least 50%
land coverage. 1,928 sits between the two candidate rules and matches neither: majority-land gives
1,764, any-intersection gives 2,006, and the chapter's figure is closest to PSA's 1,918 km² land
area, which is a figure about land rather than a count of cells. Recorded as a deviation in
ADR 0004; Chapter 3 needs updating.

## Models

**13. What is your split and cross-validation protocol?**
A random 80:20 split with a fixed seed. The grid search runs entirely inside the training split with
5-fold cross-validation, comparing candidates on cross-validated RMSE. The test set is untouched
during tuning and used once for the reported metrics. ADR 0010.

**14. Which hyperparameters, and how were the values chosen?**
`n_estimators` [200, 500], `max_depth` [None, 10, 20], `min_samples_split` [2, 5],
`min_samples_leaf` [1, 2], `max_features` ["sqrt", 0.5] — 48 candidate sets, 240 fits per model.
Deliberately modest: with about 1,020 training rows a larger grid would be selecting on
cross-validation noise. Revenue selected 500 trees at depth 10; occupancy 200 trees at depth 10.
Chapter 3 names the parameters but gives no values, so the grid is recorded in ADR 0010.

**15. Your occupancy model has a test R² of 0.038. Isn't that useless?**
It is the observed performance of the specified model, and it is a finding: occupancy is barely
explicable from location and municipal demographics alone. Adding property characteristics would
raise it, but they are excluded by the design, and including them would break prediction for the
1,476 cells with no listing. The number stands as reported rather than being tuned away. ADR 0012.

**16. Which predictors actually matter?**
Permutation importance on held-out data. For revenue: recreation POI density, restaurant density,
population density, distance to the nearest transport facility. For occupancy: population density
dominates and everything else is indistinguishable from noise — which is what a municipality-level
variable looks like when it is the only source of between-municipality variation. ADR 0013.

**17. Are the models overfitting?**
Moderately, and it is reported: revenue train R² 0.354 against test 0.258; occupancy 0.241 against
0.038.

## Scoring

**18. How are the weights derived?**
Entropy Weight Method. Each indicator is min-max normalised with its direction applied, converted to
proportions, then Shannon entropy with `k = 1/ln(n)`, where n is the 1,764 spatial units. The degree
of diversification is `1 − e`, and weights are `d/Σd`. A zero proportion contributes nothing to the
entropy sum. `suitability/entropy_weights.py`.

**19. Why must the weights sum to 1?**
It is an identity of the method, and it is asserted in code at a 1e-9 tolerance rather than assumed.
The run reports exactly 1.0.

**20. Why did POI density take 0.8614 of the weight?**
Because entropy weighting rewards *concentration*, not spread. POI density has by far the lowest
entropy, 0.757, since most cells hold few POIs while a handful hold hundreds. The other four
indicators sit at 0.98–0.997, meaning their values are nearly uniform across cells, which EWM reads
as carrying almost no information and effectively discards.

**21. What does a suitability class actually mean?**
A relative ranking within Laguna of predicted STR performance potential combined with spatial
attractiveness. It is not a profitability forecast for any property, and for 1,476 of the 1,764
cells it is not based on any observation at all.

**22. Why `FisherJenks` and not `NaturalBreaks`?**
Because `mapclassify.NaturalBreaks` is not reproducible: on identical input it returned the same
breaks four runs out of five and a different break on the fifth, moving a cell between classes.
`FisherJenks` is the exact Jenks natural-breaks algorithm and returned identical breaks every run.
ADR 0014.

**23. How do you know predictions were not joined to the wrong cells?**
The prediction files carry a `cell_id` written by the producer, and the scoring stage joins on that
key with assertions that it is present, unique, non-null and covers exactly the grid's cells.
Regression tests reverse the rows, shuffle them, shuffle each prediction file independently, and
shuffle the grid, asserting every cell still receives its own prediction. A file without a key fails
loudly. Re-running reproduced the predictions bit-identically. ADR 0015.

## Challenge questions

**24. Your score is 86% one variable. Isn't this just a POI density map?**
Largely yes, and it is the substantive finding rather than a defect to hide. The composite score is a
monotone transform of POI density, which measures urban amenity provision, so the map tracks the
urban–rural gradient and 1,538 of 1,764 cells fall in the lowest class. I tested the obvious remedy —
splitting POI density into Chapter 3's six categories — and tourist-attraction density did become
the single largest weight at 0.217, but the POI bloc's *total* weight rose from 86% to 98%, because
splitting one concentrated indicator into six concentrated ones increases their collective entropy
share. So the split made the dominance worse, and it was recorded rather than adopted.

**25. You rank 1,476 cells that have no listing. On what basis?**
On the models, deliberately: because the predictors are location and demographic rather than
property attributes, the trained models apply to any cell, which is the point of the two-stage
design in Chapter 3. Those values are predicted STR performance potential, not observations, and
both the documentation and the viewer say so. ADR 0012.

**26. Where is your sensitivity analysis?**
Not done. The grid builder accepts other cell sizes but only the 1km grid has been run, so the 500m
and 2km checks Chapter 3 describes are outstanding. It is the largest remaining piece of work.

**27. What would you do differently?**
Three things. Run the 500m and 2km sensitivities, since the result is currently untested for cell
size. Deal with the weight concentration — either fewer POI indicators or a principled floor — so
the score is not 86% one variable. And acquire barangay-level population *with* boundaries, which
would replace a 30-valued municipality proxy with a real within-municipality gradient; the
population numbers alone are useless without the geometry.

## Limitations to volunteer before you are asked

- Only **288 of 1,764 cells** contain an active listing; the rest are predictions.
- The **occupancy model's test R² is 0.038**. The revenue model's is 0.258. Both are reported
  untrimmed, on a heavy-tailed target.
- The **500m and 2km sensitivity runs and the six spatial outputs are not produced**.
- **Chapter 3 is not synchronised** with the code — the 1,928-cell figure, the ~64-per-municipality
  figure, the outlier-screening sentence and the administrative-boundary extent all differ.
- **Chapter 3's file is malformed** and its 29 equation images are absent from the repository.
