# Conclusion

The study did what it set out to do. It built a rural location-pattern screen for short-term rentals in Laguna from six location measurements, and that screen ranks listed rural cells above empty rural cells in municipalities it was not trained on. The ranking is above chance, and it stays above chance when the draw of empty cells changes. The same tests show that the ranking is carried by the two road distances.

## What was built

The unit of analysis is a 1 km rural cell, not a single property. The grid has 1,764 cells. Of these, 1,041 rural cells have an assigned municipality and are the cells the model scores. A cell is labeled listed when it contains at least one active listing. The rural set has 110 listed cells and 931 empty cells.

The training sample keeps all 110 listed cells and draws 110 empty cells. The seed used for the main tables is 42. Urban cells, unclassified cells, and rural cells without a municipality are outside the scored set.

The inputs are six location measurements, fixed before the forest was fit:

- Landscape places within 5 km.
- Distance to the nearest landscape place.
- Distance to the nearest mapped road of any class.
- Distance to the nearest major road.
- Distance to the nearest town centre.
- Falls and mountains within 5 km.

The 5 km radius is the neighborhood used for the two counts. The four distance measurements are not cut off at 5 km. Listing revenue, occupancy, price, and other listing attributes are not inputs. The listing data supply the label only: whether the cell already contains an active listing.

The forest has 200 trees, a maximum depth of 10, and balanced class weights. These settings were fixed. They were not searched. Each tree sends a cell to a leaf. The leaf’s number is the share of training cells in that leaf that were listed. The forest score is the average of the 200 leaf shares. It is not a count of how many trees voted yes.

That score is a similarity score. A higher score means the cell looks more like the rural cells where an active listing is already observed. It does not measure revenue, occupancy, zoning, flood risk, or investment return. The map colors at 0.40 and 0.60 are display cuts so the map can be read in three colors. They were not estimated from the data, and they are not the classification cutoff.

## The ranking succeeded

The test that matches the claim is municipality holdout. Whole municipalities are left out of training, and the forest is scored only on those unseen towns. On the seed-42 draw, the mean ROC-AUC across the five folds is 0.727. The five fold values are 0.583, 0.736, 0.760, 0.721, and 0.837. Chance is 0.50. A mean of 0.727 means the forest ranks a listed cell above an empty cell well more often than a coin toss, in towns it did not see while fitting.

One draw of the 110 empty cells is not the whole result. The same five-fold test was repeated with ten draws, using seeds 7, 13, 21, 42, 99, 101, 202, 303, 404, and 505. The ten-draw mean ROC-AUC is 0.695, with draws from 0.621 to 0.763. A separate run that keeps every empty rural cell, with no draw, has ROC-AUC 0.695. The seed-42 figure of 0.727 is the draw used in the main tables. The figure that stays in place when the empty cells change is 0.695. Both are above chance. The ranking is real, and it is repeatable.

The five seed-42 folds are not equal. They run from 0.58 to 0.84. Listed and empty scores overlap, so some listed cells score low and some empty cells score high. That overlap is why the screen is a ranking of similarity and not a yes-or-no decision on an unlabeled cell. The study still succeeds on the question it asked: whether location measurements alone can rank listed rural cells above empty ones in unseen municipalities. They can.

## What the ranking uses

The shuffle test asks which of the six measurements the ranking actually uses. In each held-out fold, one measurement is scrambled and the drop in ROC-AUC is recorded. A larger drop means the ranking relied more on that measurement.

On the seed-42 draw, the drops are:

| Measurement | Mean drop in ROC-AUC |
| --- | ---: |
| Distance to the nearest major road | 0.114 |
| Distance to the nearest mapped road | 0.080 |
| Landscape places within 5 km | 0.018 |
| Distance to the nearest town centre | 0.018 |
| Distance to the nearest landscape place | 0.014 |
| Falls and mountains within 5 km | 0.010 |

The two road distances carry the ranking. The landscape and town-centre measurements barely move it. This matches the training sample. Listed cells sit closer to roads than the drawn empty cells: mean distance to the nearest road is 0.14 km for listed cells and 0.38 km for empty cells, and mean distance to the nearest major road is 0.44 km and 1.11 km. The landscape counts differ much less. Falls and mountains within 5 km average 2.75 in both groups.

The same road result holds when the empty-cell draw changes, and it holds at the other grid sizes that were checked. The two road distances are the top two features in 9 of 10 draws at 1 km, in 10 of 10 draws at 500 m, and in 7 of 10 draws at 1.5 km. They are also the top two when every empty cell is kept. Landscape measurements do not take over the ranking at those sizes.

The six measurements were specified before fitting. The shuffle test did not select them. It shows, after the fact, which of the specified measurements the ranking uses. The other four stay in the model because they were part of the planned description of the cell. A forest that sees only the two road distances has a ten-draw ROC-AUC of 0.674, against 0.695 for all six. The gap is smaller than the change from one empty-cell draw to the next, which is consistent with the small shuffle drops on the non-road measurements.

A shorter road distance is associated with the listed pattern. It is not shown to cause a listing. The road file contains on the order of 55,000 segments, but the model never sees that count. Each cell receives one nearest-road distance and one major-road distance. The road distances may reflect settlement patterns. That is a reading of the association, not a separate test.

## What the score is for

The screen is successful as a ranking of location similarity. An empty rural cell with a higher score is more similar, on these six measurements, to rural cells that already have an active listing. The score can be used to order empty rural cells for a closer look. It cannot be read as the chance that a listing will appear, as expected revenue or occupancy, or as a judgment of zoning, flood safety, or investment return.

The map that accompanies the model colors only empty rural cells, using the display cuts 0.40 and 0.60. Those cuts make the map readable. The evaluated result is the ROC-AUC, which does not depend on them.
