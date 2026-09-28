# Predicted Airbnb performance classes

The map color is the random forest's predicted class: Low, Moderate, or High. Each cell also has three probabilities, and the class is the largest of them. The color is not a peso amount and it is not a promised return.

The training label uses only cells that already have a listing. Their own mean revenue and occupancy are min-max normalized across those cells and averaged with equal weight. The bottom quarter of that score is Low, the middle half is Moderate, and the top quarter is High. On the last run that was 72, 144, and 72 cells. A cell with no listing is not labeled Low. It is predicted from its surroundings.

Surrounding revenue and occupancy exclude listings inside the cell. If no other listing is within 5 km, those means stay missing. They are not stored as zero.

`poi_laguna.json` contributes the place count within 5 km and the distance to the nearest place. The file is 172 landscape places: falls, mountains, lakes and ponds, and rivers. It is not a general mix of shops and restaurants.

Across five municipality folds, mean macro F1 was 0.314 (SD 0.084). A model that always says Moderate scores 0.221. A model that guesses with the class frequencies scores 0.368. The forest beat the first and not the second. Mean one-vs-rest ROC-AUC was 0.533, the same as the frequency-matched baseline. Averaged across those folds, shuffling any of the five features did not lower macro F1. That is a predictive association, not evidence that a place or a nearby listing caused the revenue. Checks at 1 km and 3 km, with medians, and with tourist features alone are stored beside the model. The map still uses 5 km, means, and all five features.
