# Does the road-distance finding hold at other grid sizes?

The thesis finding is that the two road-distance features dominate the six-feature random forest. This test asks whether that finding still holds when the grid is 500 m or 1.5 km, and when the draw of empty cells changes.

The 1 km row uses the original thesis grid. The published seed-42 drops and the published all-empty road drops were reproduced before the other sizes were run.

## How the drop is measured

The forest is unchanged: 200 trees, depth 10, square root of the features at each split, minimum one sample per leaf, minimum two samples per split, balanced class weights, and no tuning. Validation is the same five municipality folds.

For each fold, one feature is shuffled in the held-out cells. The shuffle is repeated 20 times. The recorded value is the average drop in ROC-AUC. A larger drop means the ranking relied more on that feature. A negative drop means shuffling it did not hurt the ranking.

Two samples are used at each grid size.

- **Balanced.** All listed rural cells, plus an equal number of empty rural cells. Repeated for seeds 7, 13, 21, 42, 99, 101, 202, 303, 404, and 505.
- **All empty.** Every empty rural cell, once, with no draw.

## Ten balanced draws

The mean is the average of the ten draws. The lowest and highest are the smallest and largest draw. The seed-42 column is the single draw used in the thesis tables. The last column counts how many of that draw's five folds fell when the feature was shuffled.

### 500 m

| Rank | Feature | Mean drop | Lowest draw | Highest draw | Seed 42 | Folds that fell |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Nearest mapped road | 0.118 | 0.074 | 0.154 | 0.121 | 5 of 5 |
| 2 | Nearest major road | 0.071 | 0.052 | 0.084 | 0.058 | 5 of 5 |
| 3 | Nearest landscape place | 0.010 | −0.015 | 0.045 | 0.015 | 4 of 5 |
| 4 | Nearest town centre | 0.008 | −0.010 | 0.036 | 0.032 | 5 of 5 |
| 5 | Falls and mountains within 5 km | 0.003 | −0.007 | 0.015 | 0.012 | 4 of 5 |
| 6 | Landscape places within 5 km | −0.005 | −0.016 | 0.004 | 0.004 | 3 of 5 |

### 1 km

| Rank | Feature | Mean drop | Lowest draw | Highest draw | Seed 42 | Folds that fell |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Nearest major road | 0.081 | 0.020 | 0.124 | 0.114 | 5 of 5 |
| 2 | Nearest mapped road | 0.072 | 0.034 | 0.163 | 0.080 | 5 of 5 |
| 3 | Falls and mountains within 5 km | 0.009 | −0.015 | 0.036 | 0.010 | 4 of 5 |
| 4 | Nearest town centre | 0.008 | −0.020 | 0.040 | 0.018 | 4 of 5 |
| 5 | Landscape places within 5 km | 0.008 | −0.013 | 0.022 | 0.018 | 4 of 5 |
| 6 | Nearest landscape place | −0.007 | −0.041 | 0.014 | 0.014 | 3 of 5 |

Seed 42 matches the published drops: major road 0.114, nearest road 0.080, landscape places within 5 km 0.018, town centre 0.018, nearest landscape place 0.014, and falls and mountains 0.010.

### 1.5 km

| Rank | Feature | Mean drop | Lowest draw | Highest draw | Seed 42 | Folds that fell |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Nearest mapped road | 0.060 | 0.019 | 0.095 | 0.063 | 5 of 5 |
| 2 | Nearest major road | 0.059 | 0.014 | 0.133 | 0.046 | 4 of 5 |
| 3 | Nearest town centre | 0.017 | −0.024 | 0.051 | 0.022 | 4 of 5 |
| 4 | Nearest landscape place | 0.016 | −0.009 | 0.046 | 0.006 | 3 of 5 |
| 5 | Landscape places within 5 km | 0.003 | −0.015 | 0.014 | 0.000 | 2 of 5 |
| 6 | Falls and mountains within 5 km | −0.002 | −0.016 | 0.016 | 0.016 | 4 of 5 |

## All empty cells

One run at each size. No empty cells are drawn. The standard deviation is across the five folds.

### 500 m

| Rank | Feature | Mean drop | Std across folds | Folds that fell |
| ---: | --- | ---: | ---: | ---: |
| 1 | Nearest mapped road | 0.109 | 0.023 | 5 of 5 |
| 2 | Nearest major road | 0.069 | 0.044 | 5 of 5 |
| 3 | Nearest town centre | 0.007 | 0.025 | 4 of 5 |
| 4 | Nearest landscape place | 0.007 | 0.017 | 4 of 5 |
| 5 | Landscape places within 5 km | 0.004 | 0.008 | 3 of 5 |
| 6 | Falls and mountains within 5 km | −0.009 | 0.037 | 3 of 5 |

### 1 km

| Rank | Feature | Mean drop | Std across folds | Folds that fell |
| ---: | --- | ---: | ---: | ---: |
| 1 | Nearest major road | 0.077 | 0.032 | 5 of 5 |
| 2 | Nearest mapped road | 0.062 | 0.016 | 5 of 5 |
| 3 | Falls and mountains within 5 km | 0.020 | 0.018 | 4 of 5 |
| 4 | Landscape places within 5 km | 0.002 | 0.017 | 3 of 5 |
| 5 | Nearest landscape place | −0.002 | 0.011 | 2 of 5 |
| 6 | Nearest town centre | −0.011 | 0.033 | 2 of 5 |

These two road drops match the published all-empty values, 0.077 and 0.062.

### 1.5 km

| Rank | Feature | Mean drop | Std across folds | Folds that fell |
| ---: | --- | ---: | ---: | ---: |
| 1 | Nearest mapped road | 0.074 | 0.048 | 5 of 5 |
| 2 | Nearest major road | 0.065 | 0.061 | 4 of 5 |
| 3 | Nearest town centre | 0.016 | 0.021 | 4 of 5 |
| 4 | Landscape places within 5 km | 0.004 | 0.036 | 4 of 5 |
| 5 | Falls and mountains within 5 km | 0.002 | 0.017 | 3 of 5 |
| 6 | Nearest landscape place | 0.001 | 0.035 | 4 of 5 |

## How often the two roads are the top two

| Grid | Balanced draws | All-empty run |
| --- | ---: | ---: |
| 500 m | 10 of 10 | Yes |
| 1 km | 9 of 10 | Yes |
| 1.5 km | 7 of 10 | Yes |

The runs where another feature entered the top two:

| Grid | Seed | Feature that entered | What happened |
| --- | ---: | --- | --- |
| 1 km | 303 | Nearest town centre | Second, at 0.040. Nearest mapped road fell to third, at 0.038. |
| 1.5 km | 101 | Nearest landscape place | Second, at 0.026. The major road stayed first, at 0.133. |
| 1.5 km | 202 | Nearest town centre | Second, at 0.051. The major road stayed first, at 0.076. |
| 1.5 km | 404 | Nearest town centre | Second, at 0.016. Nearest mapped road stayed first, at 0.081. The major road fell to third, at 0.014. |

## What the test says

The two road distances still dominate at 500 m. They are the top two in every balanced draw and in the all-empty run, and shuffling either one lowers ROC-AUC in all five folds of the seed-42 draw. The order is not the same as the thesis. At 500 m the nearest mapped road is first and the major road is second. At 1 km the major road is first. The landscape and town-centre drops stay near zero. Landscape places within 5 km average a small negative drop, so shuffling that column does not hurt the ranking.

At 1.5 km the two roads are still first and second on average, and they are still the top two when every empty cell is used. The lead is weaker. The two road drops are almost the same, 0.060 and 0.059, and a town-centre or landscape feature enters the top two in 3 of the 10 draws. In the clearest of those draws the intruding feature is still far below the leading road: at seed 101 the nearest landscape place drops ROC-AUC by 0.026, while the major road drops it by 0.133. Town centre is the non-road feature that enters the top two most often. It does not replace the roads as the main signal.

One fold pattern does not carry over. At 1 km, seed 42, both road features lower ROC-AUC in all five folds. At 1.5 km, seed 42, the major road does so in 4 of 5 folds.

The 1 km reproduction checks passed. No other check failed.
