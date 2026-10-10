# Road-feature importance at three grid sizes

Each number is the drop in ROC-AUC after shuffling one feature in the held-out cells, averaged over 20 shuffles and then over the five municipality folds. The balanced rows draw an equal number of empty cells. The 1 km seed-42 drops and the 1 km all-empty road drops matched the published values.

## 500 m, ten balanced draws

| Rank | Feature | Mean drop | Lowest draw | Highest draw | Seed 42 drop | Seed 42 folds that fell |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Nearest mapped road | 0.118 | 0.074 | 0.154 | 0.121 | 5 of 5 |
| 2 | Nearest major road | 0.071 | 0.052 | 0.084 | 0.058 | 5 of 5 |
| 3 | Nearest landscape place | 0.010 | -0.015 | 0.045 | 0.015 | 4 of 5 |
| 4 | Nearest town centre | 0.008 | -0.010 | 0.036 | 0.032 | 5 of 5 |
| 5 | Falls and mountains within 5 km | 0.003 | -0.007 | 0.015 | 0.012 | 4 of 5 |
| 6 | Landscape places within 5 km | -0.005 | -0.016 | 0.004 | 0.004 | 3 of 5 |

## 500 m, all empty cells

| Rank | Feature | Mean drop | Std across folds | Folds that fell |
| ---: | --- | ---: | ---: | ---: |
| 1 | Nearest mapped road | 0.109 | 0.023 | 5 of 5 |
| 2 | Nearest major road | 0.069 | 0.044 | 5 of 5 |
| 3 | Nearest town centre | 0.007 | 0.025 | 4 of 5 |
| 4 | Nearest landscape place | 0.007 | 0.017 | 4 of 5 |
| 5 | Landscape places within 5 km | 0.004 | 0.008 | 3 of 5 |
| 6 | Falls and mountains within 5 km | -0.009 | 0.037 | 3 of 5 |

## 1000 m, ten balanced draws

| Rank | Feature | Mean drop | Lowest draw | Highest draw | Seed 42 drop | Seed 42 folds that fell |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Nearest major road | 0.081 | 0.020 | 0.124 | 0.114 | 5 of 5 |
| 2 | Nearest mapped road | 0.072 | 0.034 | 0.163 | 0.080 | 5 of 5 |
| 3 | Falls and mountains within 5 km | 0.009 | -0.015 | 0.036 | 0.010 | 4 of 5 |
| 4 | Nearest town centre | 0.008 | -0.020 | 0.040 | 0.018 | 4 of 5 |
| 5 | Landscape places within 5 km | 0.008 | -0.013 | 0.022 | 0.018 | 4 of 5 |
| 6 | Nearest landscape place | -0.007 | -0.041 | 0.014 | 0.014 | 3 of 5 |

## 1000 m, all empty cells

| Rank | Feature | Mean drop | Std across folds | Folds that fell |
| ---: | --- | ---: | ---: | ---: |
| 1 | Nearest major road | 0.077 | 0.032 | 5 of 5 |
| 2 | Nearest mapped road | 0.062 | 0.016 | 5 of 5 |
| 3 | Falls and mountains within 5 km | 0.020 | 0.018 | 4 of 5 |
| 4 | Landscape places within 5 km | 0.002 | 0.017 | 3 of 5 |
| 5 | Nearest landscape place | -0.002 | 0.011 | 2 of 5 |
| 6 | Nearest town centre | -0.011 | 0.033 | 2 of 5 |

## 1500 m, ten balanced draws

| Rank | Feature | Mean drop | Lowest draw | Highest draw | Seed 42 drop | Seed 42 folds that fell |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Nearest mapped road | 0.060 | 0.019 | 0.095 | 0.063 | 5 of 5 |
| 2 | Nearest major road | 0.059 | 0.014 | 0.133 | 0.046 | 4 of 5 |
| 3 | Nearest town centre | 0.017 | -0.024 | 0.051 | 0.022 | 4 of 5 |
| 4 | Nearest landscape place | 0.016 | -0.009 | 0.046 | 0.006 | 3 of 5 |
| 5 | Landscape places within 5 km | 0.003 | -0.015 | 0.014 | 0.000 | 2 of 5 |
| 6 | Falls and mountains within 5 km | -0.002 | -0.016 | 0.016 | 0.016 | 4 of 5 |

## 1500 m, all empty cells

| Rank | Feature | Mean drop | Std across folds | Folds that fell |
| ---: | --- | ---: | ---: | ---: |
| 1 | Nearest mapped road | 0.074 | 0.048 | 5 of 5 |
| 2 | Nearest major road | 0.065 | 0.061 | 4 of 5 |
| 3 | Nearest town centre | 0.016 | 0.021 | 4 of 5 |
| 4 | Landscape places within 5 km | 0.004 | 0.036 | 4 of 5 |
| 5 | Falls and mountains within 5 km | 0.002 | 0.017 | 3 of 5 |
| 6 | Nearest landscape place | 0.001 | 0.035 | 4 of 5 |

## Do the two road features stay on top?

500 m, balanced: the two road features are the top two in 10 of 10 runs.
1000 m, balanced: the two road features are the top two in 9 of 10 runs.
1500 m, balanced: the two road features are the top two in 7 of 10 runs.
500 m, all_empty: the two road features are the top two in 1 of 1 runs.
1000 m, all_empty: the two road features are the top two in 1 of 1 runs.
1500 m, all_empty: the two road features are the top two in 1 of 1 runs.

A feature other than the two road distances entered the top two in these runs:

- 1000 m balanced seed 303 puts Nearest town centre in the top two
- 1500 m balanced seed 101 puts Nearest landscape place in the top two
- 1500 m balanced seed 202 puts Nearest town centre in the top two
- 1500 m balanced seed 404 puts Nearest town centre in the top two

## Plain reading

The two road distances still lead at 500 m and at 1.5 km. No landscape or town-centre feature takes over the average ranking at either size.

At 500 m they are the top two in all 10 draws and in the all-empty run. Every one of the five folds falls when either road column is shuffled, in the seed-42 draw and in the all-empty run. The order changes from the thesis: nearest mapped road is first (mean drop 0.118) and nearest major road is second (0.071). At 1 km the major road is first. The landscape and town-centre drops stay small, and landscape places within 5 km average slightly below zero.

At 1 km the published seed-42 order is unchanged: major road 0.114, nearest road 0.080, both positive in all five folds. Across the ten draws they are the top two in 9. Seed 303 is the exception: nearest town centre is second (0.040) and nearest mapped road falls to third (0.038). The all-empty run still has the two roads first, at 0.077 and 0.062.

At 1.5 km the two roads are still first and second on average, and they are almost tied (0.060 and 0.059). They are the top two in 7 of 10 draws and in the all-empty run. In the other three draws, nearest town centre or nearest landscape place takes second place and one road drops out. In seed 101 the nearest landscape place is second, but its drop is 0.026 against 0.133 for the major road, so it passes one road without becoming the main feature. Town centre is the non-road feature that enters the top two most often. The gap between the roads and the other features is narrower here than at 500 m or 1 km.

Seed 42 at 1.5 km is a small inconsistency with the 1 km pattern: shuffling the major road lowers ROC-AUC in 4 of 5 folds, not all 5. The reproduction checks for the published 1 km numbers passed. No count check failed.

