# Lines to remember at the defense

Read the question, then say the answer under it. The numbers are the original 1 km screening model: 110 listed rural cells, 110 empty cells in the training sample, six location features, 200 trees.

## Why these six features?

The six numbers describe the cell. They do not come from the listing’s revenue or occupancy. We specified them before training. We did not run a test that picked them out of a larger list.

- Landscape places within 5 km: how many curated landscape places sit near the cell.
- Distance to the nearest landscape place: how far the closest of those places is. A cell can be near one place and still have few places nearby, so this is not the same number as the count.
- Distance to the nearest mapped road of any class: straight-line kilometres to the nearest road in the Laguna road file. Local roads count. It is not a driving time, and it is not a count of the roads in the file.
- Distance to the nearest major road: straight-line kilometres to the nearest road tagged major. A cell can be next to a local road and still be far from a major road.
- Distance to the nearest town centre: straight-line kilometres to the nearest poblacion.
- Falls and mountains within 5 km: only Falls and Mountains inside 5 km. Rivers, lakes, and ponds stay in the full landscape count.

After the forest was fit, the shuffle test showed what the ranking uses. Shuffling a road distance lowers ROC-AUC. Shuffling a landscape or town-centre number barely changes it. On the training sample, listed and empty cells also differ more on the roads: nearest road 0.14 km versus 0.38 km, nearest major road 0.44 km versus 1.11 km.

If they ask why the other four are still in the model: they were part of the description we specified in advance. A forest that sees only the two road distances has a ten-draw ROC-AUC of 0.674, against 0.695 for all six. That gap is smaller than the change from one empty-cell draw to the next.

## Why 5 km?

Five kilometres is the neighborhood we fixed before training. It is used only for the two counts: landscape places within 5 km, and falls and mountains within 5 km. The other four features are distances to the nearest road, major road, landscape place, or town centre. Those distances are not cut off at 5 km.

We did not search for a radius that maximized ROC-AUC for this forest. Say that plainly if they ask.

## What are the 0.40 and 0.60 colors?

They are display cuts for the map. Below 0.40 is red. From 0.40 up to, but not including, 0.60 is gold. 0.60 or above is green. They are round numbers so the map can be read in three colors. They were not estimated from the data.

ROC-AUC does not use them. The cutoff that is estimated is Youden’s J, and it is only for the yes-or-no classification check. On the seed-42 sample it is different in each held-out fold: 0.566, 0.450, 0.412, 0.381, and 0.357. None of those is 0.40 or 0.60.

The score is also not the chance that an empty cell will get a listing. Training is balanced, 110 listed and 110 empty, while the rural map has 110 listed and 931 empty.

## How does the random forest work?

The forest is 200 trees. Each tree looks at the six location numbers and asks yes-or-no questions, such as whether the major road is closer than some distance. Each question sends the cell left or right. The cell ends in a leaf.

The leaf’s number is the share of training cells in that leaf that were listed. The forest score is the average of the 200 leaf shares. It is not a count of how many trees voted yes.

Settings that were fixed, not searched: 200 trees, depth at most 10, square root of the features considered at each split, balanced class weights, seed 42.

## One tree, as an example

This is a teaching sketch so the idea is easy to say. It is not one of the 200 fitted trees.

```text
Is the nearest major road within 0.7 km?
├── Yes
│   └── Is the nearest mapped road within 0.2 km?
│       ├── Yes → 8 of 10 training cells in this leaf were listed → 0.80
│       └── No  → 4 of 10 were listed → 0.40
└── No  → 2 of 10 were listed → 0.20
```

One tree might give 0.80. Another might give 0.40. If the 200 leaf shares average 0.62, the forest score for that cell is 0.62. On the map, 0.62 is green only because it is at or above the display cut of 0.60.

## Three lines not to say

- Do not say a test selected the six features.
- Do not say 5 km was the radius with the best score.
- Do not say 0.40 and 0.60 are the classification cutoff, or that a green cell is likely to succeed.
