# Accuracy on the 110 existing listing cells

The same 110 rural 1 km cells are tested at every grid size. Each cell already contains an active listing. One listing inside that cell is the test point.

At every size the point is given to the cell that contains it. A forest trained on the other municipalities scores that cell. The predicted class is listed when the score is at or above that fold's Youden cutoff, and empty otherwise. The actual class is listed, because the cell contains the listing. A prediction is correct only when the model also says listed.

The rule does not change between grid sizes. A coarser grid can put more than one of the 110 points in the same cell. Each of the 110 points is still counted once. A point that falls in a cell the grid drops, because less than half of that cell is land, is counted incorrect: there is no kept cell there for the model to call listed.

| Grid size | Listing cells tested | Predicted correctly | Predicted incorrectly | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| 500 m | 110 | 79 | 31 | 71.8% |
| 1000 m | 110 | 90 | 20 | 81.8% |
| 1500 m | 110 | 80 | 30 | 72.7% |
| 2000 m | 110 | 57 | 53 | 51.8% |

**1 km is the most accurate on these 110 listing cells, at 81.8% (90 correct, 20 incorrect).**

At 1.5 km the 110 points fall in 90 cells. At 2 km they fall in 76 cells. Points that share a cell share that cell's prediction, and each point is still counted.

Some points leave the rural cells this model scores. Those count as incorrect at every size.

| Grid size | Model said empty | Listing fell in an urban cell | Cell dropped (under half land) |
| --- | ---: | ---: | ---: |
| 500 m | 19 | 11 | 1 |
| 1 km | 20 | 0 | 0 |
| 1.5 km | 17 | 8 | 5 |
| 2 km | 41 | 7 | 5 |
