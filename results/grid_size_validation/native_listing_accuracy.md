# Accuracy on each grid's own listing cells

Each grid is tested on the rural cells that contain a listing at that size. The actual class is listed. A forest trained on the other municipalities scores the cell. The predicted class is listed when the score is at or above that fold's Youden cutoff. A cell is correct only when the model also says listed. The rule is the same at every size. The denominator is that grid's own listing-cell count.

| Grid size | Rural cells with a listing | Empty rural cells | Predicted correctly | Predicted incorrectly | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 500 m | 122 | 4000 | 100 | 22 | 82.0% |
| 1000 m | 110 | 931 | 90 | 20 | 81.8% |
| 1500 m | 90 | 380 | 74 | 16 | 82.2% |
| 2000 m | 71 | 195 | 43 | 28 | 60.6% |

**1500 m is the most accurate on its own listing cells, at 82.2% (74 of 90).**
