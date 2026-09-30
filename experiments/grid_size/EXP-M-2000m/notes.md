# EXP-M-2000m

Same five features and same model settings as the production 1 km grid, but the grid itself is rebuilt at 2000 m using the generic build_grid() function (str_suitability.spatial.grid), which config.py already anticipates via SENSITIVITY_CELL_SIZES_M = (500, 2000).

Grid cells: 441. Labeled cells: 160 (vs. 288 at 1 km). Municipalities represented: 27.

Grouped-CV macro F1: 0.2944. LOMO mean macro F1: 0.2242, std 0.2137 across 27 municipality folds.

A smaller cell holds fewer listings per cell (sparser surrounding-market signal per cell but more cells), and a larger cell pools more listings into fewer, coarser cells. Both change the labeled sample size, which changes the difficulty of the validation problem itself, not just the model.
