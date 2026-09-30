# EXP-N-binary_high

Collapses the three-class target to binary: top 25% by historical_performance_score = High (1), everything else = Not High (0). Class counts: {'NotHigh': 216, 'High': 72}.

Grouped-CV macro F1: 0.4963, High precision 0.2491, High recall 0.3386. LOMO mean macro F1: 0.5442, std 0.2400 across 28 municipality folds.

This metric is NOT directly comparable to the three-class macro F1 above (fewer classes makes the classification problem easier by construction); it answers a different question ("is this square High or not") than the three-class screener.
