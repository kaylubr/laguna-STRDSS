"""Alternative target definitions (experiment group N).

These all reuse the SAME underlying 0..1 historical performance score the current
thesis model computes (0.5 x normalized revenue + 0.5 x normalized occupancy on
cells with at least one listing). Only the percentile cutoffs that turn that score
into classes change. This file does not recompute revenue/occupancy normalization
and does not touch the official 25/75 target used by the production model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def alt_three_class_target(score: pd.Series, low_pct: float, high_pct: float) -> pd.Series:
    """low_pct/high_pct are percentiles (0-100) for the Low/High cutoffs, e.g.
    (20, 80) for a 20/60/20 split or (30, 70) for a 30/40/30 split."""
    low_cut = np.percentile(score, low_pct)
    high_cut = np.percentile(score, high_pct)
    target = pd.Series(1, index=score.index)  # Moderate
    target[score <= low_cut] = 0
    target[score >= high_cut] = 2
    return target


def binary_high_target(score: pd.Series, high_pct: float = 75.0) -> pd.Series:
    """1 = top high_pct percentile (High), 0 = Not High. Top 25% by default, matching
    the production model's High cutoff."""
    high_cut = np.percentile(score, high_pct)
    return (score >= high_cut).astype(int)
