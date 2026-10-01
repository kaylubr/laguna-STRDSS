"""Run the site_candidate_v2 seed robustness check.

The command is:

    uv run python scripts/run_site_candidate_v2_seed_robustness.py

This redraws the empty rural squares at ten seeds and compares the v2 and
baseline forests. It writes a separate CSV and markdown summary. Stored v2
results are not changed.
"""

from str_suitability.modeling.site_candidate_v2_seed_robustness import run_seed_robustness


if __name__ == "__main__":
    frame = run_seed_robustness()
    print(frame[["seed", "v2_roc_auc_mean", "baseline_roc_auc_mean", "gain_mean", "folds_positive_gain"]].to_string(index=False))
