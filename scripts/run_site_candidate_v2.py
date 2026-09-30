"""Reproduce the site_candidate_v2 comparison.

The command is:

    uv run python scripts/run_site_candidate_v2.py
"""

from str_suitability.modeling.site_candidate_v2 import run_site_candidate_v2


if __name__ == "__main__":
    report = run_site_candidate_v2()
    print(report["research_use_decision"])
    print(report["selected_v2_model"])
    for name, payload in report["matched_deltas"].items():
        print(name, payload["delta_auc_fold"], payload["mean"])
