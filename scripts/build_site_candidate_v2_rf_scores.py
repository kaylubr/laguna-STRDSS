"""Write the site_candidate_v2 random-forest full-grid scores.

The command is:

    uv run python scripts/build_site_candidate_v2_rf_scores.py

This writes a new score file and adds the thesis_model field to the report. The
logistic outputs and the stored comparison are not changed.
"""

from str_suitability.modeling.site_candidate_v2_rf import (
    annotate_report_with_thesis_model,
    write_random_forest_scores,
)


if __name__ == "__main__":
    scored = write_random_forest_scores()
    report = annotate_report_with_thesis_model()
    print(f"{scored['model_name'].iloc[0]} scored {len(scored)} cells")
    print(f"report thesis_model: {report['thesis_model']['name']}")
