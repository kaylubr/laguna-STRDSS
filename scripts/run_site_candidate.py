"""Run the Random Forest rural site-similarity workflow."""

from str_suitability.modeling.site_candidate import run_site_candidate


if __name__ == "__main__":
    result = run_site_candidate()
    metrics = result["metrics"]["roc_auc"]
    print(f"Random Forest mean municipality-fold ROC-AUC: {metrics['mean']:.3f}")
    print(f"Scores written for {result['population']['rural_cells_scored']} rural cells")