import json

from str_suitability.modeling.presence import run_presence_screen


def main() -> None:
    report = run_presence_screen()
    forest = report["forest"]
    guess = report["stratified_baseline"]
    earnings = report["earnings_per_bedroom"]
    print(json.dumps(
        {
            "listed": report["listed_rural_squares"],
            "sampled_unlisted": report["unlisted_rural_squares_sampled"],
            "scheme": forest["scheme"],
            "forest_auc": forest["roc_auc"],
            "guess_auc": guess["roc_auc"],
            "forest_accuracy": forest["accuracy"],
            "guess_accuracy": guess["accuracy"],
            "separates_better_than_random": report["separates_better_than_random"],
            "confusion": forest["confusion_matrix"],
            "earnings_squares": earnings["squares"],
            "earnings_correlations": earnings["correlations"],
            "earnings_regression": earnings["grouped_regression"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
