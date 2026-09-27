import json

from str_suitability.rural.pipeline import run_rural_pipeline

if __name__ == "__main__":
    print(json.dumps(run_rural_pipeline(), indent=2, default=str))
