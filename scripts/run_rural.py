import json

from str_suitability.rural.pipeline import run_rural_pipeline
from str_suitability.rural.prepare import ensure_rural_inputs

if __name__ == "__main__":
    ensure_rural_inputs()
    print(json.dumps(run_rural_pipeline(), indent=2, default=str))
