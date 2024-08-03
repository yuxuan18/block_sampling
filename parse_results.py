import json
from collections import defaultdict
import numpy as np

def parse(input_filename: str, output_filename: str, 
          error_rates: list=[0.01, 0.025, 0.05, 0.075, 0.1], n_run: int=100):
    with open(input_filename) as f:
        errors = {
            rate: [] for rate in error_rates
        }
        costs = {
            rate: [] for rate in error_rates
        }
        for line in f.readlines():
            result = json.loads(line)
            error_rate = result["error_rate"]
            if error_rate in error_rates and len(errors[error_rate]) < n_run:
                errors[error_rate].append(result["error"])
                costs[error_rate].append(result["cost"])
    with open(output_filename + "_error.json", "w") as f:
        json.dump(errors, f, indent=4)
    with open(output_filename + "_cost.json", "w") as f:
        json.dump(costs, f, indent=4)

parse("block_twitter_count_new.jsonl", "results/block_twitter_count_new")
