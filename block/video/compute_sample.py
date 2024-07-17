import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from block.statistics.estimation import block_sample_size_avg_predicate, uniform_sample_size_avg_predicate

def block_sample(pilot_sample_size: int, failure_rate: float, error_rate: float):
    block_label = pd.read_csv("data/fast_reproduce/seven_block_labels.csv")

    pilot_sample = block_label.sample(n=pilot_sample_size, replace=False)
    success = len(pilot_sample[pilot_sample["block_count"] > 0])
    success_block_counts = pilot_sample[pilot_sample["block_count"] > 0]["block_count"].to_list()
    success_block_sums = pilot_sample[pilot_sample["block_count"] > 0]["block_sum"].to_list()
    sample_size = block_sample_size_avg_predicate(
        len(block_label), pilot_sample_size, success,
        success_block_counts, success_block_sums,
        failure_rate, error_rate
    )
    sample = block_label.sample(n=sample_size, replace=False)
    sample_result = sample["block_sum"].sum() / sample["block_count"].sum()
    groundtruth = block_label["block_sum"].sum() / block_label["block_count"].sum()
    print(groundtruth)
    error = abs(sample_result - groundtruth) / groundtruth
    cost = pilot_sample_size + sample_size
    return {
        "failure_rate": failure_rate,
        "error_rate": error_rate,
        "pilot_sample_size": pilot_sample_size,
        "sample_size": sample_size,
        "cost": cost,
        "error": error
    }

def uniform_sample(pilot_sample_size: int, failure_rate: float, error_rate: float):
    frame_counts = np.load("data/fast_reproduce/seven_days_frame_count.npy")
    pilot_sample = np.random.choice(frame_counts, pilot_sample_size, replace=False)
    success = len(pilot_sample[pilot_sample > 0])
    success_frame_counts = pilot_sample[pilot_sample > 0].tolist()
    sample_size = uniform_sample_size_avg_predicate(
        len(frame_counts), pilot_sample_size, success,
        success_frame_counts, failure_rate, error_rate
    )
    sample = np.random.choice(frame_counts, sample_size, replace=False)
    sample = sample[sample > 0]
    sample_result = sample.sum() / len(sample)
    frame_success = frame_counts[frame_counts > 0]
    groundtruth = frame_success.sum() / len(frame_success)
    print(groundtruth)
    error = abs(sample_result - groundtruth) / groundtruth
    cost = pilot_sample_size + sample_size
    return {
        "failure_rate": failure_rate,
        "error_rate": error_rate,
        "pilot_sample_size": pilot_sample_size,
        "sample_size": sample_size,
        "cost": cost,
        "error": error
    }


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "block":
        for error in [0.01 * i for i in range(1,11)]:
            print("error:", error)
            for _ in tqdm(range(500)):
                result = block_sample(1000, 0.05, error)
                with open("video_block_sample.json", "a+") as f:
                    f.write(json.dumps(result) + "\n")
    elif mode == "uniform":
        for error in [0.01 * i for i in range(1,11)]:
            print("error:", error)
            for _ in tqdm(range(500)):
                result = uniform_sample(1000, 0.05, error)
                with open("video_uniform_sample.json", "a+") as f:
                    f.write(json.dumps(result) + "\n")



