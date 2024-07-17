import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from estimation import uniform_sample_size_avg


def block_sample(pilot_sample_size: int, failure_rate: float, error_rate: float):
    block_count = np.load("data/fast_reproduce/block_count.npy")

    pilot_sample = np.random.choice(block_count, pilot_sample_size, replace=False)
    pilot_mean = np.mean(pilot_sample).item()
    pilot_std = np.sqrt(np.var(pilot_sample, ddof=1)).item()

    sample_size = uniform_sample_size_avg(pilot_mean, pilot_std, pilot_sample_size, failure_rate, error_rate)

    sample = np.random.choice(block_count, sample_size, replace=False)
    result = np.mean(sample).item() * block_count.shape[0]
    groundtruth = np.sum(block_count)
    error = np.abs(result - groundtruth) / groundtruth
    cost = sample_size + pilot_sample_size
    return {
        "failure_rate": failure_rate,
        "error_rate": error_rate,
        "pilot_sample_size": pilot_sample_size,
        "sample_size": sample_size,
        "cost": cost,
        "error": error
    }

def uniform_sample(pilot_sample_size: int, failure_rate: float, error_rate: float):
    labels = pd.read_csv("data/fast_reproduce/tweets_label.csv")
    data = np.array(labels["label"].values.astype(int))
    n_positive = np.sum(data)
    n_data = data.shape[0]
    prob = n_positive / n_data


    # pilot_sample_size times bernoulli trials
    pilot_sample = np.random.binomial(1, prob, pilot_sample_size)
    pilot_mean = np.mean(pilot_sample).item()
    pilot_std = np.std(pilot_sample).item()

    sample_size = uniform_sample_size_avg(pilot_mean, pilot_std, pilot_sample_size, failure_rate, error_rate)

    sample = np.random.binomial(1, prob, sample_size)
    result = np.mean(sample).item()
    groundtruth = np.mean(data)
    error = np.abs(result - groundtruth) / groundtruth
    cost = sample_size + pilot_sample_size
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
        for error in [0.01 * i for i in range(1, 11)]:
            print("error:", error)
            for _ in tqdm(range(500)):
                result = block_sample(100, 0.05, error)
                with open("block_sample_results.json", "a+") as f:
                    f.write(json.dumps(result) + "\n")
    elif mode == "uniform":
        for error in [0.01 * i for i in range(1, 11)]:
            print("error:", error)
            for _ in tqdm(range(500)):
                result = uniform_sample(100, 0.05, error)
                with open("uniform_sample_results.json", "a+") as f:
                    f.write(json.dumps(result) + "\n")
