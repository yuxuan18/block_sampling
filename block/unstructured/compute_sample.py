import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from block.statistics.estimation import uniform_sample_size_avg, block_sample_size_avg_predicate, uniform_sample_size_avg_predicate

def block_sample_avg(pilot_sample_size: int, failure_rate: float, error_rate: float):
    block_count = np.load("data/fast_reproduce/block_count.npy")
    block_sum = np.load("data/fast_reproduce/block_sum.npy")

    assert len(block_count) == len(block_sum)

    block_stats = pd.DataFrame({"block_count": block_count, "block_sum": block_sum})
    pilot_sample = block_stats.sample(pilot_sample_size, replace=False)
    pilot_counts = pilot_sample["block_count"].to_list()
    pilot_sums = pilot_sample["block_sum"].to_list()
    sample_size = block_sample_size_avg_predicate(len(block_stats), pilot_sample_size,
                                                  pilot_sample_size, pilot_counts, pilot_sums,
                                                  failure_rate, error_rate)
    
    sample = block_stats.sample(sample_size, replace=False)
    result = sample["block_sum"].sum() / sample["block_count"].sum()
    groundtruth = block_sum.sum() / block_count.sum()
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

def uniform_sample_avg_predicate(pilot_sample_size: int, failure_rate: float, error_rate: float):
    labels = pd.read_csv("data/fast_reproduce/tweets_label.csv")
    tweets = pd.read_csv("data/fast_reproduce/twitter.csv", low_memory=False)
    tweets["index"] = tweets.index
    tweets["Likes"] = tweets["Likes"].apply(lambda x: int(x) if str(x).isdigit() else 0)
    data = pd.merge(tweets, labels, left_on="index", right_on="tweet_id")

    pilot_sample = data.sample(pilot_sample_size, replace=False)
    success_pilot_sample = pilot_sample[pilot_sample["label"]]["Likes"].to_list()
    sample_size = uniform_sample_size_avg_predicate(len(tweets)*1000, pilot_sample_size, len(success_pilot_sample),
                                                    success_pilot_sample, failure_rate, error_rate)
    final_sample = data.sample(sample_size, replace=True)
    result = final_sample[final_sample["label"]]["Likes"].mean()
    groundtruth = data[data["label"]]["Likes"].mean()
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


def block_sample(pilot_sample_size: int, failure_rate: float, error_rate: float, dataset: str="twitter"):
    if dataset == "twitter":
        block_count = np.load("data/fast_reproduce/block_count_amazon.npy")
    else:
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

def uniform_sample(pilot_sample_size: int, failure_rate: float, error_rate: float, dataset: str="twitter"):
    if dataset == "amazon":
        labels = np.load("data/fast_reproduce/amazon-labels.npy")
        data = labels
    else:
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

def block_sample_fixed_size(sample_size: int, agg: str="avg", dataset: str="twitter"):
    if dataset == "twitter":
        block_count = np.load("data/fast_reproduce/block_count.npy")
        block_sum = np.load("data/fast_reproduce/block_sum.npy")
    else:
        block_count = np.load("data/fast_reproduce/block_count_amazon.npy")
        block_sum = np.load("data/fast_reproduce/block_count_amazon.npy")

    assert len(block_count) == len(block_sum)

    block_stats = pd.DataFrame({"block_count": block_count, "block_sum": block_sum})
    sample = block_stats.sample(sample_size, replace=True)
    if agg == "avg":
        result = sample["block_sum"].sum() / sample["block_count"].sum()
        groundtruth = block_sum.sum() / block_count.sum()
    else:
        result = sample["block_count"].sum() / len(sample) * len(block_stats)
        groundtruth = block_count.sum()
    error = np.abs(result - groundtruth) / groundtruth
    return {
        "sample_size": sample_size,
        "error": error
    }

def uniform_sample_fixed_size(sample_size: int, agg: str="avg", dataset: str="twitter"):
    if dataset == "amazon":
        labels = np.load("data/fast_reproduce/amazon-labels.npy")
        sample = np.random.choice(labels, sample_size, replace=True)
        if agg == "count":
            result = np.sum(sample).item() /  len(sample) * len(labels)
            groundtruth = np.sum(labels).item()
    else:
        labels = pd.read_csv("data/fast_reproduce/tweets_label.csv")
        tweets = pd.read_csv("data/fast_reproduce/twitter.csv", low_memory=False)
        tweets["index"] = tweets.index
        tweets["Likes"] = tweets["Likes"].apply(lambda x: int(x) if str(x).isdigit() else 0)
        data = pd.merge(tweets, labels, left_on="index", right_on="tweet_id")
        if agg == "avg":
            sample = data.sample(sample_size, replace=True)
            result = sample[sample["label"]]["Likes"].mean()
            groundtruth = data[data["label"]]["Likes"].mean()
        else:
            sample = data.sample(sample_size, replace=True)
            result = len(sample[sample["label"]]) / len(sample) * len(data)
            groundtruth = len(data[data["label"]])

    error = np.abs(result - groundtruth) / groundtruth
    return {
        "sample_size": sample_size,
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
