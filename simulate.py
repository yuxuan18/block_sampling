# function 1
# input: pilot sample rate, error rate
# output: time

# function2
# input: time
# output: error rate

from block.unstructured.compute_sample import (uniform_sample, uniform_sample_avg_predicate, block_sample, block_sample_avg,
                                               uniform_sample_fixed_size, block_sample_fixed_size)
import numpy as np
import argparse
import json
from tqdm import tqdm
import scipy

def record2page(n_data: int):
    n_page = 1500 * 1000
    total_n_data = n_page * 6400
    sample = np.random.choice(total_n_data, n_data)
    n_pages = len(np.unique(sample // 6400))
    return n_pages

def get_runtime(result: dict, sampling_method: str):
    if sampling_method == "uniform":
        n_data = result["cost"]
        n_page = record2page(n_data)
        if n_page / 10.6 > 14400:
            runtime = n_data / 1500 / 8 + 14400
        else:
            runtime = n_data / 1500 / 8 + n_page / 10.6
    else:
        n_page = result["cost"]
        runtime = n_page * 6400 / 1500 / 8 + n_page / 10.6
    return runtime

def get_sample_size_by_time(timeout: int, sampling_method: str):
    if sampling_method == "uniform":
        runtime = lambda m: 1500*1000*(1 - (1 - m/1500/1000/6400)**6400) / 10.6 + m / 8 / 1500
        # runtime = timeout, solve for m
        sample_size = scipy.optimize.fsolve(lambda m: runtime(m) - timeout, 1000)[0]
        print("sample size:", sample_size)
        n_page = record2page(int(sample_size))
        if n_page / 10.6 > 14400:
            sample_size = (timeout-14400) * 8 * 1500
    else:
        sample_size = timeout / (1/10.6 + 6400/8/1500)
    return int(sample_size) + 1

def simulate_time(pilot_sample_size: int, error_rate: float, sampling_method: str,
                  agg: str="avg", dataset: str="twitter"):
    if sampling_method == "uniform":
        if agg == "count":
            result = uniform_sample(pilot_sample_size, 0.95, error_rate, dataset=dataset)
        elif agg == "avg":
            result = uniform_sample_avg_predicate(pilot_sample_size, 0.95, error_rate)

    elif sampling_method == "block":
        if agg == "count":
            result = block_sample(pilot_sample_size, 0.95, error_rate, dataset=dataset)
        elif agg == "avg":
            result = block_sample_avg(pilot_sample_size, 0.95, error_rate)
        
    runtime = get_runtime(result, sampling_method)
    return runtime

def simulate_error(runtime: int, sampling_method: str, agg: str="avg", dataset: str="twitter"):
    sample_size = get_sample_size_by_time(runtime, sampling_method)
    print("sample size:", sample_size)
    if sampling_method == "uniform":
        result = uniform_sample_fixed_size(sample_size, agg=agg, dataset=dataset)
    elif sampling_method == "block":
        result = block_sample_fixed_size(sample_size, agg=agg, dataset=dataset)
    return result["error"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="time", help="mode to run")
    parser.add_argument("--pilot_sample_size", type=int, default=1000, help="pilot sample size")
    parser.add_argument("--error_rate", type=float, default=0.01, help="error rate")
    parser.add_argument("--sampling_method", type=str, default="uniform", help="sampling method")
    parser.add_argument("--agg", type=str, default="avg", help="aggregation type")
    parser.add_argument("--dataset", type=str, default="twitter", help="dataset name")
    parser.add_argument("--timeout", type=int, default=60, help="timeout in seconds")
    parser.add_argument("--output", type=str, default="", help="output file name")
    parser.add_argument("--cost_file", type=str, default="", help="cost file name")
    args = parser.parse_args()

    if args.cost_file != "":
        with open(args.cost_file) as f:
            costs = json.load(f)
        for error in costs:
            cost = costs[error]
            costs[error] = []
            for c in tqdm(cost):
                if args.sampling_method == "uniform":
                    n_data = c
                    n_page = record2page(n_data)
                    if n_page / 10.6 > 14400:
                        runtime = n_data / 1500 / 8 + 14400
                    else:
                        runtime = n_data / 1500 / 8 + n_page / 10.6
                    costs[error].append(runtime)
                elif args.sampling_method == "block":
                    n_page = c
                    runtime = n_page * 6400 / 1500 / 8 + n_page / 10.6
                    costs[error].append(runtime)
        with open(args.output, "w") as f:
            json.dump(costs, f, indent=4)
    elif args.mode == "time":
        runtime = simulate_time(args.pilot_sample_size, args.error_rate, args.sampling_method, agg=args.agg, dataset=args.dataset)
        print(f"runtime: {runtime}")
    elif args.mode == "error":
        error = simulate_error(args.timeout, args.sampling_method, agg=args.agg, dataset=args.dataset)
        if args.output == "":
            print(f"error: {error}")
        else:
            with open(args.output, "a+") as f:
                result = {
                    "timeout": args.timeout,
                    "error": error
                }
                f.write(json.dumps(result) + "\n")
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
        