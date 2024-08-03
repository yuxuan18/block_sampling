from block.unstructured.page_access_parquet import (S3FileReader, build_local_index, sample_data_pages, seek_random_record)
from block.unstructured.compute_sample import uniform_sample, uniform_sample_avg_predicate
from block.util.timer import Timer
from tqdm import tqdm

import pyarrow.parquet as pq
import s3fs
import pandas as pd
import argparse
import json
import numpy as np
import multiprocessing

def perf_only(n_data: int, bucket_name: str, file_name: str, 
              replicate: int=1000, fetch_pages: bool=True):
    s3_file_reader = S3FileReader(bucket_name, file_name)
    timer = Timer()
    timer.start()

    # sample data pages
    if not fetch_pages:
        sample = pd.DataFrame()
        print(f"Downloading all files ...")
        for _ in tqdm(range(replicate)):
            s3_path = f's3://{bucket_name}/{file_name}'
            s3 = s3fs.S3FileSystem()
            table = pq.read_table(s3_path, filesystem=s3)
            df = table.to_pandas()
            df_sample = df.sample(n_data // 1000)
            sample = pd.concat([sample, df_sample])
    else:
        sample = np.random.choice(9468393*replicate, n_data)
        n_pages = len(np.unique(sample // 6400))
        print(f"Expecting to read {n_pages} pages")
        read_file_func = s3_file_reader.read_file
        file_index, meta_data = build_local_index(read_file_func)
        process = multiprocessing.Process(target=sample_data_pages,
                                          args=([file_index]*replicate, [meta_data]*replicate,
                                                n_pages, [read_file_func]*replicate))
        process.start()
        process.join(timeout=60)
        if process.is_alive():
            process.terminate()
            process.join()
            print("Timeout")
        else:
            print("Finished")
    timer.check("sample_data_pages")

    # entities = entity_recognition(sample["Tweet"].to_list())
    # clean_none(sample, "Tweet")
    # sentiments = sentiment_analysis(sample["Tweet"][:n_data].to_list(), batch_size=1024)
    # timer.check("inference")

    result = timer.get_records()
    result["n_data"] = n_data

    return result

def sample_only(pilot_sample_size: int, failure_rate: float, error_rate: float, agg: str="count", dataset: str="twitter"):
    if agg == "count":
        result = uniform_sample(pilot_sample_size, failure_rate, error_rate, dataset=dataset)
    elif agg == "avg":
        result = uniform_sample_avg_predicate(pilot_sample_size, failure_rate, error_rate)
    return result

def timeout(bucket_name: str, file_name: str, timeout: int):
    s3_file_reader = S3FileReader(bucket_name, file_name)
    read_file_func = s3_file_reader.read_file

    # build file index
    file_index, meta_data = build_local_index(read_file_func)

    n_records = seek_random_record(timeout, file_index=file_index, read_file_func=read_file_func, meta_data=meta_data)
    return {
        "timeout": timeout,
        "n_records": n_records
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="perf", help="mode to run")
    parser.add_argument("--pilot_sample_size", type=int, default=1000, help="pilot sample size")
    parser.add_argument("--failure_rate", type=float, default=0.05, help="failure rate")
    parser.add_argument("--error_rate", type=float, default=0.01, help="error rate")
    parser.add_argument("--agg", type=str, default="count", help="aggregation type")
    parser.add_argument("--n_data", type=int, default=1000, help="number of data points")
    parser.add_argument("--bucket_name", type=str, default="test-cidr", help="bucket name")
    parser.add_argument("--file_name", type=str, default="tweet.parquet", help="file name")
    parser.add_argument("--output", type=str, default="output.json", help="output file name")
    parser.add_argument("--dataset", type=str, default="twitter", help="dataset name")
    parser.add_argument("--cost_file", type=str, default="", help="cost file name")
    parser.add_argument("--timeout", type=int, default=60, help="timeout in seconds")

    args = parser.parse_args()
    if args.mode == "perf":
        if args.cost_file:
            with open(args.cost_file, "r") as f:
                cost = json.load(f)
            for error in cost:
                print("error:", error)
                for n_data in cost[error][:20]:
                    result = perf_only(n_data, args.bucket_name, args.file_name, fetch_pages=True)
                    with open(args.output, "a+") as f:
                        f.write(json.dumps(result) + "\n")
        else:
            result = perf_only(args.n_data, args.bucket_name, args.file_name, fetch_pages=True)
    elif args.mode == "sample":
        result = sample_only(args.pilot_sample_size, args.failure_rate, args.error_rate, args.agg, dataset=args.dataset)
    elif args.mode == "timeout":
        result = timeout(args.bucket_name, args.file_name, args.timeout)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    with open(args.output, "a+") as f:
        f.write(json.dumps(result) + "\n")
