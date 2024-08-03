from block.unstructured.page_access_parquet import (S3FileReader, build_local_index, 
                                                    sample_data_pages)
from block.unstructured.compute_sample import block_sample, block_sample_avg
from block.unstructured.page_access_parquet import seek_random_pages
import argparse
import json
import multiprocessing


def perf_only(n_blocks: int, bucket_name: str, file_name: str, replicate: int=1000):
    s3_file_reader = S3FileReader(bucket_name, file_name)
    read_file_func = s3_file_reader.read_file

    # build file index
    file_index, meta_data = build_local_index(read_file_func)

    # sample data pages
    # using with clause to stop a function in 60 seconds
    process = multiprocessing.Process(target=sample_data_pages, 
                                      args=([file_index]*replicate, [meta_data]*replicate,
                                            n_blocks, [read_file_func]*replicate))
    process.start()
    process.join(timeout=60)
    if process.is_alive():
        process.terminate()
        process.join()
        print("Timeout")
    else:
        print("Finished")

def sample_only(pilot_sample_size: int, failure_rate: float, error_rate: float, agg: str="count", dataset: str="twitter"):
    print(f"Executing sample only experiment ...")
    if agg == "count":
        result = block_sample(pilot_sample_size, failure_rate, error_rate, dataset)
    elif agg == "avg":
        result = block_sample_avg(pilot_sample_size, failure_rate, error_rate)
    return result

def fix_time(timeout: int, bucket_name: str, file_name: str):
    s3_file_reader = S3FileReader(bucket_name, file_name)
    read_file_func = s3_file_reader.read_file

    # build file index
    file_index, meta_data = build_local_index(read_file_func)

    n_pages = seek_random_pages(timeout, file_index=file_index, read_file_func=read_file_func, meta_data=meta_data)
    return {
        "timeout": timeout,
        "n_pages": n_pages
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="perf", help="mode to run")
    parser.add_argument("--pilot_sample_size", type=int, default=100, help="pilot sample size")
    parser.add_argument("--failure_rate", type=float, default=0.05, help="failure rate")
    parser.add_argument("--error_rate", type=float, default=0.01, help="error rate")
    parser.add_argument("--agg", type=str, default="count", help="aggregation type")
    parser.add_argument("--n_blocks", type=int, default=1000, help="number of blocks to sample")
    parser.add_argument("--bucket_name", type=str, default="test-cidr", help="bucket name")
    parser.add_argument("--file_name", type=str, default="tweet.parquet", help="file name in the bucket")
    parser.add_argument("--output", type=str, default="output.json", help="output file name")
    parser.add_argument("--cost_file", type=str, default="", help="cost file name")
    parser.add_argument("--dataset", type=str, default="twitter", help="dataset name")
    parser.add_argument("--timeout", type=int, default=60, help="timeout in seconds")
    args = parser.parse_args()

    if args.mode == "perf":
        if args.cost_file != "":
            with open(args.cost_file) as f:
                costs = json.load(f)
            for error in costs:
                print("error rate:", error)
                for cost in costs[error][:20]:
                    result = perf_only(int(cost), args.bucket_name, args.file_name)
        else:
            result = perf_only(args.n_blocks, args.bucket_name, args.file_name)
    elif args.mode == "sample":
        result = sample_only(args.pilot_sample_size, args.failure_rate, args.error_rate, args.agg, dataset=args.dataset)
    elif args.mode == "timeout":
        result = fix_time(args.timeout, args.bucket_name, args.file_name)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    with open(args.output, "a+") as f:
        f.write(json.dumps(result) + "\n")
