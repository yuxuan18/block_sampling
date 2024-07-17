from src.text.page_access_parquet import (S3FileReader, build_local_index, sample_data_pages)
from src.text.compute_sample import uniform_sample
from src.util.timer import Timer
from src.util.clean import clean_none
from src.text.inference import sentiment_analysis, entity_recognition

import pyarrow as pa
from tqdm import tqdm
import pyarrow.parquet as pq
import s3fs
import pandas as pd

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
        probs = n_data / 9468393 / replicate
        n_pages = int((1 - (1-probs) ** 6400) * 1501 * replicate) + 1
        print(f"Expecting to read {n_pages} pages")
        read_file_func = s3_file_reader.read_file
        file_index, meta_data = build_local_index(read_file_func)
        sample = sample_data_pages([file_index]*replicate, [meta_data]*replicate,
                                   n_data, [read_file_func]*replicate)
    timer.check("sample_data_pages")

    # entities = entity_recognition(sample["Tweet"].to_list())
    clean_none(sample, "Tweet")
    sentiments = sentiment_analysis(sample["Tweet"][:n_data].to_list(), batch_size=1024)
    timer.check("inference")

    print(timer.get_records())

def sample_only(pilot_sample_size: int, failure_rate: float, error_rate: float):
    result = uniform_sample(pilot_sample_size, failure_rate, error_rate)
    print(result)

if __name__ == "__main__":
    sample_only(1000, 0.05, 0.01)

