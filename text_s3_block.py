from block.text.page_access_parquet import (S3FileReader, build_local_index, 
                                          sample_data_pages)
from block.text.compute_sample import block_sample
from block.util.timer import Timer
from block.util.clean import clean_none
from block.text.inference import sentiment_analysis, entity_recognition

def perf_only(n_blocks: int, bucket_name: str, file_name: str, replicate: int=1000):
    s3_file_reader = S3FileReader(bucket_name, file_name)
    read_file_func = s3_file_reader.read_file
    timer = Timer()
    timer.start()

    # build file index
    file_index, meta_data = build_local_index(read_file_func)
    timer.check("build_local_index")

    # sample data pages
    sample = sample_data_pages([file_index]*replicate, [meta_data]*replicate,
                               n_blocks, [read_file_func]*replicate)
    timer.check("sample_data_pages")
    
    entities = entity_recognition(sample["Tweet"].to_list())
    clean_none(sample, "Tweet")
    sentiments = sentiment_analysis(sample["Tweet"].to_list(), batch_size=1024)
    timer.check("inference")

    print(timer.get_records())

def sample_only(pilot_sample_size: int, failure_rate: float, error_rate: float):
    result = block_sample(pilot_sample_size, failure_rate, error_rate)
    print(result)

if __name__ == "__main__":
    sample_only(100, 0.05, 0.01)