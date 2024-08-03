import io
import os
import boto3
import struct
import random
import thriftpy2
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from typing import Callable, Tuple, Any, List
from thriftpy2.transport import TMemoryBuffer
from thriftpy2.protocol.compact import TCompactProtocol
import time
parquet_thrift = thriftpy2.load("block/unstructured/parquet.thrift", module_name="parquet_thrift")


class ParquetPageLoc:
    def __init__(self, offset, size, first_row_index) -> None:
        self.offset = offset
        self.size = size
        self.first_row_index = first_row_index

class ParquetRowGroupIndex:
    def __init__(self, offset, size, data_page_offset, n_rows) -> None:
        self.offset = offset
        self.size = size
        self.data_page_offset = data_page_offset
        self.n_rows = n_rows
        self.data_page_locs = []
    
    def set_offset(self, offset: int) -> None:
        self.offset = offset
    
    def add_data_page_offset(self, loc: ParquetPageLoc) -> None:
        self.data_page_locs.append(loc)

class ParquetFileIndex:
    def __init__(self):
        self.row_groups = []
    
    def add_row_group(self, row_group: ParquetRowGroupIndex) -> None:
        self.row_groups.append(row_group)
    
    def get_n_pages(self) -> int:
        n_pages = 0
        for row_group in self.row_groups:
            n_pages += len(row_group.data_page_locs)
        return n_pages

    def get_page_loc(self, page_index: int) -> ParquetPageLoc:
        for row_group in self.row_groups:
            if page_index < len(row_group.data_page_locs):
                return row_group.data_page_locs[page_index]
            page_index -= len(row_group.data_page_offsets)
        raise ValueError("page_index out of range")
    
    def print(self) -> None:
        print(f"n_row_groups: {len(self.row_groups)}")
        for i, row_group in enumerate(self.row_groups):
            print(f"\trow_group {i}:")
            print(f"\trow group offset: {row_group.offset}")
            print(f"\trow group size: {row_group.size}")
            print(f"\trow group column data_page_offset: {row_group.data_page_offset}")
            print(f"\trow group n_data_pages: {len(row_group.data_page_locs)}")
            for j, data_page_loc in enumerate(row_group.data_page_locs):
                print(f"\t\tdata_page {j}:")
                print(f"\t\tdata page offset: {data_page_loc.offset}")
                print(f"\t\tdaga page size: {data_page_loc.size}")
                print(f"\t\tdata page first_row_index: {data_page_loc.first_row_index}")

def build_local_index(read_file_func: Callable[[int,int], bytes]) -> Tuple[ParquetFileIndex, Any]:
    # read the foot magic number amd the size of footer
    size_of_footer_bytes = read_file_func(-8, 4)
    size_of_footer = struct.unpack('<I', size_of_footer_bytes)[0]

    # read the footer
    footer_bytes = read_file_func(-size_of_footer-8, size_of_footer)

    # parse the footer into metadata
    transport = TMemoryBuffer(footer_bytes)
    protocol = TCompactProtocol(transport)
    metadata = parquet_thrift.FileMetaData()
    metadata.read(protocol)

    file_index = ParquetFileIndex()
    offset_index_offsets = []
    for row_group in metadata.row_groups:
        size = row_group.total_compressed_size      # same as columns[0].meta_data.total_compressed_size
        row_group_offset = row_group.file_offset    # same as the dictionary page offset
        row_group_metadata = row_group.columns[0]
        column_metadata = row_group_metadata.meta_data
        offset_index_offset = row_group_metadata.offset_index_offset
        offset_index_length = row_group_metadata.offset_index_length
        offset_index_offsets.append(
            (offset_index_offset, offset_index_offset + offset_index_length)
        )
        row_group_index = ParquetRowGroupIndex(row_group_offset, size, column_metadata.data_page_offset, row_group.num_rows)
        file_index.add_row_group(row_group_index)
    
    # the offset_indexes are contiguous
    for i in range(len(offset_index_offsets)-1):
        assert offset_index_offsets[i][1] == offset_index_offsets[i+1][0]

    # read the offset indexes
    offset_index_offset = offset_index_offsets[0][0]
    offset_index_length = offset_index_offsets[-1][1] - offset_index_offsets[0][0]
    offset_indices_bytes = read_file_func(offset_index_offset, offset_index_length)
    for i, offset_index_offset in enumerate(offset_index_offsets):
        length = offset_index_offset[1] - offset_index_offset[0]
        offset_index_bytes = offset_indices_bytes[:length]
        offset_indices_bytes = offset_indices_bytes[length:]
        
        transport = TMemoryBuffer(offset_index_bytes)
        protocol = TCompactProtocol(transport)
        offset_index = parquet_thrift.OffsetIndex()
        offset_index.read(protocol)
        
        for raw_page_loc in offset_index.page_locations:
            page_loc = ParquetPageLoc(raw_page_loc.offset, raw_page_loc.compressed_page_size, raw_page_loc.first_row_index)
            file_index.row_groups[i].add_data_page_offset(page_loc)

    return file_index, metadata

def gen_local_file_reader(file_path: str) -> Callable[[int,int], bytes]:
    def read_file(offset: int, length: int) -> bytes:
        with open(file_path, 'rb') as f:
            if offset < 0:
                f.seek(offset, 2)
            else:
                f.seek(offset)
            return f.read(length)
    
    return read_file

class S3FileReader:
    def __init__(self, bucket_name: str, file_path: str) -> None:
        self.s3 = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'], 
                               aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
        self.bucket_name = bucket_name
        self.file_path = file_path

        self.total_size = self.s3.head_object(Bucket=bucket_name, Key=file_path)['ContentLength']
    
    def read_file(self, offset: int, length: int) -> bytes:
        if offset > 0:
            return self.s3.get_object(Bucket=self.bucket_name, Key=self.file_path, 
                                      Range=f"bytes={offset}-{offset+length-1}")['Body'].read()
        else:
            return self.s3.get_object(Bucket=self.bucket_name, Key=self.file_path, 
                                      Range=f"bytes={self.total_size+offset}-{self.total_size+offset+length-1}")['Body'].read()

    def download_file(self) -> None:
        self.s3.download_file(self.bucket_name, self.file_path, self.file_path)

def sample_data_pages(parquet_files: List[ParquetFileIndex], 
                      meta_data_list: List[Any],
                      sample_size: int,
                      file_readers: List[Callable[[int,int], bytes]]) -> pd.DataFrame:
    n_pages = []
    for file_index in parquet_files:
        n_pages.append(file_index.get_n_pages())
    
    sampled_pages = random.sample(range(sum(n_pages)), k=sample_size)
    sampled_pages_per_file = defaultdict(list)
    for sampled_page in sampled_pages:
        for i, n_page in enumerate(n_pages):
            if sampled_page < n_page:
                sampled_pages_per_file[i].append(sampled_page)
                break
            sampled_page -= n_page
    
    # fetch each file
    all_dfs = []
    print("Downloading data pages from s3 ...")
    for i, sampled_pages in tqdm(sampled_pages_per_file.items()):
        file_index = parquet_files[i]
        read_file_func = file_readers[i]
        meta_data = meta_data_list[i]
        df = read_pages(file_index, read_file_func, meta_data, sampled_pages)
        all_dfs.append(df)
    return pd.concat(all_dfs)

def read_pages(file_index: ParquetFileIndex, read_file_func: Callable[[int,int], bytes], 
               meta_data: Any, page_indices: List[int]) -> pd.DataFrame:
    page_indices_per_row_group = defaultdict(list)
    page_indices = sorted(page_indices)
    for page_index in page_indices:
        for i, row_group in enumerate(file_index.row_groups):
            if page_index < len(row_group.data_page_locs):
                page_indices_per_row_group[i].append(page_index)
                break
            page_index -= len(row_group.data_page_locs)
        
    file_data_bytes = b""
    row_groups_meta_data = []
    for row_group_index, page_indices in page_indices_per_row_group.items():
        row_group_meta_data = meta_data.row_groups[row_group_index]
        row_group_bytes = b""
        row_group = file_index.row_groups[row_group_index]
        # fetch the dictionary page
        row_group_meta_data.columns[0].meta_data.dictionary_page_offset = len(file_data_bytes) + 4
        dictionary_page_offset = row_group.offset
        dictionary_page_length = row_group.data_page_offset - row_group.offset
        row_group_bytes += read_file_func(dictionary_page_offset, dictionary_page_length)
        # fetch the data pages
        row_group_meta_data.columns[0].meta_data.data_page_offset = len(file_data_bytes) + 4
        for page_index in page_indices:
            data_page_loc = row_group.data_page_locs[page_index]
            row_group_bytes += read_file_func(data_page_loc.offset, data_page_loc.size)
        # finalize
        row_group_meta_data.columns[0].meta_data.total_compressed_size = len(row_group_bytes)
        file_data_bytes += row_group_bytes
        row_groups_meta_data.append(row_group_meta_data)
    
    new_meta_data = deepcopy(meta_data)
    new_meta_data.row_groups = row_groups_meta_data

    transport = TMemoryBuffer()
    protocol = TCompactProtocol(transport)
    new_meta_data.write(protocol)
    meta_data_bytes = transport.getvalue()


    len_meta_data = len(meta_data_bytes)
    meta_data_length_bytes = struct.pack('<I', len_meta_data)
    file_data_bytes = b'PAR1' + file_data_bytes + meta_data_bytes + meta_data_length_bytes + b'PAR1'

    # read the bytes as a complete parquet file
    file_buffer = io.BytesIO(file_data_bytes)
    table = pq.read_table(file_buffer)
    df = table.to_pandas()
    return df

def seek_random_pages(timeout: int, file_index: ParquetFileIndex, read_file_func: Callable[[int,int], bytes], 
                      meta_data: Any):
    n_pages = file_index.get_n_pages()
    start = time.time()
    fetched_pages = 0
    while time.time() - start < timeout:
        page_index = random.randint(0, n_pages-1)
        row_group_index = 0
        for i, row_group in enumerate(file_index.row_groups):
            if page_index < len(row_group.data_page_locs):
                row_group_index = i
                break
            page_index -= len(row_group.data_page_locs)

        file_data_bytes = b""
        row_groups_meta_data = []
        row_group_meta_data = meta_data.row_groups[row_group_index]
        row_group_bytes = b""
        row_group = file_index.row_groups[row_group_index]
        # fetch the dictionary page
        row_group_meta_data.columns[0].meta_data.dictionary_page_offset = len(file_data_bytes) + 4
        dictionary_page_offset = row_group.offset
        dictionary_page_length = row_group.data_page_offset - row_group.offset
        row_group_bytes += read_file_func(dictionary_page_offset, dictionary_page_length)
        # fetch the data pages
        row_group_meta_data.columns[0].meta_data.data_page_offset = len(file_data_bytes) + 4
        data_page_loc = row_group.data_page_locs[page_index]
        row_group_bytes += read_file_func(data_page_loc.offset, data_page_loc.size)
        # finalize
        row_group_meta_data.columns[0].meta_data.total_compressed_size = len(row_group_bytes)
        file_data_bytes += row_group_bytes
        row_groups_meta_data.append(row_group_meta_data)

        new_meta_data = deepcopy(meta_data)
        new_meta_data.row_groups = row_groups_meta_data

        transport = TMemoryBuffer()
        protocol = TCompactProtocol(transport)
        new_meta_data.write(protocol)
        meta_data_bytes = transport.getvalue()


        len_meta_data = len(meta_data_bytes)
        meta_data_length_bytes = struct.pack('<I', len_meta_data)
        file_data_bytes = b'PAR1' + file_data_bytes + meta_data_bytes + meta_data_length_bytes + b'PAR1'

        # read the bytes as a complete parquet file
        file_buffer = io.BytesIO(file_data_bytes)
        table = pq.read_table(file_buffer)
        df = table.to_pandas()
        fetched_pages += 1
    return fetched_pages

def seek_random_record(timeout: int, file_index: ParquetFileIndex, read_file_func: Callable[[int, int], bytes],
                       meta_data: Any):
    n_pages = file_index.get_n_pages()
    start = time.time()
    seeked_pages = set()
    n_records = 0
    while time.time() - start < timeout:
        record_index = random.randint(0, n_pages*6400-1)
        page_index = record_index // 6400
        if page_index in seeked_pages:
            n_records += 1
            continue
        row_group_index = 0
        for i, row_group in enumerate(file_index.row_groups):
            if page_index < len(row_group.data_page_locs):
                row_group_index = i
                break
            page_index -= len(row_group.data_page_locs)

        file_data_bytes = b""
        row_groups_meta_data = []
        row_group_meta_data = meta_data.row_groups[row_group_index]
        row_group_bytes = b""
        row_group = file_index.row_groups[row_group_index]
        # fetch the dictionary page
        row_group_meta_data.columns[0].meta_data.dictionary_page_offset = len(file_data_bytes) + 4
        dictionary_page_offset = row_group.offset
        dictionary_page_length = row_group.data_page_offset - row_group.offset
        row_group_bytes += read_file_func(dictionary_page_offset, dictionary_page_length)
        # fetch the data pages
        row_group_meta_data.columns[0].meta_data.data_page_offset = len(file_data_bytes) + 4
        data_page_loc = row_group.data_page_locs[page_index]
        row_group_bytes += read_file_func(data_page_loc.offset, data_page_loc.size)
        # finalize
        row_group_meta_data.columns[0].meta_data.total_compressed_size = len(row_group_bytes)
        file_data_bytes += row_group_bytes
        row_groups_meta_data.append(row_group_meta_data)

        new_meta_data = deepcopy(meta_data)
        new_meta_data.row_groups = row_groups_meta_data

        transport = TMemoryBuffer()
        protocol = TCompactProtocol(transport)
        new_meta_data.write(protocol)
        meta_data_bytes = transport.getvalue()

        len_meta_data = len(meta_data_bytes)
        meta_data_length_bytes = struct.pack('<I', len_meta_data)
        file_data_bytes = b'PAR1' + file_data_bytes + meta_data_bytes + meta_data_length_bytes + b'PAR1'

        # read the bytes as a complete parquet file
        file_buffer = io.BytesIO(file_data_bytes)
        table = pq.read_table(file_buffer)
        df = table.to_pandas()
        n_records += 1
    return n_records

if __name__ == "__main__":
    file_path = "tweet.parquet"
    read_file_func = gen_local_file_reader(file_path)
    s3_file_reader = S3FileReader("test-cidr", "tweet.parquet")
    read_file_func = s3_file_reader.read_file
    file_index, meta_data = build_local_index(read_file_func)
    sample = sample_data_pages([file_index]*1000, [meta_data]*1000, 80000, [read_file_func]*1000)
    # sample.to_csv("sample.csv")
    # print(f"sample of length {len(sample['Tweet'].unique())}:")
    # print(sample)
