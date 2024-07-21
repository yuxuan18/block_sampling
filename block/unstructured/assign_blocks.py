from page_access_parquet import build_local_index, gen_local_file_reader
import pandas as pd
from tqdm import tqdm

local_file_reader = gen_local_file_reader('tweet.parquet')
local_index, _ = build_local_index(local_file_reader)
block_ids = []
block_count = 0
for row_group in local_index.row_groups:
    last_row_index = 0
    n_rows = row_group.n_rows
    for data_page_loc in  tqdm(row_group.data_page_locs):
        block_ids += [block_count-1 for _ in range(data_page_loc.first_row_index - last_row_index)]
        last_row_index = data_page_loc.first_row_index
        block_count += 1
    block_ids += [block_count-1 for _ in range(n_rows - last_row_index)]
    print(f"current size {len(block_ids)}")
pd.DataFrame({"block_id": block_ids}).to_csv('block_ids.csv', index=False)