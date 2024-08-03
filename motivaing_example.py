import numpy as np
import scipy
from tqdm import tqdm

block_size = 100
data = np.load('data/amazon-labels.npy')
n_blocks = len(data) // block_size + 1
std = np.std(data)
mean = np.mean(data)
sample_size = (scipy.stats.norm.ppf(0.975) * std / mean / 0.05)**2
sample_block_size = (int(sample_size) + 1)  // n_blocks + 1
errors = []
for _ in tqdm(range(1000)):
    sampled_blocks = np.random.choice(n_blocks, sample_block_size, replace=False)
    sampled_data = []
    for sampled_block in sampled_blocks:
        sampled_data += np.array(data[sampled_block * block_size: (sampled_block + 1) * block_size]).tolist()
    error = abs(np.mean(sampled_data) - mean) / mean * 100
    errors.append(error)

print(np.mean(errors), np.percentile(errors, 95), np.percentile(errors, 5))