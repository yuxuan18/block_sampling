from scipy.stats import norm, t, chi2
from typing import List
import numpy as np

def mean_lower_bound(sample_mean: float, sample_size: int, sample_std: float, failure_prob: float):
    return sample_mean - t.ppf(1-failure_prob, sample_size - 1) * sample_std / sample_size ** 0.5

def std_upper_bound(sample_std: float, sample_size: int, failure_prob: float):
    return sample_std * ((sample_size - 1) / chi2.ppf(failure_prob/2, sample_size - 1)) ** 0.5

def uniform_sample_size_avg(sample_mean: float, sample_std: float, sample_size: int, failure_prob: float, target_error: float) -> int:
    L_m = mean_lower_bound(sample_mean, sample_size, sample_std, failure_prob/4)
    U_std = std_upper_bound(sample_std, sample_size, failure_prob/4)
    n = (U_std * norm.ppf(1 - failure_prob/2) / (L_m * target_error)) ** 2
    return int(n) + 1

def uniform_sample_size_sum(sample_mean: float, sample_std: float, sample_size: int, failure_prob: float, target_error: float):
    return uniform_sample_size_avg(sample_mean, sample_std, sample_size, failure_prob, target_error)

def uniform_sample_size_count(sample_mean: float, sample_std: float, sample_size: int, failure_prob: float, target_error: float):
    return uniform_sample_size_avg(sample_mean, sample_std, sample_size, failure_prob, target_error)

def block_sample_size_avg(block_sums: List[float], block_sizes: List[int], 
                          failure_prob: float, target_error: float, 
                          equi_block: bool) -> int:
    if equi_block:
        block_avgs = [bs / block_sizes[0] for bs in block_sums]
        sample_mean = np.mean(block_avgs).item()
        sample_std = np.std(block_avgs).item()
        sample_size = len(block_avgs)
        print("sample_mean:", sample_mean)
        print("sample_size:", sample_size)
        print(block_avgs[10], block_sums[10])
        return uniform_sample_size_avg(sample_mean, sample_std, sample_size, failure_prob, target_error)
    else:
        return -1