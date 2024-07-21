from scipy.stats import norm, t, chi2
from typing import List
import numpy as np
import math

def mean_lower_bound(sample_mean: float, sample_size: int, sample_std: float, failure_prob: float):
    return sample_mean - t.ppf(1-failure_prob, sample_size - 1) * sample_std / sample_size ** 0.5

def std_upper_bound(sample_std: float, sample_size: int, failure_prob: float):
    return sample_std * ((sample_size - 1) / chi2.ppf(failure_prob, sample_size - 1)) ** 0.5

def uniform_sample_size_avg(sample_mean: float, sample_std: float, sample_size: int, failure_prob: float, target_error: float) -> int:
    L_m = mean_lower_bound(sample_mean, sample_size, sample_std, failure_prob/4)
    U_std = std_upper_bound(sample_std, sample_size, failure_prob/4)
    n = (U_std * norm.ppf(1 - failure_prob/2) / (L_m * target_error)) ** 2
    return int(n) + 1

def hypergeometric_successes_lower(population_size: int, sample_size: int, sample_success: int, failure_probability: float):
    z = norm.ppf(1 - failure_probability)
    a = (sample_size ** 2 / population_size ** 2) + \
        z**2 * (population_size - sample_size) * sample_size / population_size**2 / (population_size - 1)
    b = -(
        2 * sample_size * sample_success / population_size + z**2 * (population_size - sample_size) * sample_size / population_size / (population_size - 1)
    )
    c = sample_success**2
    return int( (-b - math.sqrt(b**2 - 4 * a * c)) / 2 / a ) - 1

def hypergeometric_n_lower(population_size: int, success: int, sample_success: int, failure_probability: float):
    z = norm.ppf(1 - failure_probability)
    a = sample_success
    b = z * math.sqrt(success * (population_size - success) /  (population_size - 1) )
    c = sample_success - success
    t = (-b + math.sqrt(b**2 - 4 * a * c)) / 2 / a
    return int( population_size / (1 + t**2) ) + 1

def uniform_sample_size_sum(sample_mean: float, sample_std: float, sample_size: int, failure_prob: float, target_error: float):
    return uniform_sample_size_avg(sample_mean, sample_std, sample_size, failure_prob, target_error)

def uniform_sample_size_count(sample_mean: float, sample_std: float, sample_size: int, failure_prob: float, target_error: float):
    return uniform_sample_size_avg(sample_mean, sample_std, sample_size, failure_prob, target_error)

def block_sample_size_avg_predicate(population_size: int, sample_size: int, sample_success: int, 
                          success_block_counts: List, success_block_sums: List,
                          failure_prob: float, target_error: float) -> int:
    block_count_sample_size = uniform_sample_size_avg(
        np.mean(success_block_counts).item(), np.std(success_block_counts).item(), 
        sample_success, failure_prob/4, target_error
    )
    block_sum_sample_size = uniform_sample_size_avg(
        np.mean(success_block_sums).item(), np.std(success_block_sums).item(), 
        sample_success, failure_prob/4, target_error
    )

    target_sample_success = max(block_count_sample_size, block_sum_sample_size)

    if sample_success == sample_size:
        return target_sample_success

    success_lower = hypergeometric_successes_lower(population_size, sample_size, sample_success, failure_prob/4)

    required_sample_size = hypergeometric_n_lower(population_size, success_lower, target_sample_success, failure_prob/4)

    return required_sample_size

def uniform_sample_size_avg_predicate(population_size: int, sample_size: int, sample_success: int,
                                      success_results: List, failure_prob: float, target_error: float):
    target_sample_success = uniform_sample_size_avg(
        np.mean(success_results).item(), np.std(success_results).item(),
        sample_success, failure_prob/4, target_error
    )

    success_lower = hypergeometric_successes_lower(population_size, sample_size, sample_success, failure_prob/4)

    required_sample_size = hypergeometric_n_lower(population_size, success_lower, target_sample_success, failure_prob/4)

    return required_sample_size