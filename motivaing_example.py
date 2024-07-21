import random
import numpy as np
from scipy.stats import norm

seed = 2333

random.seed(2333)
np.random.seed(2333)

n_data = 10000
n_block = 100

data = np.random.normal(1, 0.5, n_data)
std = np.std(data)
mean = np.mean(data)
uniform_sample_size = (norm.ppf(0.975) * std / mean / 0.05) ** 2
print(uniform_sample_size)

blocked_data = np.reshape(data, (n_block, -1))
blocked_mean = np.std(blocked_data, axis=1)
block_std = np.std(blocked_mean)
block_mean = np.mean(blocked_mean)
block_sample_size = (norm.ppf(0.975) * block_std / block_mean / 0.05) ** 2
print(block_sample_size)

uniform_sample = random.sample(range(n_data), int(uniform_sample_size)+1)
uniform_sample_dict = {}
for i in uniform_sample:
    if i // 100 not in uniform_sample_dict:
        uniform_sample_dict[i//100] = 1
    else:
        uniform_sample_dict[i//100] += 1
block_sample = random.sample(range(n_block), int(block_sample_size)+1)

print("uniform sample:")
for i in range(100):
    if i in uniform_sample_dict:
        print(i, uniform_sample_dict[i])
print(sorted(block_sample))
