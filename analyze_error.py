import sys
import json
import numpy as np

filename = sys.argv[1]
with open(filename) as f:
    errors = json.load(f)

for error in errors:
    results = errors[error]
    print(f"Error rate: {error}, min: {np.min(results):.3f}, avg: {np.mean(results):.3f}, 95%: {np.percentile(results, 95):.3f}, max: {np.max(results):.3f}")


