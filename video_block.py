from block.video.gop_access_pyav import block_seek
from block.video.compute_sample import block_sample
import random
import sys

def perf_only(n_block: int, video_path: str, replicate: int=7):
    # divide the n_block randomly into replicates
    assignments = random.choices(range(replicate), k=n_block)
    for replicate_id in range(replicate):
        block_seek(len([a for a in assignments if a == replicate_id]), video_path, block_size=15, gop_size=150)


def sample_only(pilot_sample_size: int, failure_rate: float, error_rate: float):
    result = block_sample(pilot_sample_size, failure_rate, error_rate)
    print(result)
    
if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "perf":
        perf_only(102668, "data/one-week/video-1.mp4")
    elif mode == "sample":
        sample_only(2000, 0.05, 0.05)
