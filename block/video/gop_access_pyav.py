import av
import os
import csv
import random
from tqdm import tqdm
import time

def seek_gops(video_path, sample_keyframe_unique):
    seeking_time = 0
    decoding_time = 0
    sample_frames = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        ctx = av.Codec('h264_cuvid', 'r').create()
        ctx.extradata = stream.codec_context.extradata
        for keyframe_idx, dur in tqdm(sample_keyframe_unique):
            start = time.time()
            container.seek(keyframe_idx, backward=True, any_frame=False)
            seeking_time += time.time() - start
            n_frames = 0
            is_start = True
            start = time.time()
            for pucket in container.demux(stream):
                if is_start:
                    assert pucket.is_keyframe
                    is_start = False
                for frame in ctx.decode(pucket):
                    frame = frame.to_ndarray()
                    sample_frames.append(frame)
                    n_frames += 1
                if n_frames >= dur:
                    break
            decoding_time += time.time() - start
    return seeking_time,decoding_time,sample_frames

def block_seek(block_sample_size: int, video_path: str, block_size: int, gop_size: int=150):
    # fetch the I frames
    print("fetching keyframes")
    video_name = video_path.split("/")[-1].split(".")[0]
    fetch_keyframes = f"ffprobe -v error -select_streams v:0 -skip_frame nokey -show_entries frame=pts -of csv {video_path} > keyframes-{video_name}.csv"
    if not os.path.exists(f"keyframes-{video_name}.csv"):
        os.system(fetch_keyframes)
    keyframe_pts = []
    with open(f"keyframes-{video_name}.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            keyframe_pts.append(int(row[1]))
    n_gops = len(keyframe_pts)
    print(f"total GOPs: {n_gops}, block sample size {block_sample_size}")

    assert gop_size % block_size == 0
    n_unit_per_gop = gop_size // block_size
    n_unit = n_gops * n_unit_per_gop
    print("total units", n_unit)
    sample_unit_idx = random.sample(range(n_unit), block_sample_size)
    sample_keyframe = [[unit_idx // n_unit_per_gop, (unit_idx % n_unit_per_gop + 1) * block_size] for unit_idx in sample_unit_idx]
    sample_keyframe = sorted(sample_keyframe, key=lambda x: x[0])
    sample_keyframe_unique = {}
    for keyframe_idx, dur in sample_keyframe:
        if keyframe_idx not in sample_keyframe_unique:
            sample_keyframe_unique[keyframe_idx] = dur
        else:
            sample_keyframe_unique[keyframe_idx] = max(dur, sample_keyframe_unique[keyframe_idx])
    sample_keyframe_unique = sorted(list(sample_keyframe_unique.items()), key=lambda x: x[0])
    print(f"number of sampled keyframes: {len(sample_keyframe_unique)}")

    seeking_time, decoding_time, sample_frames = seek_gops(video_path, sample_keyframe_unique)
    print(f"sampled frames {len(sample_frames)}")
    print(f"seeking time: {seeking_time}, decoding time: {decoding_time}")

    return seeking_time, decoding_time, sample_frames

def random_seek(sample_size: int, video_path: str, gop_size: int=150):
    # fetch the I frames
    print("fetching keyframes")
    video_name = video_path.split("/")[-1].split(".")[0]
    fetch_keyframes = f"ffprobe -v error -select_streams v:0 -skip_frame nokey -show_entries frame=pts -of csv {video_path} > keyframes-{video_name}.csv"
    if not os.path.exists(f"keyframes-{video_name}.csv"):
        os.system(fetch_keyframes)
    keyframe_pts = []
    with open(f"keyframes-{video_name}.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            keyframe_pts.append(int(row[1]))
    n_gops = len(keyframe_pts)
    print(f"total GOPs: {n_gops}")

    # fetch frame count
    print("fetching frame count")
    frame_count = f"ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 {video_path}"
    n_frame = int(os.popen(frame_count).read())
    print(f"total number of frames: {n_frame}, sample size {sample_size}")

    sample_frame_index = random.sample(range(n_frame), sample_size)
    sample_gop = [[frame // gop_size, frame % gop_size] for frame in sample_frame_index]
    sample_gop = sorted(sample_gop, key=lambda x: x[0])
    sample_gop_unique = {}
    for gop_idx, dur in sample_gop:
        if gop_idx not in sample_gop_unique:
            sample_gop_unique[gop_idx] = dur
        else:
            sample_gop_unique[gop_idx] = max(dur, sample_gop_unique[gop_idx])
    
    sample_gop_unique = sorted(list(sample_gop_unique.items()), key=lambda x: x[0])
    print(f"number of sampled GOPs: {len(sample_gop_unique)}")
    seeking_time, decoding_time, sample_frames = seek_gops(video_path, sample_gop_unique)
    print(f"sampled frames {len(sample_frames)}")
    print(f"seeking time: {seeking_time}, decoding time: {decoding_time}")

    return seeking_time, decoding_time, sample_frames

if __name__ == "__main__":
    block_seek(10998, "data/video.mp4", 50)