import glob
import os

filenames = glob.glob("data/2017-12-14/*")
with open("video_list.txt", "w") as f:
    for filename in filenames:
        f.write(f"file {filename}\n")

os.system("ffmpeg -f concat -safe 0 -i video_list.txt -c copy data/video.mp4")