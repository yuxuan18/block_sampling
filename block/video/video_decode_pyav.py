import av
from tqdm import tqdm

def decode_video(video_path):
    with tqdm(total=2591970) as pbar:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            ctx = av.Codec('h264_cuvid', 'r').create()
            ctx.extradata = stream.codec_context.extradata
            for packet in container.demux(stream):
                for frame in ctx.decode(packet):
                    frame = frame.to_ndarray()
                    pbar.update(1)

if __name__ == "__main__":
    decode_video("data/one-week/video-1.mp4")