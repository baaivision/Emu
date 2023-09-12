"""
Benchmark cut detector speed
"""
import time
from video2dataset.dataloader import get_bytes_dataloader
from video2dataset.subsamplers import CutDetectionSubsampler
import ffmpeg
import os
import tempfile

# Benchmark videos are the WebVid validation split (5000 videos)
SHARDS = "/fsx/daniel_mend/test_v2ds/webvid_val/dataset/{00000..00004}.tar"


def benchmark_cut_detector(workers, cut_detection_mode, framerates=None):
    subsampler = CutDetectionSubsampler(cut_detection_mode=cut_detection_mode, framerates=framerates)
    dl = get_bytes_dataloader(SHARDS, workers)

    count = 0
    time_taken = 0
    n_frames = 0
    for samp in dl:
        key, vb, cap, meta = samp
        streams = {"video": vb}

        t = time.time()
        cuts = subsampler(streams)
        dt = time.time() - t
        time_taken += dt
        n_frames += cuts["cuts_original_fps"][-1][-1]  # this is the number of frames in the video
        count += 1
    return {"vids_per_second": count / time_taken, "frames_per_second": n_frames / time_taken}


x = benchmark_cut_detector(48, "all")
print(x)

# 3.947849946894834 VIDS/S
# 2054.9439520235096 FRAMES/S
