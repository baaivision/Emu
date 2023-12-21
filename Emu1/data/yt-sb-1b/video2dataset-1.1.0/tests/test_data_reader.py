import os

import pandas as pd
import pytest
import tempfile
import subprocess
from video2dataset.data_reader import VideoDataReader
import ffmpeg


@pytest.mark.parametrize("input_file", ["test_yt.csv"])
def test_data_reader(input_file):
    encode_formats = {"video": "mp4", "audio": "mp3"}
    current_folder = os.path.dirname(__file__)
    url_list = pd.read_csv(os.path.join(current_folder, f"test_files/{input_file}"))["contentUrl"]
    with tempfile.TemporaryDirectory() as tmpdir:
        video_data_reader = VideoDataReader(
            video_size=360,
            audio_rate=12000,
            dl_timeout=60,
            tmp_dir=tmpdir,
            encode_formats=encode_formats,
            yt_meta_args=None,
        )
        for i, url in enumerate(url_list):
            key, streams, yt_meta_dict, error_message = video_data_reader((i, url))

            assert len(streams.get("audio", [])) > 0
            assert len(streams.get("video", [])) > 0
