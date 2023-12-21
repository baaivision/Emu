"""test video2dataset downloaders"""
import os
import pytest
import ffmpeg


from video2dataset.data_reader import YtDlpDownloader, WebFileDownloader


YT_URL = "https://www.youtube.com/watch?v=jLX0D8qQUBM"
MP4_URL = "https://ak.picdn.net/shutterstock/videos/1053841541/preview/stock-footage-travel-blogger-shoot-a-story-on-top-of-mountains-young-man-holds-camera-in-forest.mp4"

full_encode_formats = {"video": "mp4", "audio": "m4a"}


@pytest.mark.parametrize("modalities", [["video", "audio"], ["video"], ["audio"]])
@pytest.mark.parametrize("video_size", [361, 1080])
def test_yt_downloader(modalities, video_size):
    encode_formats = dict([(modality, full_encode_formats[modality]) for modality in modalities])

    ytdlp_downloader = YtDlpDownloader(
        tmp_dir="/tmp", metadata_args=None, video_size=video_size, audio_rate=12000, encode_formats=encode_formats
    )

    modality_paths, yt_meta_dict, error_message = ytdlp_downloader(YT_URL)
    assert error_message is None

    for modality, path in modality_paths.items():
        if modality == "video":
            probe = ffmpeg.probe(modality_paths["video"])
            video_stream = [stream for stream in probe["streams"] if stream["codec_type"] == "video"][0]
            height = int(video_stream["height"])

            assert height == 480
        elif modality == "audio":
            with open(path, "rb") as f:
                audio_bytes = f.read()
                assert len(audio_bytes) > 0
        else:
            assert error_messsage == "Error: Untested modality"
        os.remove(path)


def test_webfile_downloader():
    webfile_downloader = WebFileDownloader(timeout=10, tmp_dir="/tmp", encode_formats={"video": "mp4"})

    modality_paths, error_message = webfile_downloader(MP4_URL)

    assert error_message is None
    with open(modality_paths["video"], "rb") as f:
        assert len(f.read()) > 0
    os.remove(modality_paths["video"])
