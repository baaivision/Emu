"""test video2dataset subsamplers"""
import os
import subprocess
import pytest
import ffmpeg
import tempfile


from video2dataset.subsamplers import (
    ClippingSubsampler,
    get_seconds,
    ResolutionSubsampler,
    FrameSubsampler,
    AudioRateSubsampler,
    CutDetectionSubsampler,
)


SINGLE = [[50.0, 60.0]]
MULTI = [
    ["00:00:09.000", "00:00:13.500"],
    ["00:00:13.600", "00:00:24.000"],
    ["00:00:45.000", "00:01:01.230"],
    ["00:01:01.330", "00:01:22.000"],
    ["00:01:30.000", "00:02:00.330"],
]


@pytest.mark.parametrize("clips", [SINGLE, MULTI])
def test_clipping_subsampler(clips):
    current_folder = os.path.dirname(__file__)
    # video lenght - 2:02
    video = os.path.join(current_folder, "test_files/test_video.mp4")
    with open(video, "rb") as vid_f:
        video_bytes = vid_f.read()
    audio = os.path.join(current_folder, "test_files/test_audio.mp3")
    with open(audio, "rb") as aud_f:
        audio_bytes = aud_f.read()

    subsampler = ClippingSubsampler(3, {"video": "mp4", "audio": "mp3"})

    metadata = {
        "key": "000",
        "clips": clips,
    }

    streams = {"video": video_bytes, "audio": audio_bytes}
    stream_fragments, meta_fragments, error_message = subsampler(streams, metadata)
    video_fragments = stream_fragments["video"]
    audio_fragments = stream_fragments["audio"]
    assert error_message is None
    assert len(audio_fragments) == len(video_fragments) == len(meta_fragments) == len(clips)

    for vid_frag, meta_frag in zip(video_fragments, meta_fragments):
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(vid_frag)

            key_ind = int(meta_frag["key"].split("_")[-1])
            s, e = meta_frag["clips"][0]

            assert clips[key_ind] == [s, e]  # correct order

            s_s, e_s = get_seconds(s), get_seconds(e)
            probe = ffmpeg.probe(tmp.name)
            video_stream = [stream for stream in probe["streams"] if stream["codec_type"] == "video"][0]
            frag_len = float(video_stream["duration"])

            # currently some segments can be pretty innacurate
            assert abs(frag_len - (e_s - s_s)) < 5.0


@pytest.mark.parametrize("size,resize_mode", [(144, ["scale"]), (1620, ["scale", "crop", "pad"])])
def test_resolution_subsampler(size, resize_mode):
    current_folder = os.path.dirname(__file__)
    # video lenght - 2:02, 1080x1920
    video = os.path.join(current_folder, "test_files/test_video.mp4")
    with open(video, "rb") as vid_f:
        video_bytes = vid_f.read()

    subsampler = ResolutionSubsampler(size, resize_mode)

    subsampled_videos, error_message = subsampler([video_bytes])
    assert error_message is None

    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(subsampled_videos[0])

        probe = ffmpeg.probe(tmp.name)
        video_stream = [stream for stream in probe["streams"] if stream["codec_type"] == "video"][0]
        h_vid, w_vid = video_stream["height"], video_stream["width"]

        assert h_vid == size
        if resize_mode == ["scale"]:
            assert w_vid == 256  # 1920 / (1080/144)
        else:
            assert w_vid == size


@pytest.mark.parametrize("target_frame_rate", [6, 15, 30])
def test_frame_rate_subsampler(target_frame_rate):
    current_folder = os.path.dirname(__file__)
    # video length - 2:02, 1080x1920, 30 fps
    video = os.path.join(current_folder, "test_files/test_video.mp4")
    with open(video, "rb") as vid_f:
        video_bytes = vid_f.read()

    subsampler = FrameSubsampler(target_frame_rate)

    subsampled_videos, error_message = subsampler([video_bytes])
    assert error_message is None

    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(subsampled_videos[0])

        probe = ffmpeg.probe(tmp.name)
        video_stream = [stream for stream in probe["streams"] if stream["codec_type"] == "video"][0]
        frame_rate = int(video_stream["r_frame_rate"].split("/")[0])

        assert frame_rate == target_frame_rate


@pytest.mark.parametrize("sample_rate", [44100, 24000])
def test_audio_rate_subsampler(sample_rate):
    current_folder = os.path.dirname(__file__)
    audio = os.path.join(current_folder, "test_files/test_audio.mp3")
    with open(audio, "rb") as aud_f:
        audio_bytes = aud_f.read()

    subsampler = AudioRateSubsampler(sample_rate, {"audio": "mp3"})

    subsampled_audios, error_message = subsampler([audio_bytes])

    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
        tmp.write(subsampled_audios[0])

        out = subprocess.check_output(f"file {tmp.name}".split()).decode("utf-8")
        assert "Audio file with ID3 version" in out

        result = ffmpeg.probe(tmp.name)
        read_sample_rate = result["streams"][0]["sample_rate"]

        assert int(read_sample_rate) == sample_rate


@pytest.mark.parametrize(
    "cut_detection_mode,framerates", [("longest", []), ("longest", [15]), ("all", []), ("all", [15])]
)
def test_cut_detection_subsampler(cut_detection_mode, framerates):
    current_folder = os.path.dirname(__file__)
    video = os.path.join(current_folder, "test_files/test_video.mp4")
    with open(video, "rb") as vid_f:
        video_bytes = vid_f.read()
    streams = {"video": video_bytes}

    subsampler = CutDetectionSubsampler(cut_detection_mode, framerates)

    cuts = subsampler(streams)

    if cut_detection_mode == "longest":
        assert len(cuts["cuts_original_fps"]) == 1
        assert cuts["cuts_original_fps"] == [[0, 2096]]

        if len(framerates) > 0:
            assert cuts["cuts_15"] == [[0, 2096]]

    if cut_detection_mode == "all":
        assert len(cuts["cuts_original_fps"]) > 1
        assert cuts["cuts_original_fps"][-1] == [3015, 3677]

        if len(framerates) > 0:
            assert cuts["cuts_15"][-1] == [3016, 3677]
