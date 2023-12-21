"""
frame subsampler adjusts the fps of the videos to some constant value
"""


import tempfile
import os
import ffmpeg


class FrameSubsampler:
    """
    Adjusts the frame rate of the videos to the specified frame rate.
    Args:
        frame_rate (int): Target frame rate of the videos.
    """

    def __init__(self, frame_rate):
        self.frame_rate = frame_rate

    def __call__(self, video_bytes):
        subsampled_bytes = []
        for vid_bytes in video_bytes:
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, "input.mp4"), "wb") as f:
                    f.write(vid_bytes)
                try:
                    _ = ffmpeg.input(f"{tmpdir}/input.mp4")
                    _ = _.filter("fps", fps=self.frame_rate)
                    _ = _.output(f"{tmpdir}/output.mp4", reset_timestamps=1).run(capture_stdout=True, quiet=True)
                except Exception as err:  # pylint: disable=broad-except
                    return [], str(err)

                with open(f"{tmpdir}/output.mp4", "rb") as f:
                    subsampled_bytes.append(f.read())
        return subsampled_bytes, None
