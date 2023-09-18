"""
frame subsampler adjusts the fps of the videos to some constant value
"""


import tempfile
import os
import ffmpeg


class AudioRateSubsampler:
    """
    Adjusts the frame rate of the videos to the specified frame rate.
    Args:
        frame_rate (int): Target frame rate of the videos.
    """

    def __init__(self, sample_rate, encode_formats):
        self.sample_rate = sample_rate
        self.encode_formats = encode_formats

    def __call__(self, audio_bytes):
        subsampled_bytes = []
        for aud_bytes in audio_bytes:
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, "input.m4a"), "wb") as f:
                    f.write(aud_bytes)
                ext = self.encode_formats["audio"]
                try:
                    # TODO: for now assuming m4a, change this
                    ffmpeg_args = {"ar": str(self.sample_rate), "f": ext}
                    _ = ffmpeg.input(f"{tmpdir}/input.m4a")
                    _ = _.output(f"{tmpdir}/output.{ext}", **ffmpeg_args).run(capture_stdout=True, quiet=True)
                except Exception as err:  # pylint: disable=broad-except
                    return [], str(err)

                with open(f"{tmpdir}/output.{ext}", "rb") as f:
                    subsampled_bytes.append(f.read())
        return subsampled_bytes, None
