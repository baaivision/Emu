"""
resolution subsampler adjusts the resolution of the videos to some constant value
"""
import os
import ffmpeg
import tempfile


class ResolutionSubsampler:
    """
    Adjusts the resolution of the videos to the specified height and width.

    Args:
        video_size (int): Target resolution of the videos.
        resize_mode (list[str]): List of resize modes to apply. Possible options are:
            scale: scale video keeping aspect ratios (currently always picks video height)
            crop: center crop to video_size x video_size
            pad: center pad to video_size x video_size
    """

    def __init__(self, video_size, resize_mode):
        self.video_size = video_size
        self.resize_mode = resize_mode

    def __call__(self, video_bytes):
        subsampled_bytes = []
        for vid_bytes in video_bytes:
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, "input.mp4"), "wb") as f:
                    f.write(vid_bytes)
                try:
                    _ = ffmpeg.input(f"{tmpdir}/input.mp4")
                    if "scale" in self.resize_mode:
                        _ = _.filter("scale", -2, self.video_size)
                    if "crop" in self.resize_mode:
                        _ = _.filter("crop", w=self.video_size, h=self.video_size)
                    if "pad" in self.resize_mode:
                        _ = _.filter("pad", w=self.video_size, h=self.video_size)
                    _ = _.output(f"{tmpdir}/output.mp4", reset_timestamps=1).run(capture_stdout=True, quiet=True)
                except Exception as err:  # pylint: disable=broad-except
                    return [], str(err)

                with open(f"{tmpdir}/output.mp4", "rb") as f:
                    subsampled_bytes.append(f.read())
        return subsampled_bytes, None
