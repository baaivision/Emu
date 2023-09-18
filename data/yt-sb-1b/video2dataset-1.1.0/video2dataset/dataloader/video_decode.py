"""Video Decoders"""
import os
import re
import numpy as np
import torch
import decord
import tempfile
from typing import Iterable

from webdataset.autodecode import decoders


decord.bridge.set_bridge("torch")


class PRNGMixin:
    """
    Adds a prng property which is a numpy RandomState which gets
    reinitialized whenever the pid changes to avoid synchronized sampling
    behavior when used in conjunction with multiprocessing.
    """

    @property
    def prng(self):
        currentpid = os.getpid()
        if getattr(self, "_initpid", None) != currentpid:
            self._initpid = currentpid
            self._prng = np.random.RandomState()
        return self._prng


class AbstractVideoDecoder(PRNGMixin):
    def get_frames(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} is abstract")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} is abstract")


class VideoDecorder(AbstractVideoDecoder):
    """Basic video decoder that uses decord"""

    def __init__(self, n_frames=None, fps=None, num_threads=4, tmpdir="/scratch/", min_fps=1, max_fps=32):
        super().__init__()
        self.n_frames = n_frames
        if fps is not None and not isinstance(fps, Iterable):
            fps = [
                fps,
            ]
        self.fps = fps
        self.min_fps = min_fps
        self.max_fps = max_fps
        # for frame rate conditioning
        if self.fps == "sample":
            # this means we sample fps in every iteration
            self.fs_ids = {fr: i for i, fr in enumerate(range(self.min_fps, self.max_fps + 1))}
        elif isinstance(self.fps, list):
            self.fs_ids = {fr: i for i, fr in enumerate(self.fps)}
        else:
            self.fs_ids = None
            assert self.fps is None, f"invalid fps value specified: {self.fps}...."

        self.num_threads = num_threads

        infostring2 = ""
        if self.fps == "sample":
            infostring1 = "a randomly sampled"
            infostring2 = f" in between [{self.min_fps},{self.max_fps}]"
        elif isinstance(self.fps, list):
            infostring1 = ",".join([str(f) for f in self.fps])
        else:
            infostring1 = "native"
        info = (
            f'Decoding video clips of length {self.n_frames} with "decord".'
            + f" Subsampling clips to {infostring1} fps {infostring2}"
        )

        print(info)

        self.tmpdir = tmpdir
        print(f"Setting {self.tmpdir} as temporary directory for the decoder")

    def get_frames(self, reader, n_frames, stride, **kwargs):  # pylint: disable=arguments-differ
        if n_frames * stride > len(reader) - 1:
            raise ValueError("video clip not long enough for decoding")

        # sample frame start and choose scene
        frame_start = self.prng.choice(int(len(reader)) - int(n_frames * stride), 1).item()
        # only decode the frames which are actually needed
        frames = reader.get_batch(np.arange(frame_start, frame_start + n_frames * stride, stride).tolist())

        return frames, frame_start

    def __call__(self, key, data, scene_list=None):  # pylint: disable=arguments-differ
        extension = re.sub(r".*[.]", "", key)
        if extension not in "mp4 ogv mjpeg avi mov h264 mpg webm wmv".split():
            return None

        additional_info = {}
        with tempfile.TemporaryDirectory(dir=self.tmpdir) as dirname:
            fname = os.path.join(dirname, f"file.{extension}")
            with open(fname, "wb") as stream:
                stream.write(data)
            reader = decord.VideoReader(fname, num_threads=self.num_threads)

        if self.n_frames is None:
            n_frames = len(reader)
        else:
            n_frames = self.n_frames

        native_fps = int(np.round(reader.get_avg_fps()))
        if isinstance(self.fps, list):
            fps_choices = list(filter(lambda x: x <= native_fps, self.fps))
            if not fps_choices:
                return None
            chosen_fps = self.prng.choice(list(fps_choices), 1).item()

        elif self.fps == "sample":
            if native_fps < self.min_fps:
                return None
            max_fps = min(native_fps, self.max_fps)
            additional_info.update({"fps_id": 0})
            chosen_fps = self.prng.choice(np.arange(self.min_fps, max_fps + 1), 1).item()
        else:
            chosen_fps = native_fps

        fs_id = self.fs_ids[chosen_fps] if self.fs_ids else 0
        stride = int(np.round(native_fps / chosen_fps))
        additional_info.update({"fps_id": torch.Tensor([fs_id] * n_frames).long()})

        frames, start_frame = self.get_frames(reader, n_frames, stride, scene_list=scene_list)
        frames = frames.float().numpy()

        additional_info["original_height"] = torch.full((frames.shape[0],), fill_value=frames.shape[1]).long()
        additional_info["original_width"] = torch.full((frames.shape[0],), fill_value=frames.shape[2]).long()

        if self.n_frames is not None and frames.shape[0] < self.n_frames:
            raise ValueError("Decoded video not long enough, skipping")

        # return compatible with torchvisioin API
        additional_info.update({"native_fps": chosen_fps if chosen_fps is not None else native_fps})
        additional_info.update({"start_frame": start_frame})
        out = (list(frames), additional_info)
        return out


class VideoDecorderWithCutDetection(VideoDecorder):
    """Video decoder that uses decord with cut detection"""

    def __init__(self, *args, cuts_key="npy", **kwargs):
        super().__init__(*args, **kwargs)
        self.cuts_key = cuts_key
        if self.cuts_key not in decoders:
            raise KeyError(
                f"{self.__class__.__name__} received {self.cuts_key} as cuts_key,"
                + " but that one is no decoder known to webdataset"
            )

    def __call__(self, key, data):  # pylint: disable=arguments-differ,signature-differs
        extension = re.sub(r".*[.]", "", key)
        if extension not in "mp4 ogv mjpeg avi mov h264 mpg webm wmv".split():
            return None

        cut_list = decoders[self.cuts_key](data[self.cuts_key])
        data = data[extension]

        return super().__call__(key, data, scene_list=cut_list)

    def get_frames(self, reader, n_frames, stride, scene_list):  # pylint: disable=arguments-differ

        min_len = n_frames * stride
        # filter out subclips shorther than minimal required length
        scene_list = list(filter(lambda x: x[1] - x[0] > min_len, scene_list))
        if len(scene_list) == 0:
            raise ValueError("video clips not long enough for decoding")

        clip_id = self.prng.choice(len(scene_list), 1).item()
        start, end = scene_list[clip_id].tolist()
        # sample frame start and choose scene
        frame_start = self.prng.choice(int(end - start) - int(n_frames * stride), 1).item() + start
        # only decode the frames which are actually needed
        frames = reader.get_batch(np.arange(frame_start, frame_start + n_frames * stride, stride).tolist())

        return frames, frame_start
