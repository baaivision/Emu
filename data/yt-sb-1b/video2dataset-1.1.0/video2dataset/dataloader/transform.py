"""
Transformations you might want to apply to data during loading
"""
import math
import cv2
import torch
import numpy as np

from .video_decode import PRNGMixin


class VideoResizer(PRNGMixin):
    """Resizes frames to specified height and width"""

    def __init__(
        self,
        size=None,
        crop_size=None,
        random_crop=False,
        key="mp4",
        width_key="original_width",
        height_key="original_height",
    ):
        self.key = key

        self.height_key = height_key
        self.width_key = width_key

        # resize size as [h,w]
        self.resize_size = size
        # crop size as [h,w]
        self.crop_size = crop_size
        if isinstance(self.crop_size, int):
            self.crop_size = [self.crop_size] * 2
        self.random_crop = random_crop and self.crop_size is not None

        if self.crop_size or self.resize_size:
            print(f"{self.__class__.__name__} is resizing video to size {self.resize_size} ...")

            if self.crop_size:
                print(f'... and {"random" if self.random_crop else "center"} cropping to size {self.crop_size}.')
        else:
            print(f"WARNING: {self.__class__.__name__} is not resizing or croppping videos. Is this intended?")

    def _get_rand_reference(self, resize_size, h, w):
        """gets random reference"""
        assert resize_size is None or (
            self.crop_size[0] <= resize_size[0] and self.crop_size[1] <= resize_size[1]
        ), "Resize size must be greater or equal than crop_size"

        # consistent random crop
        min_x = math.ceil(self.crop_size[1] / 2)
        max_x = w - min_x
        if min_x == max_x:
            # catch corner case
            max_x = min(max_x + 1, w)
        min_y = math.ceil(self.crop_size[0] / 2)
        max_y = h - min_y
        if min_y == max_y:
            # catch corner case
            max_y = min(max_y + 1, h)

        try:
            x = self.prng.randint(min_x, max_x, 1).item()
            y = self.prng.randint(min_y, max_y, 1).item()
        except ValueError as e:
            print("Video size not large enough, consider reducing size")
            print(e)
            raise e

        reference = [y, x]
        return reference

    def _get_resize_size(self, frame, orig_h, orig_w):
        """gets resize size"""
        if self.resize_size is not None:
            if isinstance(self.resize_size, int):
                f = self.resize_size / min((orig_h, orig_w))
                resize_size = [int(round(orig_h * f)), int(round(orig_w * f))]
            else:
                resize_size = self.resize_size
            h, w = resize_size
        else:
            resize_size = None
            h, w = frame.shape[:2]

        return resize_size, (h, w)

    def _get_reference_frame(self, resize_size, h, w):
        """gets reference frame"""
        if self.random_crop:
            reference = self._get_rand_reference(resize_size, h, w)
        else:
            reference = [s // 2 for s in [h, w]]

        return reference

    def __call__(self, data):
        if self.crop_size is None and self.resize_size is None:
            if isinstance(data[self.key], list):
                # convert to tensor
                data[self.key] = torch.from_numpy(np.stack(data[self.key]))
            return data

        result = []

        if self.key not in data:
            raise KeyError(f"Specified key {self.key} not in data")

        vidkey = self.key
        frames = data[vidkey]

        # for videos: take height and width of first frames since the same for all frames anyways,
        # if resize size is integer, then this is used as the new size of the smaller size
        orig_h = data[self.height_key][0].item()
        orig_w = data[self.width_key][0].item()

        resize_size, (h, w) = self._get_resize_size(frames[0], orig_h, orig_w)
        reference = self._get_reference_frame(resize_size, h, w)

        for frame in frames:

            if resize_size is not None:
                frame = cv2.resize(frame, tuple(reversed(resize_size)), interpolation=cv2.INTER_LANCZOS4)

            if self.crop_size is not None:
                x_ = reference[1] - int(round(self.crop_size[1] / 2))
                y_ = reference[0] - int(round(self.crop_size[0] / 2))

                frame = frame[int(y_) : int(y_) + self.crop_size[0], int(x_) : int(x_) + self.crop_size[1]]

            # TODO: maybe lets add other options for normalization
            # will need for VideoCLIP built on top of CLIP
            frame = frame.astype(float) / 127.5 - 1.0

            frame = torch.from_numpy(frame)
            result.append(frame)

        data[vidkey] = torch.stack(result).to(torch.float16)
        return data


class CutsAdder:
    """Adds cuts to video sample"""

    def __init__(self, cuts_key, video_key="mp4"):
        self.cuts_key = cuts_key
        self.video_key = video_key

    def __call__(self, sample):
        assert self.cuts_key in sample, f'no field with key "{self.cuts_key}" in sample, but this is required.'
        assert self.video_key in sample, f'no field with key "{self.video_key}" in sample, but this is required.'
        sample[self.video_key] = {self.video_key: sample[self.video_key], self.cuts_key: sample[self.cuts_key]}
        del sample[self.cuts_key]
        return sample
