"""Custom WebDataset classes"""
import numpy as np
import torch
from torch.utils.data import IterableDataset


import webdataset as wds
from webdataset import DataPipeline, filters, shardlists, cache, tariterators
from webdataset.compat import FluidInterface
from webdataset.autodecode import Decoder, ImageHandler


def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [s[key] for s in samples] for key in keys}

    result = {}
    for key, values in batched.items():  # Iterate over both key and values
        first_value = values[0]
        if isinstance(first_value, (int, float)):
            if combine_scalars:
                result[key] = np.array(values)
        elif isinstance(first_value, torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(values)
        elif isinstance(first_value, np.ndarray):
            if combine_tensors:
                result[key] = np.array(values)
        else:
            result[key] = values

    return result


class KeyPassThroughDecoder(Decoder):
    """Decoder which allows you to pass through some keys"""

    def __init__(self, *args, passthrough_keys=None, **kwargs):
        """
        Initialize the KeyPassThroughDecoder.

        :param *args: Positional arguments to be passed to the base Decoder class.
        :param passthrough_keys: List of keys to bypass the decoding process.
        :param **kwargs: Keyword arguments to be passed to the base Decoder class.
        """
        super().__init__(*args, **kwargs)
        self.passthrough_keys = passthrough_keys or []  # Simplified passthrough_keys initialization

    def decode(self, sample):
        """
        Decode an entire sample.

        :param dict sample: The sample, a dictionary of key-value pairs.
        :return: Decoded sample.
        :rtype: dict
        """
        result = {}
        assert isinstance(sample, dict), sample
        for k, v in sample.items():  # Removed unnecessary list conversion
            if k[0] == "_":
                if isinstance(v, bytes):
                    v = v.decode("utf-8")
                result[k] = v
                continue
            if self.only is not None and k not in self.only:
                result[k] = v
                continue
            assert v is not None
            if self.partial:
                if isinstance(v, bytes):
                    result[k] = self.decode1(k, v)
                else:
                    result[k] = v
            else:
                assert (
                    isinstance(v, bytes) or k in self.passthrough_keys
                ), f"key: {k}; passthrough_keys: {self.passthrough_keys}"
                result[k] = self.decode1(k, v)
        return result


class FluidInterfaceWithChangedDecode(FluidInterface):
    def decode(
        self, *args, pre=None, post=None, only=None, partial=False, passthrough_keys=None, handler=wds.reraise_exception
    ):
        handlers = [ImageHandler(x) if isinstance(x, str) else x for x in args]
        decoder = KeyPassThroughDecoder(
            handlers, passthrough_keys=passthrough_keys, pre=pre, post=post, only=only, partial=partial
        )
        return self.map(decoder, handler=handler)


# TODO: pylint says this needs __getitem__
class WebDatasetWithChangedDecoder(DataPipeline, FluidInterfaceWithChangedDecode):
    """Small fluid-interface wrapper for DataPipeline."""

    def __init__(
        self,
        urls,
        handler=wds.reraise_exception,
        resampled=False,
        shardshuffle=None,
        cache_size=0,
        cache_dir=None,
        detshuffle=False,
        nodesplitter=shardlists.single_node_only,
        verbose=False,
    ):
        super().__init__()
        if isinstance(urls, IterableDataset):
            assert not resampled
            self.append(urls)
        elif isinstance(urls, dict):
            assert "datasets" in urls
            self.append(shardlists.MultiShardSample(urls))
        elif resampled:
            self.append(shardlists.ResampledShards(urls))
        else:
            self.append(shardlists.SimpleShardList(urls))
            self.append(nodesplitter)
            self.append(shardlists.split_by_worker)
            if shardshuffle is True:
                shardshuffle = 100
            if shardshuffle is not None:
                if detshuffle:
                    self.append(filters.detshuffle(shardshuffle))
                else:
                    self.append(filters.shuffle(shardshuffle))
        if cache_size == 0:
            self.append(tariterators.tarfile_to_samples(handler=handler))
        else:
            assert cache_size == -1 or cache_size > 0
            self.append(
                cache.cached_tarfile_to_samples(
                    handler=handler,
                    verbose=verbose,
                    cache_size=cache_size,
                    cache_dir=cache_dir,
                )
            )
