"""the downloader module handles the downloading"""

import math
import time
import pyarrow as pa
import traceback

import fsspec

from multiprocessing.pool import ThreadPool
from threading import Semaphore
from typing import List, Any
import numpy as np

from video2dataset.data_reader import VideoDataReader
from video2dataset.logger import CappedCounter
from video2dataset.logger import write_stats
from video2dataset.subsamplers import (
    ClippingSubsampler,
    CutDetectionSubsampler,
    FrameSubsampler,
    NoOpSubsampler,
    ResolutionSubsampler,
    AudioRateSubsampler,
)


def compute_key(key, shard_id, oom_sample_per_shard, oom_shard_count):
    true_key = (10**oom_sample_per_shard) * shard_id + key
    key_format = oom_sample_per_shard + oom_shard_count
    str_key = "{true_key:0{key_format}d}".format(  # pylint: disable=consider-using-f-string
        key_format=key_format, true_key=true_key
    )
    return str_key


class DownloadWorker:
    """The downloader class gets calls with shards, download them then call the writer to write them down"""

    def __init__(
        self,
        sample_writer_class,
        save_caption,
        output_folder,
        column_list,
        thread_count,
        timeout,
        number_sample_per_shard,
        oom_shard_count,
        video_size,
        resize_mode,
        video_fps,
        audio_rate,
        tmp_dir,
        yt_metadata_args,
        captions_are_subtitles,
        detect_cuts,
        cut_detection_mode,
        cuts_are_clips,
        encode_formats,
        cut_framerates,
        oom_clip_count=5,
    ) -> None:
        self.sample_writer_class = sample_writer_class
        self.save_caption = save_caption
        self.output_folder = output_folder
        self.column_list = column_list
        self.number_sample_per_shard = number_sample_per_shard
        self.oom_shard_count = oom_shard_count
        self.thread_count = thread_count
        self.captions_are_subtitles = captions_are_subtitles

        self.encode_formats = encode_formats

        self.data_reader = VideoDataReader(video_size, audio_rate, timeout, tmp_dir, yt_metadata_args, encode_formats)

        self.clipping_subsampler = ClippingSubsampler(oom_clip_count, encode_formats)
        self.cut_detection_mode = cut_detection_mode
        self.cut_framerates = cut_framerates
        self.detect_cuts = detect_cuts
        if detect_cuts:
            self.cut_detector = CutDetectionSubsampler(cut_detection_mode=cut_detection_mode, framerates=cut_framerates)
        self.cuts_are_clips = cuts_are_clips
        self.noop_subsampler = NoOpSubsampler()

        video_subsamplers: List[Any] = []
        if resize_mode is not None:
            video_subsamplers.append(ResolutionSubsampler(video_size, resize_mode))
        if video_fps > 0:
            video_subsamplers.append(FrameSubsampler(video_fps))

        audio_subsamplers: List[Any] = []
        if audio_rate > 0:
            audio_subsamplers.append(AudioRateSubsampler(audio_rate, encode_formats))

        self.subsamplers = {"video": video_subsamplers, "audio": audio_subsamplers, "sb": []}

    def __call__(
        self,
        row,
    ):
        try:
            self.download_shard(row)
            return (True, row)
        except Exception as err:  # pylint: disable=broad-except
            traceback.print_exc()
            print(f"shard {row[0]} failed with error {err}")
            return (False, row)

    def download_shard(
        self,
        row,
    ):
        """Function to start an video downloading in one process"""

        shard_id, shard_file = row
        start_time = time.time()

        fs, shard_path = fsspec.core.url_to_fs(shard_file)
        with fs.open(shard_path, "rb") as f:
            df = pa.ipc.open_file(f).read_all()
        schema = df.schema
        schema = (
            schema.append(pa.field("key", pa.string()))
            .append(pa.field("status", pa.string()))
            .append(pa.field("error_message", pa.string()))
        )

        pydict = df.select(self.column_list).to_pydict()
        shard_to_dl = list(enumerate(zip(*(pydict[col] for col in self.column_list))))
        del pydict
        del df

        status_dict = CappedCounter()

        count = len(shard_to_dl)
        successes = 0
        failed_to_download = 0
        failed_to_subsample = 0
        bytes_downloaded = 0
        url_indice = self.column_list.index("url")
        caption_indice = self.column_list.index("caption") if "caption" in self.column_list else None
        key_url_list = [(key, x[url_indice]) for key, x in shard_to_dl]

        semaphore = Semaphore(self.thread_count)

        def data_generator():
            for e in key_url_list:
                semaphore.acquire()  # pylint: disable=(consider-using-with)
                yield e

        loader = data_generator()

        # give schema to writer
        sample_writer = self.sample_writer_class(
            shard_id, self.output_folder, self.save_caption, self.oom_shard_count, schema, self.encode_formats
        )
        oom_sample_per_shard = math.ceil(math.log10(self.number_sample_per_shard))

        with ThreadPool(self.thread_count) as thread_pool:
            for key, streams, yt_meta_dict, error_message in thread_pool.imap_unordered(
                self.data_reader,  # pylint: disable=(unnecessary-lambda)
                loader,
            ):
                try:
                    _, sample_data = shard_to_dl[key]
                    str_key = compute_key(key, shard_id, oom_sample_per_shard, self.oom_shard_count)
                    meta = {
                        **{self.column_list[i]: sample_data[i] for i in range(len(self.column_list))},
                        "key": str_key,
                        "status": None,
                        "error_message": error_message,
                        "yt_meta_dict": yt_meta_dict,
                    }

                    if error_message is not None:
                        if "[youtube]" in error_message:  # video-specific error, remove videoID
                            error_message = "ERROR: [youtube]:" + error_message.split(":")[-1]
                        failed_to_download += 1
                        status = "failed_to_download"
                        status_dict.increment(error_message)
                        meta["status"] = status
                        sample_writer.write(
                            {},
                            str_key,
                            sample_data[caption_indice] if caption_indice is not None else None,
                            meta,
                        )
                        semaphore.release()
                        continue

                    for stream in streams.values():
                        bytes_downloaded += len(stream)

                    metas = [meta]

                    if self.captions_are_subtitles:  # create clips
                        subtitles = meta["yt_meta_dict"]["subtitles"]
                        meta["clips"] = [[line_dict["start"], line_dict["end"]] for line_dict in subtitles]
                        meta["lines"] = [" ".join(line_dict["lines"]) for line_dict in subtitles]

                    elif self.detect_cuts:  # apply cut detection to get clips
                        meta["cuts"] = self.cut_detector(streams)

                        if self.cuts_are_clips:
                            cuts = (np.array(meta["cuts"]["cuts_original_fps"]) / meta["cuts"]["original_fps"]).tolist()
                            meta["clips"] = cuts

                    # 1 video -> many videos (either clipping or noop which does identity broadcasting)
                    broadcast_subsampler = (
                        self.clipping_subsampler
                        if ("clips" in self.column_list or self.captions_are_subtitles)
                        else self.noop_subsampler
                    )
                    subsampled_streams, metas, error_message = broadcast_subsampler(streams, meta)

                    for modality in subsampled_streams:
                        for modality_subsampler in self.subsamplers[modality]:
                            subsampled_modality, error_message = modality_subsampler(subsampled_streams[modality])
                            subsampled_streams[modality] = subsampled_modality

                    if error_message is not None:
                        failed_to_subsample += 1
                        status = "failed_to_subsample"
                        status_dict.increment(error_message)
                        meta["status"] = status
                        meta["clips"] = []
                        meta["error_message"] = error_message
                        sample_writer.write(
                            {},
                            str_key,
                            sample_data[caption_indice] if caption_indice is not None else None,
                            meta,
                        )
                        semaphore.release()
                        continue

                    successes += 1
                    status = "success"
                    status_dict.increment(status)
                    subsampled_streams_list = [
                        dict(zip(subsampled_streams, s)) for s in zip(*subsampled_streams.values())
                    ]
                    for subsampled_streams, meta in zip(subsampled_streams_list, metas):
                        meta["status"] = status

                        text_caption = (sample_data[caption_indice] if caption_indice is not None else None,)
                        if self.captions_are_subtitles:
                            text_caption = meta["yt_meta_dict"].pop("subtitles")

                        sample_writer.write(
                            subsampled_streams,
                            meta["key"],
                            text_caption,
                            meta,
                        )
                except Exception as err:  # pylint: disable=broad-except
                    traceback.print_exc()
                    print(f"Sample {key} failed to download: {err}")
                semaphore.release()

            sample_writer.close()
            thread_pool.terminate()
            thread_pool.join()
            del thread_pool

        end_time = time.time()
        write_stats(
            self.output_folder,
            shard_id,
            count,
            successes,
            failed_to_download,
            failed_to_subsample,
            bytes_downloaded,
            start_time,
            end_time,
            status_dict,
            self.oom_shard_count,
        )
        fs.rm(shard_path)
