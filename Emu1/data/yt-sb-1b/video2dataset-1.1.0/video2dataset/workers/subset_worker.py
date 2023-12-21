"""creates a subset of an existing dataset inside the sample dimension"""
import time
import json
import pyarrow as pa
import traceback

import fsspec

from video2dataset.dataloader import get_video_dataset
from video2dataset.logger import CappedCounter, write_stats


class SubsetWorker:
    """The loader class reads the shards, then the selected data is chosen and writen by the writer"""

    def __init__(
        self,
        sample_writer_class,
        output_folder,
        thread_count,
        number_sample_per_shard,
        oom_shard_count,
        encode_formats,
    ) -> None:
        self.sample_writer_class = sample_writer_class
        self.output_folder = output_folder
        self.number_sample_per_shard = number_sample_per_shard
        self.oom_shard_count = oom_shard_count
        self.thread_count = thread_count
        self.encode_formats = encode_formats
        self.save_caption = True

    def __call__(
        self,
        row,
    ):
        try:
            self.process_shard(row)
            return (True, row)
        except Exception as err:  # pylint: disable=broad-except
            traceback.print_exc()
            print(f"shard {row[0]} failed with error {err}")
            return (False, row)

    def process_shard(
        self,
        row,
    ):
        """Function to start an video processing in one process"""

        shard, shard_id = row
        start_time = time.time()

        fs, shard_path = fsspec.core.url_to_fs(shard[: -len(".tar")] + ".parquet")
        with fs.open(shard_path, "rb") as f:
            df = pa.parquet.read_table(f)
            schema = df.schema

        status_dict = CappedCounter()

        # give schema to writer
        sample_writer = self.sample_writer_class(
            shard_id, self.output_folder, self.save_caption, self.oom_shard_count, schema, self.encode_formats
        )

        successes = 0
        failed_to_subsample = 0
        error_message = None

        dataloader = get_video_dataset(
            urls=[shard],
            batch_size=1,
            decoder_kwargs={},
        )
        for sample in dataloader:
            # Gather subset of dataset
            key = sample["__key__"]
            caption = sample.get("txt", b"").decode("utf-8")
            meta = json.loads(sample.get("json", b"{}").decode("utf-8"))

            streams = {}
            for mod, fmt in self.encode_formats.items():
                streams[mod] = sample[fmt]

            if error_message is not None:
                failed_to_transform += 1
                status = "failed_to_subsample"
                status_dict.increment(error_message)
                meta["status"] = status
                meta["error_message"] = error_message
                sample_writer.write(
                    {},
                    str_key,
                    sample_data[caption_indice] if caption_indice is not None else None,
                    meta,
                )
                continue

            successes += 1
            status = "success"
            status_dict.increment(status)
            meta["status"] = status

            sample_writer.write(
                streams,
                key,
                caption,
                meta,
            )

        sample_writer.close()
        end_time = time.time()

        write_stats(
            self.output_folder,
            shard_id,
            1,  # count
            successes,
            0,  # failed to download
            failed_to_subsample,
            0,  # bytes downloaded
            start_time,
            end_time,
            status_dict,
            self.oom_shard_count,
        )
