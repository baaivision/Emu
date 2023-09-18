from video2dataset.data_writer import (
    FilesSampleWriter,
    WebDatasetSampleWriter,
    ParquetSampleWriter,
    DummySampleWriter,
    TFRecordSampleWriter,
)

import os
import glob
import pytest
import tarfile
import pandas as pd
import pyarrow as pa


@pytest.mark.parametrize("modalities", [["video", "audio"], ["video"], ["audio"]])
@pytest.mark.parametrize("writer_type", ["files", "webdataset", "parquet", "dummy", "tfrecord"])
def test_writer(modalities, writer_type, tmp_path):
    current_folder = os.path.dirname(__file__)
    test_folder = str(tmp_path)
    output_folder = test_folder + "/" + "test_write"
    os.mkdir(output_folder)

    schema = pa.schema(
        [
            pa.field("key", pa.string()),
            pa.field("caption", pa.string()),
            pa.field("status", pa.string()),
            pa.field("error_message", pa.string()),
            pa.field("width", pa.int32()),
            pa.field("height", pa.int32()),
            pa.field("audio_rate", pa.int32()),
        ]
    )
    if writer_type == "files":
        writer_class = FilesSampleWriter
    elif writer_type == "webdataset":
        writer_class = WebDatasetSampleWriter
    elif writer_type == "parquet":
        writer_class = ParquetSampleWriter
    elif writer_type == "dummy":
        writer_class = DummySampleWriter
    elif writer_type == "tfrecord":
        writer_class = TFRecordSampleWriter

    streams = {}
    encode_formats = {}
    for mod in modalities:
        encode_formats[mod] = "mp4" if mod == "video" else "mp3"
        with open(os.path.join(current_folder, f"test_files/test_{mod}.{encode_formats[mod]}"), "rb") as f:
            streams[mod] = f.read()

    n_samples = 1

    writer = writer_class(0, output_folder, True, 5, schema, encode_formats)
    i = 0  # TODO: maybe add more samples
    writer.write(
        streams=streams,
        key=str(i),
        caption=str(i),
        meta={
            "key": str(i),
            "caption": str(i),
            "status": "ok",
            "error_message": "",
            "width": 100,
            "height": 100,
            "audio_rate": 12000,
        },
    )
    writer.close()

    if writer_type != "dummy":

        df = pd.read_parquet(output_folder + "/00000.parquet")

        expected_columns = [
            "key",
            "caption",
            "status",
            "error_message",
            "width",
            "height",
            "audio_rate",
        ]

        if writer_type == "parquet":
            for fmt in encode_formats.values():
                expected_columns.append(fmt)

        assert df.columns.tolist() == expected_columns

        assert df["key"].iloc[0] == "0"
        assert df["caption"].iloc[0] == "0"
        assert df["status"].iloc[0] == "ok"
        assert df["error_message"].iloc[0] == ""
        assert df["width"].iloc[0] == 100
        assert df["height"].iloc[0] == 100
        assert df["audio_rate"].iloc[0] == 12000

    n_files = (len(encode_formats) + len(["caption", "meta"])) * n_samples

    if writer_type == "files":
        saved_files = list(glob.glob(output_folder + "/00000/*"))
        assert len(saved_files) == n_files
    elif writer_type == "webdataset":
        l = glob.glob(output_folder + "/*.tar")
        assert len(l) == 1
        if l[0] != output_folder + "/00000.tar":
            raise Exception(l[0] + " is not 00000.tar")
        assert len(tarfile.open(output_folder + "/00000.tar").getnames()) == n_files
    elif writer_type == "parquet":
        l = glob.glob(output_folder + "/*.parquet")
        assert len(l) == 1
        if l[0] != output_folder + "/00000.parquet":
            raise Exception(l[0] + " is not 00000.parquet")
        assert len(df.index) == n_samples
    elif writer_type == "dummy":
        l = glob.glob(output_folder + "/*")
        assert len(l) == 0
    elif writer_type == "tfrecord":
        l = glob.glob(output_folder + "/*.tfrecord")
        assert len(l) == 1
        if l[0] != output_folder + "/00000.tfrecord":
            raise Exception(l[0] + " is not 00000.tfrecord")
