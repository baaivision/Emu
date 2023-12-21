import os

import pandas as pd
import pytest
from video2dataset.data_reader import get_yt_meta


@pytest.mark.parametrize("input_file", ["test_yt.csv"])
def test_meta(input_file):
    yt_metadata_args = {
        "writesubtitles": False,
        "get_info": True,
    }
    current_folder = os.path.dirname(__file__)
    url_list = pd.read_csv(os.path.join(current_folder, f"test_files/{input_file}"))["contentUrl"]
    for url in url_list:
        yt_meta_dict = get_yt_meta(url, yt_metadata_args)

        assert type(yt_meta_dict) == dict
        assert type(yt_meta_dict["info"]) == dict
        assert "id" in yt_meta_dict["info"].keys()
        assert "title" in yt_meta_dict["info"].keys()


@pytest.mark.parametrize("input_file", ["test_yt.csv"])
def test_no_meta(input_file):
    yt_metadata_args = {
        "writesubtitles": False,
        "get_info": False,
    }
    current_folder = os.path.dirname(__file__)
    url_list = pd.read_csv(os.path.join(current_folder, f"test_files/{input_file}"))["contentUrl"]
    for url in url_list:
        yt_meta_dict = get_yt_meta(url, yt_metadata_args)

        assert type(yt_meta_dict) == dict
        assert yt_meta_dict["info"] == None


@pytest.mark.parametrize("input_file", ["test_yt.csv"])
def test_subtitles(input_file):
    yt_metadata_args = {
        "writesubtitles": True,
        "subtitleslangs": ["en"],
        "writeautomaticsub": True,
        "get_info": False,
    }
    current_folder = os.path.dirname(__file__)
    url_list = pd.read_csv(os.path.join(current_folder, f"test_files/{input_file}"))["contentUrl"]
    for url in url_list:
        yt_meta_dict = get_yt_meta(url, yt_metadata_args)

        assert type(yt_meta_dict) == dict
        assert type(yt_meta_dict["subtitles"]) == list
        assert type(yt_meta_dict["subtitles"][0]) == dict


@pytest.mark.parametrize("input_file", ["test_yt.csv"])
def test_no_subtitles(input_file):
    yt_metadata_args = {
        "writesubtitles": False,
        "subtitleslangs": ["en"],
        "writeautomaticsub": True,
        "get_info": False,
    }
    current_folder = os.path.dirname(__file__)
    url_list = pd.read_csv(os.path.join(current_folder, f"test_files/{input_file}"))["contentUrl"]
    for url in url_list:
        yt_meta_dict = get_yt_meta(url, yt_metadata_args)

        assert type(yt_meta_dict) == dict
        assert yt_meta_dict["subtitles"] == None
