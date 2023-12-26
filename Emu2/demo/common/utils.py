# -*- coding: utf-8 -*-

# =================================================== 
#
#    Author        : Fan Zhang
#    Email         : zhangfan@baai.ac.cn
#    Institute     : Beijing Academy of Artificial Intelligence (BAAI)
#    Create On     : 2023-12-13 09:48
#    Last Modified : 2023-12-22 12:09
#    File Name     : utils.py
#    Description   : 
#
# =================================================== 

from datetime import datetime
import logging
import logging.config
import os
import os.path as osp
import uuid
from PIL import Image

from decord import VideoReader

def extract_frames(video, num_frames):
    video = VideoReader(video)
    total_frames = len(video)
    segment = int(total_frames // num_frames)

    frames = video.get_batch(list(range(int(segment//2), total_frames, segment))).asnumpy()
    frames = [Image.fromarray(f) for f in frames]
    return frames


def gen_id():
    logid = datetime.now().strftime("%Y%m%d%H%M%d")
    logid += f"{uuid.uuid4().hex}"
    return logid


LOG_PATH = osp.dirname(__file__)
while not LOG_PATH.endswith("demo"):
    LOG_PATH = osp.dirname(LOG_PATH)
LOG_PATH = osp.join(LOG_PATH, "log")
os.makedirs(LOG_PATH, exist_ok=True)

def config_logger(logger_name):
    logger_config = {
        "version": 1,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(filename)s: %(lineno)d - [%(levelname)s] - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "INFO",
            },
            "file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "filename": osp.join(LOG_PATH, f"{logger_name}.log"),
                "formatter": "standard",
                "level": "INFO",
                "when": "D",
                "interval": 7,
                "backupCount": 90,
            },
        },
        "loggers": {
            logger_name: {
                "handlers": ["file", "console"],
                "level": "INFO",
                "propagate": True,
            },
        },
    }

    logging.config.dictConfig(logger_config)
    logger = logging.getLogger(logger_name)
    return logger

os.makedirs(osp.join(osp.dirname(__file__), "..", "log"), exist_ok=True)
frontend_logger = config_logger("frontend")
backend_logger = config_logger("backend")

