# -*- coding: utf-8 -*-

# ===========================================================================================
#
#    Copyright (c) Beijing Academy of Artificial Intelligence (BAAI). All rights reserved.
#
#    Author        : Fan Zhang
#    Email         : zhangfan@baai.ac.cn
#    Institute     : Beijing Academy of Artificial Intelligence (BAAI)
#    Create On     : 2023-12-12 02:54
#    Last Modified : 2023-12-22 09:55
#    File Name     : meta.py
#    Description   :
#
# ===========================================================================================

import base64
from dataclasses import dataclass
import io
from enum import Enum
from PIL import Image
from typing import List, Tuple

import cv2
import numpy as np

from .emu_constants import EVA_IMAGE_SIZE, GRD_SYMBOL, BOP_SYMBOL, EOP_SYMBOL, BOO_SYMBOL, EOO_SYMBOL
from .emu_constants import DEFAULT_VIDEO_TOKEN, DEFAULT_EOS_TOKEN, USER_TOKEN, ASSISTANT_TOKEN, FAKE_VIDEO_END_TOKEN

from .utils import gen_id, frontend_logger as logging


class Role(Enum):
    UNKNOWN = 0,
    USER = 1,
    ASSISTANT = 2,


class DataType(Enum):
    UNKNOWN = 0,
    TEXT = 1,
    IMAGE = 2,
    GROUNDING = 3,
    VIDEO = 4,
    ERROR = 5,


@dataclass
class DataMeta:
    datatype: DataType = DataType.UNKNOWN
    text: str = None
    image: Image.Image = None
    mask: Image.Image = None
    coordinate: List[int] = None
    frames: List[Image.Image] = None
    stack_frame: Image.Image = None

    @property
    def grounding(self):
        return self.coordinate is not None

    @property
    def text_str(self):
        return self.text

    @property
    def image_str(self):
        return self.image2str(self.image)

    @property
    def video_str(self):
        ret = f'<div style="overflow:scroll"><b>[VIDEO]</b></div>{self.image2str(self.stack_frame)}'
        return ret

    @property
    def grounding_str(self):
        ret = ""
        if self.text is not None:
            ret += f'<div style="overflow:scroll"><b>[PHRASE]</b>{self.text}</div>'

        ret += self.image2str(self.mask)

        if self.image is not None:
            ret += self.image2str(self.image)
        return ret

    def image2str(self, image):
        buf = io.BytesIO()
        image.save(buf, format="WEBP")
        i_str = base64.b64encode(buf.getvalue()).decode()
        return f'<div style="float:left"><img src="data:image/png;base64, {i_str}"></div>'

    def format_chatbot(self):
        match self.datatype:
            case DataType.TEXT | DataType.ERROR:
                return self.text_str
            case DataType.IMAGE:
                return self.image_str
            case DataType.VIDEO:
                return self.video_str
            case DataType.GROUNDING:
                return self.grounding_str
            case _:
                return ""

    def format_prompt(self) -> List[str | Image.Image]:
        match self.datatype:
            case DataType.TEXT:
                return [self.text]
            case DataType.IMAGE:
                return [self.image]
            case DataType.VIDEO:
                return [DEFAULT_VIDEO_TOKEN] + self.frames + [FAKE_VIDEO_END_TOKEN]
            case DataType.GROUNDING:
                ret = []
                if self.text is not None:
                    ret.append(f"{BOP_SYMBOL}{self.text}{EOP_SYMBOL}")
                ret += [BOO_SYMBOL, self.mask, EOO_SYMBOL]
                if self.image is not None:
                    ret.append(self.image)
                return ret
            case _:
                return []

    def __str__(self):
        s = ""
        if self.text is not None:
            s += f"T:{self.text}"

        if self.image is not None:
            w, h = self.image.size
            s += f"[I:{h}x{w}]"

        if self.coordinate is not None:
            l, t, r, b = self.coordinate
            s += f"[C:({l:03d},{t:03d}),({r:03d},{b:03d})]"

        if self.frames is not None:
            w, h = self.frames[0].size
            s += f"[V:{len(self.frames)}x{h}x{w}]"

        return s

    @classmethod
    def build(cls, text=None, image=None, coordinate=None, frames=None, is_error=False, *, resize: bool = True):
        ins = cls()
        ins.text = text if text != "" else None
        ins.image = cls.resize(image, force=resize)
        # ins.image = image
        ins.coordinate = cls.fix(coordinate)
        ins.frames = cls.resize(frames, force=resize)
        # ins.frames = frames

        if is_error:
            ins.datatype = DataType.ERROR
        elif coordinate is not None:
            ins.datatype = DataType.GROUNDING
            ins.draw_box()
        elif image is not None:
            ins.datatype = DataType.IMAGE
        elif text is not None:
            ins.datatype = DataType.TEXT
        else:
            ins.datatype = DataType.VIDEO
            ins.stack()

        return ins

    @classmethod
    def fix(cls, coordinate):
        if coordinate is None:
            return None

        l, t, r, b = coordinate
        l = min(EVA_IMAGE_SIZE, max(0, l))
        t = min(EVA_IMAGE_SIZE, max(0, t))
        r = min(EVA_IMAGE_SIZE, max(0, r))
        b = min(EVA_IMAGE_SIZE, max(0, b))
        return min(l, r), min(t, b), max(l, r), max(t, b)

    @classmethod
    def resize(cls, image: Image.Image | List[Image.Image] | None, *, force: bool = True):
        if image is None:
            return None

        if not force:
            return image

        if isinstance(image, Image.Image):
            image = [image]

        for idx, im in enumerate(image):
            w, h = im.size
            if w < EVA_IMAGE_SIZE or h < EVA_IMAGE_SIZE:
                continue

            if w < h:
                h = int(EVA_IMAGE_SIZE / w * h)
                w = EVA_IMAGE_SIZE
            else:
                w = int(EVA_IMAGE_SIZE / h * w)
                h = EVA_IMAGE_SIZE

            image[idx] = im.resize((w, h))

        return image if len(image) > 1 else image[0]

    def draw_box(self):
        left, top, right, bottom = self.coordinate
        mask = np.zeros((EVA_IMAGE_SIZE, EVA_IMAGE_SIZE, 3), dtype=np.uint8)
        mask = cv2.rectangle(mask, (left, top), (right, bottom), (255, 255, 255), 3)
        self.mask = Image.fromarray(mask)

    def stack(self):
        w, h = self.frames[0].size
        n = len(self.frames)
        stack_frame = Image.new(mode="RGB", size=(w*n, h))
        for idx, f in enumerate(self.frames):
            stack_frame.paste(f, (idx*w, 0))
        self.stack_frame = stack_frame


class ConvMeta:

    def __init__(self):
        self.system: str = "You are a helpful assistant, dedicated to delivering comprehensive and meticulous responses."
        self.message: List[Tuple[Role, DataMeta]] = []
        self.log_id: str = gen_id()

        logging.info(f"{self.log_id}: create new round of chat")

    def append(self, r: Role, p: DataMeta):
        logging.info(f"{self.log_id}: APPEND [{r.name}] prompt element, type: {p.datatype.name}, message: {p}")
        self.message.append((r, p))

    def format_chatbot(self):
        ret = []
        for r, p in self.message:
            cur_p = p.format_chatbot()
            if r == Role.USER:
                ret.append((cur_p, None))
            else:
                ret.append((None, cur_p))
        return ret

    def format_prompt(self):
        ret = []
        has_coor = False
        for _, p in self.message:
            has_coor |= (p.datatype == DataType.GROUNDING)
            ret += p.format_prompt()

        if has_coor:
            ret.insert(0, GRD_SYMBOL)

        logging.info(f"{self.log_id}: format generation prompt: {ret}")
        return ret

    def format_chat(self):
        ret = [self.system]

        prev_r = None
        for r, p in self.message:
            if prev_r != r:
                if prev_r == Role.ASSISTANT:
                    ret.append(f"{DEFAULT_EOS_TOKEN}{USER_TOKEN}: ")
                elif prev_r is None:
                    ret.append(f" {USER_TOKEN}: ")
                else:
                    ret.append(f" {ASSISTANT_TOKEN}: ")
                ret += p.format_prompt()
                prev_r = r
            else:
                ret += p.format_prompt()

        ret.append(f" {ASSISTANT_TOKEN}:")

        logging.info(f"{self.log_id}: format chat prompt: {ret}")
        return ret

    def clear(self):
        logging.info(f"{self.log_id}: clear chat history, end current chat round.")
        del self.message
        self.message = []
        self.log_id = gen_id()

    def pop(self):
        if self.has_gen:
            logging.info(f"{self.log_id}: pop out previous generation / chat result")
            self.message.pop()

    def pop_error(self):
        new_message = []
        for r, p in self.message:
            if p.datatype == DataType.ERROR:
                logging.info(f"{self.log_id}: pop error message: {p.text_str}")
            else:
                new_message.append((r, p))
        del self.message
        self.message = new_message

    @property
    def has_gen(self):
        if len(self.message) == 0:
            return False
        if self.message[-1][0] == Role.USER:
            return False
        return True

    def __len__(self):
        return len(self.message)
