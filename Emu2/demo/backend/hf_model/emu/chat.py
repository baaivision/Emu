# -*- coding: utf-8 -*-

# ===========================================================================================
#
#    Copyright (c) Beijing Academy of Artificial Intelligence (BAAI). All rights reserved.
#
#    Author        : Fan Zhang
#    Email         : zhangfan@baai.ac.cn
#    Institute     : Beijing Academy of Artificial Intelligence (BAAI)
#    Create On     : 2023-12-12 07:59
#    Last Modified : 2023-12-25 04:33
#    File Name     : chat.py
#    Description   :
#
# ===========================================================================================

from functools import lru_cache
from math import prod
from PIL import Image
from typing import List, Optional

import torch
import torch.nn as nn
from torchvision import transforms as TF
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .constants import EVA_IMAGE_SIZE, OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .constants import DEFAULT_IMG_PLACEHOLDER, DEFAULT_VID_PLACEHOLDER
from .constants import DEFAULT_VIDEO_TOKEN, FAKE_VIDEO_END_TOKEN
from .constants import SYSTEM_MESSAGE, GROUND_SYSTEM_MESSAGE, USER_TOKEN, ASSISTANT_TOKEN, GRD_SYMBOL, DEFAULT_EOS_TOKEN


class EmuChatGeneration():

    def __init__(
        self,
        tokenizer,
        encoder,
        eva_size=EVA_IMAGE_SIZE,
        eva_mean=OPENAI_DATASET_MEAN,
        eva_std=OPENAI_DATASET_STD,
        device_list: List[torch.device] = [torch.device("cpu")],
        torch_dtype: torch.dtype = torch.bfloat16,
        quantize: bool = False,
        **kwargs,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        with init_empty_weights():
            self.emu_model = AutoModelForCausalLM.from_pretrained(
                encoder,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        device_map = self.map_device(device_list)
        self.device = device_list[0]

        if quantize:
            self.emu_model = AutoModelForCausalLM.from_pretrained(
                encoder,
                load_in_4bit=True,
                trust_remote_code=True,
                bnb_4bit_compute_dtype=torch.float16,
                device_map=device_map
            ).eval()
            self.dtype = torch.float16
        else:
            self.emu_model = load_checkpoint_and_dispatch(self.emu_model, encoder, device_map=device_map).eval()
            self.dtype = torch_dtype

        self.transform = TF.Compose([
            TF.Resize((eva_size, eva_size), interpolation=TF.InterpolationMode.BICUBIC),
            TF.ToTensor(),
            TF.Normalize(mean=eva_mean, std=eva_std),
        ])

    @torch.no_grad()
    def __call__(
        self,
        inputs: List[Image.Image | str] | List[List[Image.Image | str]],
        is_grounding: bool = False,
        num_beams: int = 5,
        max_new_tokens: int = 10,
        min_len: int = 1,
        do_sample: bool = False,
        penalty_alpha: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        length_penalty: float = -1,
        repetition_penalty: float = 1.0,
        skip_special_tokens: bool = True,
        **kwargs,
    ):
        """
            For chat generation, inputs must be List[List[str | Image.Image]]
            Otherwise, inputs must be List[str | Image.Image]
        """
        assert isinstance(inputs, list), "inputs must be a list"

        # for chat generation
        if isinstance(inputs[0], list):
            assert len(inputs) % 2 == 1, "last message must be user input"
            (
                text_prompt,
                image_prompt,
                video_prompt,
                image_placeholder,
                video_placeholder,
            ) = self._prepare_chat_inputs(
                inputs,
                is_grounding,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            (
                text_prompt,
                image_prompt,
                video_prompt,
                image_placeholder,
                video_placeholder,
            ) = self._prepare_inputs(
                inputs,
                device=self.device,
                dtype=self.dtype,
            )

        input_ids, attention_mask = self._tokenize(
            text_prompt,
            image_placeholder=image_placeholder,
            video_placeholder=video_placeholder
        )

        outputs = self.emu_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image=image_prompt,
            video=video_prompt,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_len=min_len,
            do_sample=do_sample,
            penalty_alpha=penalty_alpha,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )

        output_text = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=skip_special_tokens,
        )

        return output_text[0]

    def _prepare_inputs(
        self,
        inputs: List[Image.Image | str],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        image_placeholder: str = DEFAULT_IMG_PLACEHOLDER,
        video_placeholder: str = DEFAULT_VID_PLACEHOLDER,
    ):
        is_video = False
        text_prompt, image_prompt, video_prompt = "", [], []
        for x in inputs:
            if x == FAKE_VIDEO_END_TOKEN:
                is_video = False
            elif isinstance(x, str):
                if x == DEFAULT_VIDEO_TOKEN:
                    is_video = True
                text_prompt += x
            elif is_video:
                text_prompt += video_placeholder
                video_prompt.append(self.transform(x))
            else:
                text_prompt += image_placeholder
                image_prompt.append(self.transform(x))

        if len(image_prompt) == 0:
            image_prompt = None
        else:
            image_prompt = torch.stack(image_prompt)
            image_prompt = image_prompt.to(device=device, dtype=dtype)

        if len(video_prompt) == 0:
            video_prompt = None
        else:
            video_prompt = torch.stack(video_prompt)
            video_prompt = video_prompt.to(device=device, dtype=dtype)

        return [text_prompt], image_prompt, video_prompt, image_placeholder, video_placeholder

    def _prepare_chat_inputs(
        self,
        inputs: List[List[Image.Image | str]],
        is_grounding: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        image_placeholder: str = DEFAULT_IMG_PLACEHOLDER,
        video_placeholder: str = DEFAULT_VID_PLACEHOLDER,
    ):
        text_prompt = GROUND_SYSTEM_MESSAGE if is_grounding else SYSTEM_MESSAGE
        image_prompt, video_prompt = None, None

        prev_r = None
        for msg in inputs:
            if prev_r == ASSISTANT_TOKEN:
                text_prompt += f"{DEFAULT_EOS_TOKEN}{USER_TOKEN}: "
                prev_r = USER_TOKEN
            elif prev_r is None:
                text_prompt += f" {USER_TOKEN}: "
                prev_r = USER_TOKEN
            else:
                text_prompt += f" {ASSISTANT_TOKEN}: "
                prev_r = ASSISTANT_TOKEN

            text, image, video, _, _ = self._prepare_inputs(msg, device, dtype, image_placeholder, video_placeholder)

            text_prompt += text[0]
            if image is not None:
                image_prompt = image if image_prompt is None else torch.cat([image_prompt, image])
            if video is not None:
                video_prompt = video if video_prompt is None else torch.cat([video_prompt, video])

        text_prompt += f" {ASSISTANT_TOKEN}:"
        if is_grounding:
            text_prompt += GRD_SYMBOL

        return [text_prompt], image_prompt, video_prompt, image_placeholder, video_placeholder

    def _tokenize(
        self,
        text: List[str],
        image_placeholder: str = DEFAULT_IMG_PLACEHOLDER,
        video_placeholder: str = DEFAULT_VID_PLACEHOLDER,
    ):
        text = [
            t.replace(image_placeholder, self.emu_model.image_placeholder).replace(video_placeholder, self.emu_model.video_placeholder)
            for t in text
        ]
        inputs = self.tokenizer(text, padding="longest", return_tensors="pt")
        return inputs.input_ids, inputs.attention_mask

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        **kwargs,
    ):
        return cls(
            tokenizer=path,
            encoder=path,
            **kwargs,
        )

    def map_device(
        self,
        device_list: List[str | torch.device],
    ):
        """
            A simple multi device strategy, which distribute blocks in large language modles averagely
            into multi devices while keeping rest layers in LLM on the first device
            model.visual:                           4B [cuda:0]
            project_down:                         omit [cuda:0]
            project_up:                           omit [cuda:0]
            model.decoder.lm.model.embed_tokens:  omit [cuda:0]
            model.decoder.lm.model.norm:          omit [cuda:0]
            model.decoder.lm.lm_head:             omit [cuda:0]
            model.decoder.lm.model.layers.[0..59]: 33B (0.55B/layer) [cuda:0 ~ cuda:x]
        """
        device_map = {
            "project_down": device_list[0],
            "project_up": device_list[0],
            "model.visual": device_list[0],
            "model.decoder.lm.model.embed_tokens": device_list[0],
            "model.decoder.lm.model.norm": device_list[0],
            "model.decoder.lm.lm_head": device_list[0],
        }

        other_params = self.params_num(self.emu_model.model.visual) + \
                       self.params_num(self.emu_model.project_down) + \
                       self.params_num(self.emu_model.project_up) + \
                       self.params_num(self.emu_model.model.decoder.lm.model.embed_tokens) + \
                       self.params_num(self.emu_model.model.decoder.lm.model.norm) + \
                       self.params_num(self.emu_model.model.decoder.lm.lm_head)

        layer_params = self.params_num(self.emu_model.model.decoder.lm.model.layers[0])
        layer_num = len(self.emu_model.model.decoder.lm.model.layers)

        total_params = other_params + layer_params * layer_num
        params_per_device = [total_params / len(device_list) for _ in device_list]
        params_per_device[0] -= other_params

        accumulate_params, device_idx = 0, 0
        for idx in range(layer_num):
            if accumulate_params + layer_params > params_per_device[device_idx] and device_idx < len(device_list) - 1:
                accumulate_params = 0
                device_idx += 1

            device_map[f"model.decoder.lm.model.layers.{idx}"] = device_list[device_idx]
            accumulate_params += layer_params

        for l, d in device_map.items():
            print(f"put {l} to device {d}")

        return device_map 

    @lru_cache
    def params_num(self, module: nn.Module):
        return sum([prod(p.shape) for p in module.parameters()])
