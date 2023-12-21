# -*- coding: utf-8 -*-

from PIL import Image
from typing import List, Optional

import torch
import torch.nn as nn
from torchvision import transforms as TF

from .emu import EmuModel
from .conf.emu_conf import CLIPVisionCfg, TextDecoderCfg
from .constants import EVA_IMAGE_SIZE, OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .constants import DEFAULT_IMG_PLACEHOLDER, DEFAULT_VID_PLACEHOLDER
from .constants import DEFAULT_VIDEO_TOKEN, FAKE_VIDEO_END_TOKEN
from .mixin import ModelParallelMixin


class EmuChatGeneration(nn.Module, ModelParallelMixin):

    def __init__(
        self,
        encoder,
        eva_size=EVA_IMAGE_SIZE,
        eva_mean=OPENAI_DATASET_MEAN,
        eva_std=OPENAI_DATASET_STD,
        **kwargs,
    ):
        super().__init__()

        vision_cfg = CLIPVisionCfg(n_query=256, v_query=64)
        text_decoder_cfg = TextDecoderCfg(instruct=True)
        self.emu_model = EmuModel(vision_cfg=vision_cfg, text_decoder_cfg=text_decoder_cfg)
        print(self.emu_model.load_state_dict(encoder, strict=False))

        self.transform = TF.Compose([
            TF.Resize((eva_size, eva_size), interpolation=TF.InterpolationMode.BICUBIC),
            TF.ToTensor(),
            TF.Normalize(mean=eva_mean, std=eva_std),
        ])

    @torch.no_grad()
    def forward(
        self,
        inputs: List[Image.Image | str] | str | Image.Image,
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
        synced_gpus: bool = False,
        skip_special_tokens: bool = True,
        **kwargs,
    ):
        if not isinstance(inputs, list):
            inputs = [inputs]

        device = self.emu_model.device
        dtype = self.emu_model.dtype

        (
            text_prompt,
            image_prompt,
            video_prompt,
            image_placeholder,
            video_placeholder,
        ) = self._prepare_inputs(
            inputs,
            device,
            dtype,
        )

        output = self.emu_model.generate(
            text=text_prompt,
            image=image_prompt,
            video=video_prompt,
            image_placeholder=image_placeholder,
            video_placeholder=video_placeholder,
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
            synced_gpus=synced_gpus,
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )

        return output[0]

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
            image_prompt = image_prompt.type(dtype).to(device)

        if len(video_prompt) == 0:
            video_prompt = None
        else:
            video_prompt = torch.stack(video_prompt)
            video_prompt = video_prompt.type(dtype).to(device)

        return [text_prompt], image_prompt, video_prompt, image_placeholder, video_placeholder

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        **kwargs,
    ):
        encoder = kwargs.pop("encoder", None)
        encoder = encoder if encoder is not None else f"{path}/pytorch_model.bin"

        return cls(
            encoder=encoder,
            **kwargs,
        )

    def multicuda(
        self,
        device_list: List[str | torch.device],
    ):
        """
            emu_model.visual:                           4B
            emu_model.decoder.lm.project_down:        omit
            emu_model.decoder.lm.project_up:          omit
            emu_model.decoder.lm.model.embed_tokens:  omit
            emu_model.decoder.lm.model.norm:          omit
            emu_model.decoder.lm.lm_head:             omit
            emu_model.decoder.lm.model.layers.[0..59]: 33B (0.55B/layer)
        """
        mp_rule = {
            "emu_model.visual": device_list[0],
            "emu_model.decoder.lm.project_down": device_list[0],
            "emu_model.decoder.lm.project_up": device_list[0],
            "emu_model.decoder.lm.model.embed_tokens": device_list[0],
            "emu_model.decoder.lm.model.norm": device_list[0],
            "emu_model.decoder.lm.lm_head": device_list[0],
        }

        other_params = self.params_num(self.emu_model.visual) + \
                       self.params_num(self.emu_model.decoder.lm.project_down) + \
                       self.params_num(self.emu_model.decoder.lm.project_up) + \
                       self.params_num(self.emu_model.decoder.lm.model.embed_tokens) + \
                       self.params_num(self.emu_model.decoder.lm.model.norm) + \
                       self.params_num(self.emu_model.decoder.lm.lm_head)

        layer_params = self.params_num(self.emu_model.decoder.lm.model.layers[0])
        layer_num = len(self.emu_model.decoder.lm.model.layers)

        total_params = other_params + layer_params * layer_num
        params_per_device = [total_params / len(device_list) for _ in device_list]
        params_per_device[0] -= other_params

        accumulate_params, device_idx = 0, 0
        for idx in range(layer_num):
            mp_rule[f"emu_model.decoder.lm.model.layers.{idx}"] = device_list[device_idx]
            accumulate_params += layer_params
            if accumulate_params > params_per_device[device_idx] and device_idx < len(device_list) - 1:
                accumulate_params = 0
                device_idx += 1

        self.parallel(mp_rule)
        return self

    def multito(self, device_list: List[str | torch.device]):
        return self.multicuda(device_list)
