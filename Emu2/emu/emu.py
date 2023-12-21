from functools import partial
from typing import Any, List, Optional, Mapping
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from transformers.generation.configuration_utils import GenerationConfig
GENERATION_CONFIG = GenerationConfig(bos_token_id=1, eos_token_id=2, pad_token_id=32000)

from .conf.emu_conf import CLIPVisionCfg, TextDecoderCfg

from .constants import *
from .eva_vit import EVAVisionTransformer
from .lm import EmuForClsAndRegression


class EmuModel(nn.Module):

    def __init__(
        self,
        vision_cfg: CLIPVisionCfg = CLIPVisionCfg(),
        text_decoder_cfg: TextDecoderCfg = TextDecoderCfg(),
    ):
        super().__init__()

        self.visual = EVAVisionTransformer(
            img_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            embed_dim=vision_cfg.width,
            depth=vision_cfg.layers,
            num_heads=vision_cfg.width // vision_cfg.head_width,
            mlp_ratio=vision_cfg.mlp_ratio,
            qkv_bias=vision_cfg.qkv_bias,
            drop_path_rate=vision_cfg.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=vision_cfg.init_value,
            patch_dropout=vision_cfg.patch_dropout,
            rope=vision_cfg.rope,
            use_mean_pooling=vision_cfg.global_average_pool,  # False
            xattn=vision_cfg.xattn,
            postnorm=vision_cfg.postnorm,
            pt_hw_seq_len=vision_cfg.pt_hw_seq_len,  # 224/14
            intp_freq=vision_cfg.intp_freq,
            naiveswiglu=vision_cfg.naiveswiglu,
            subln=vision_cfg.subln,
        )

        self.decoder = EmuForClsAndRegression(args=text_decoder_cfg, d_model=vision_cfg.width)

        # EmuModel is for inference only, so set padding and truncation to left
        self.decoder.tokenizer.truncation_side = self.decoder.tokenizer.padding_side = "left"

        self.n_query = vision_cfg.n_query
        self.v_query = vision_cfg.v_query
        self.image_placeholder = DEFAULT_IMG_TOKEN + DEFAULT_IMAGE_TOKEN * self.n_query + DEFAULT_IMG_END_TOKEN

        # temporarily borrow [gIMG] as the video frame feature placeholder.
        self.video_placeholder = DEFAULT_IMG_TOKEN + DEFAULT_gIMG_TOKEN * self.v_query + DEFAULT_IMG_END_TOKEN

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor, *, n_query=None):
        n_query = n_query if n_query is not None else self.n_query

        image_embeds = self.visual(image)
        image_embeds = image_embeds[:, 1:, :]
        b, n, c = image_embeds.shape
        sqrt_n = int(n**0.5)
        image_embeds = image_embeds.permute(0, 2, 1).view(b, c, sqrt_n, sqrt_n)

        stride = int(sqrt_n // (n_query ** 0.5))
        image_embeds = F.avg_pool2d(image_embeds, kernel_size=(stride, stride), stride=stride)
        image_embeds = image_embeds.view(b, c, -1).permute(0, 2, 1).contiguous()
        return image_embeds

    @torch.no_grad()
    def generate_image(
        self,
        text: List[str],
        image: Optional[torch.Tensor] = None,
        placeholder: str = DEFAULT_IMG_PLACEHOLDER,
    ):
        IMAGE, BOI = self.decoder.tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_TOKEN, DEFAULT_IMG_TOKEN])
        if image is not None:
            prompt_image_embeds = self.encode_image(image)
            _, _, c = prompt_image_embeds.shape
            prompt_image_embeds = prompt_image_embeds.view(-1, c)
            prompt_image_embeds = self.decoder.lm.project_up(prompt_image_embeds)

        text = [t.replace(placeholder, self.image_placeholder) for t in text]

        target_image_embeds = None
        for num_img_token in range(self.n_query):
            if num_img_token == 0:
                text = [f"{t}{DEFAULT_IMG_TOKEN}" for t in text]
            else:
                text = [f"{t}{DEFAULT_IMAGE_TOKEN}" for t in text]

            inputs = self.decoder.tokenizer(text, padding="longest", return_tensors="pt")

            device = self.decoder.lm.model.embed_tokens.weight.device
            input_ids = inputs.input_ids.to(device) # B x N
            text_embeds = self.decoder.lm.model.embed_tokens(input_ids)

            attention_mask = inputs.attention_mask.to(text_embeds.device)

            image_idx = (input_ids == IMAGE)
            cumsum_idx = torch.flip(torch.cumsum(torch.flip(image_idx, dims=[1]), dim=1), dims=[1])
            if image is not None:
                prompt_idx = torch.logical_and(image_idx, cumsum_idx > num_img_token)
                text_embeds[prompt_idx] = prompt_image_embeds.to(text_embeds.device)

            if target_image_embeds is not None:
                target_idx = torch.logical_and(image_idx, torch.logical_and(cumsum_idx > 0, cumsum_idx <= num_img_token))
                text_embeds[target_idx] = self.decoder.lm.project_up(target_image_embeds).to(text_embeds.device)

            outputs = self.decoder.lm.model(
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            image_idx = (input_ids == IMAGE) + (input_ids == BOI)
            cumsum_idx = torch.flip(torch.cumsum(torch.flip(image_idx, dims=[1]), dim=1), dims=[1])
            target_idx = torch.logical_and(image_idx, torch.logical_and(cumsum_idx > 0, cumsum_idx <= num_img_token+1))

            hidden_states = outputs.hidden_states[-1]
            target_image_embeds = hidden_states[target_idx]
            target_image_embeds = target_image_embeds.view(-1, target_image_embeds.shape[-1])
            target_image_embeds = self.decoder.lm.project_down(target_image_embeds)

        _, C = target_image_embeds.shape
        B = hidden_states.shape[0]
        target_image_embeds = target_image_embeds.view(B, -1, C)

        return target_image_embeds

    @torch.no_grad()
    def generate(
        self,
        text: List[str],
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        image_placeholder: str = DEFAULT_IMG_PLACEHOLDER,
        video_placeholder: str = DEFAULT_VID_PLACEHOLDER,
        num_beams=5,
        max_new_tokens=10,
        min_len=1,
        do_sample=False,
        penalty_alpha=None,
        top_p=None,
        top_k=None,
        temperature=None,
        length_penalty=-1,
        repetition_penalty=1.0,
        synced_gpus=False,
        skip_special_tokens=True,
        **kwargs
    ):

        GENERATION_CONFIG.pad_token_id = self.decoder.tokenizer.pad_token_id
        GENERATION_CONFIG.bos_token_id = self.decoder.tokenizer.bos_token_id
        GENERATION_CONFIG.eos_token_id = self.decoder.tokenizer.eos_token_id

        IMAGE, VIDEO = self.decoder.tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_TOKEN, DEFAULT_gIMG_TOKEN])

        text = [
            t.replace(image_placeholder, self.image_placeholder).replace(video_placeholder, self.video_placeholder)
            for t in text
        ]

        inputs = self.decoder.tokenizer(text, padding="longest", return_tensors="pt")

        device = self.decoder.lm.model.embed_tokens.weight.device
        input_ids = inputs.input_ids.to(device) # B x N
        text_embeds = self.decoder.lm.model.embed_tokens(input_ids)

        attention_mask = inputs.attention_mask.to(text_embeds.device)

        if image is not None:
            prompt_image_embeds = self.encode_image(image, n_query=self.n_query)
            _, _, c = prompt_image_embeds.shape
            prompt_image_embeds = prompt_image_embeds.view(-1, c)
            prompt_image_embeds = self.decoder.lm.project_up(prompt_image_embeds)
            image_idx = (input_ids == IMAGE)
            text_embeds[image_idx] = prompt_image_embeds.to(text_embeds.device)

        if video is not None:
            prompt_video_embeds = self.encode_image(video, n_query=self.v_query)
            _, _, c = prompt_video_embeds.shape
            prompt_video_embeds = prompt_video_embeds.view(-1, c)
            prompt_video_embeds = self.decoder.lm.project_up(prompt_video_embeds)
            video_idx = (input_ids == VIDEO)
            text_embeds[video_idx] = prompt_video_embeds.to(text_embeds.device)

        outputs = self.decoder.lm.generate(
            generation_config=GENERATION_CONFIG,
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_length=min_len,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            penalty_alpha=penalty_alpha,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            synced_gpus=synced_gpus or hasattr(next(self.parameters()), 'ds_tensor'),
            **kwargs,
        )
        output_text = self.decoder.tokenizer.batch_decode(
            outputs, skip_special_tokens=skip_special_tokens,
        )

        return output_text

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any] | str,
        *args,
        **kwargs,
    ):
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict, map_location="cpu")

        state_dict = state_dict["module"] if "module" in state_dict else state_dict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("decoder.lm.stu_regress_head"):
                new_state_dict[k.replace("decoder.lm.stu_regress_head", "decoder.lm.project_down")] = v
            elif k.startswith("vl_adapter.projection"):
                new_state_dict[k.replace("vl_adapter.projection", "decoder.lm.project_up")] = v
            else:
                new_state_dict[k] = v

        return super().load_state_dict(new_state_dict, *args, **kwargs)
