# -*- coding: utf-8 -*-

# ===========================================================================================
#
#    Copyright (c) Beijing Academy of Artificial Intelligence (BAAI). All rights reserved.
#
#    Author        : Fan Zhang
#    Email         : zhangfan@baai.ac.cn
#    Institute     : Beijing Academy of Artificial Intelligence (BAAI)
#    Create On     : 2023-12-08 07:30
#    Last Modified : 2023-12-25 04:33
#    File Name     : diffusion.py
#    Description   :
#
# ===========================================================================================

from dataclasses import dataclass
from PIL import Image
from typing import List, Optional
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as TF

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from diffusers.utils import BaseOutput
from diffusers import UNet2DConditionModel, EulerDiscreteScheduler, AutoencoderKL
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer

from .constants import EVA_IMAGE_SIZE, OPENAI_DATASET_MEAN, OPENAI_DATASET_STD, DEFAULT_IMG_PLACEHOLDER
from .mixin import ModelParallelMixin


@dataclass
class EmuVisualGenerationPipelineOutput(BaseOutput):
    image: Image.Image
    nsfw_content_detected: Optional[bool]


class EmuVisualGeneration(nn.Module, ModelParallelMixin):

    def __init__(
        self,
        tokenizer: str,
        multimodal_encoder: str,
        scheduler: str,
        unet: str,
        vae: str,
        feature_extractor: Optional[str] = None,
        safety_checker: Optional[str] = None,
        eva_size=EVA_IMAGE_SIZE,
        eva_mean=OPENAI_DATASET_MEAN,
        eva_std=OPENAI_DATASET_STD,
        device_list: List[torch.device] = [torch.device("cpu")],
        torch_dtype: torch.dtype = torch.bfloat16,
        quantize: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.unet = UNet2DConditionModel.from_pretrained(
            unet,
            torch_dtype=torch_dtype,
            **kwargs
        )
        self.vae = AutoencoderKL.from_pretrained(
            vae,
            torch_dtype=torch_dtype,
            **kwargs,
        )
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            scheduler,
        )

        self.safety_checker = None
        if safety_checker is not None:
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                safety_checker,
                torch_dtype=torch_dtype,
                **kwargs,
            )

        self.feature_extractor = None
        if feature_extractor is not None:
            self.feature_extractor = CLIPImageProcessor.from_pretrained(
                feature_extractor,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        with init_empty_weights():
            self.multimodal_encoder = AutoModelForCausalLM.from_pretrained(
                multimodal_encoder,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                **kwargs,
            )

        mme_device_map, vsd_device_map = self.map_device(device_list)

        if quantize:
            self.multimodal_encoder = AutoModelForCausalLM.from_pretrained(
                multimodal_encoder,
                load_in_4bit=True,
                trust_remote_code=True,
                bnb_4bit_compute_dtype=torch.float16,
                device_map=mme_device_map
            )
        else:
            mme_device_map = {k: str(v) for k, v in mme_device_map.items()}
            self.multimodal_encoder = load_checkpoint_and_dispatch(
                self.multimodal_encoder,
                multimodal_encoder,
                dtype=torch_dtype,
                device_map=mme_device_map,
            )

        if len(device_list) != 1 or (torch.device("cpu") not in device_list and "cpu" not in device_list):
            for l, d in mme_device_map.items():
                print(f"put {l} to device {d}")

            self.parallel(vsd_device_map)
            self.vae.decode = self._forward_hook(self.vae, self.vae.decode, pre=True, post=False)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.eval()

        self.transform = TF.Compose([
            TF.Resize((eva_size, eva_size), interpolation=TF.InterpolationMode.BICUBIC),
            TF.ToTensor(),
            TF.Normalize(mean=eva_mean, std=eva_std),
        ])

        self.negative_prompt = {}

    def device(self, module=None):
        if module is None:
            return next(self.parameters()).device
        return next(module.parameters()).device

    def dtype(self, module):
        if module is None:
            return next(self.parameters()).dtype
        return next(module.parameters()).dtype

    @torch.no_grad()
    def forward(
        self,
        inputs: List[Image.Image | str] | str | Image.Image,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.,
        crop_info: List[int] = [0, 0],
        original_size: List[int] = [1024, 1024],
    ):
        if not isinstance(inputs, list):
            inputs = [inputs]

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self.device(self.unet)
        dtype = self.dtype(self.unet)

        do_classifier_free_guidance = guidance_scale > 1.0

        # 1. Encode input prompt
        prompt_embeds = self._prepare_and_encode_inputs(
            inputs,
            do_classifier_free_guidance,
        ).to(device=device, dtype=dtype)
        batch_size = prompt_embeds.shape[0] // 2 if do_classifier_free_guidance else prompt_embeds.shape[0]

        unet_added_conditions = {}
        time_ids = torch.LongTensor(original_size + crop_info + [height, width]).to(device)
        if do_classifier_free_guidance:
            unet_added_conditions["time_ids"] = torch.cat([time_ids, time_ids], dim=0)
        else:
            unet_added_conditions["time_ids"] = time_ids
        unet_added_conditions["text_embeds"] = torch.mean(prompt_embeds, dim=1)

        # 2. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 3. Prepare latent variables
        shape = (
            batch_size,
            self.unet.config.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        latents = torch.randn(shape, device=device, dtype=dtype)
        latents = latents * self.scheduler.init_noise_sigma

        # 4. Denoising loop
        for t in tqdm(timesteps):
            # expand the latents if we are doing classifier free guidance
            # 2B x 4 x H x W
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=unet_added_conditions,
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 5. Post-processing
        images = self.decode_latents(latents)

        # 6. Run safety checker
        images, has_nsfw_concept = self.run_safety_checker(
            images,
            device,
            dtype
        )

        # 7. Convert to PIL
        images = self.numpy_to_pil(images)
        return EmuVisualGenerationPipelineOutput(
            image=images[0],
            nsfw_content_detected=None if has_nsfw_concept is None else has_nsfw_concept[0],
        )

    def _prepare_and_encode_inputs(
        self,
        inputs: List[str | Image.Image],
        do_classifier_free_guidance: bool = False,
        placeholder: str = DEFAULT_IMG_PLACEHOLDER,
    ):
        device = self.device(self.multimodal_encoder.model.visual)
        dtype = self.dtype(self.multimodal_encoder.model.visual)

        has_image, has_text = False, False
        text_prompt, image_prompt = "", []
        for x in inputs:
            if isinstance(x, str):
                has_text = True
                text_prompt += x
            else:
                has_image = True
                text_prompt += placeholder
                image_prompt.append(self.transform(x))

        if len(image_prompt) == 0:
            image_prompt = None
        else:
            image_prompt = torch.stack(image_prompt)
            image_prompt = image_prompt.to(device=device, dtype=dtype)

        # Enable Autoencoding Mode, you can ONLY input exactly one image
        if has_image and not has_text:
            text_prompt = None
            prompt = self.multimodal_encoder.model.encode_image(image=image_prompt)
            if do_classifier_free_guidance:
                key = "[NULL_IMAGE]"
                if key not in self.negative_prompt:
                    negative_image = torch.zeros_like(image_prompt)
                    self.negative_prompt[key]= self.multimodal_encoder.model.encode_image(image=negative_image)
                prompt = torch.cat([prompt, self.negative_prompt[key]], dim=0)
        # Enable Image Generation Mode
        else:
            prompt = self.multimodal_encoder.generate_image(text=[text_prompt], image=image_prompt, tokenizer=self.tokenizer)
            if do_classifier_free_guidance:
                key = ""
                if key not in self.negative_prompt:
                    self.negative_prompt[key] = self.multimodal_encoder.generate_image(text=[""], tokenizer=self.tokenizer)
                prompt = torch.cat([prompt, self.negative_prompt[key]], dim=0)

        return prompt

    def decode_latents(self, latents: torch.Tensor) -> np.ndarray:
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def numpy_to_pil(self, images: np.ndarray) -> List[Image.Image]:
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def run_safety_checker(
        self,
        image: np.ndarray,
        device: torch.device,
        dtype: torch.dtype,
    ):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        **kwargs,
    ):
        tokenizer = kwargs.pop("tokenizer", None)
        multimodal_encoder = kwargs.pop("multimodal_encoder", None)
        feature_extractor = kwargs.pop("feature_extractor", None)
        safety_checker = kwargs.pop("safety_checker", None)
        scheduler = kwargs.pop("scheduler", None)
        unet = kwargs.pop("unet", None)
        vae = kwargs.pop("vae", None)

        check_if_none = lambda x, y: y if x is None else x

        tokenizer = check_if_none(tokenizer, f"{path}/tokenizer")
        multimodal_encoder = check_if_none(multimodal_encoder, f"{path}/multimodal_encoder")
        feature_extractor = check_if_none(feature_extractor, f"{path}/feature_extractor")
        safety_checker = check_if_none(safety_checker, f"{path}/safety_checker")
        scheduler = check_if_none(scheduler, f"{path}/scheduler")
        unet = check_if_none(unet, f"{path}/unet")
        vae = check_if_none(vae, f"{path}/vae")

        return cls(
            tokenizer=tokenizer,
            multimodal_encoder=multimodal_encoder,
            feature_extractor=feature_extractor,
            safety_checker=safety_checker,
            scheduler=scheduler,
            unet=unet,
            vae=vae,
            **kwargs,
        )

    def map_device(
        self,
        device_list: List[str | torch.device],
    ):
        """
            A simple multi device strategy, which distribute blocks in large language modles averagely
            into multi devices while keeping unet, vae, safety checker and rest layers in LLM on the first device
            unet:                                                    2.8B [cuda:0]
            vae:                                                     0.xB [cuda:0]
            safety_checker:                                            yB [cuda:0]
            multimodal_encoder.project_down:                         omit [cuda:0]
            multimodal_encoder.project_up:                           omit [cuda:0]
            multimodal_encoder.model.visual:                           4B [cuda:0]
            multimodal_encoder.model.decoder.lm.model.embed_tokens:  omit [cuda:0]
            multimodal_encoder.model.decoder.lm.model.norm:          omit [cuda:0]
            multimodal_encoder.model.decoder.lm.lm_head:             omit [cuda:0]
            multimodal_encoder.model.decoder.lm.model.layers.[0..59]: 33B (0.55B/layer) [cuda:0 ~ cuda:x]
        """
        mp_rule = {
            "unet": device_list[0],
            "vae": device_list[0],
            "multimodal_encoder.project_down": device_list[0],
            "multimodal_encoder.project_up": device_list[0],
            "multimodal_encoder.model.visual": device_list[0],
            "multimodal_encoder.model.decoder.lm.model.embed_tokens": device_list[0],
            "multimodal_encoder.model.decoder.lm.model.norm": device_list[0],
            "multimodal_encoder.model.decoder.lm.lm_head": device_list[0],
        }

        other_params = self.params_num(self.multimodal_encoder.project_down) + \
                       self.params_num(self.multimodal_encoder.project_up) + \
                       self.params_num(self.multimodal_encoder.model.visual) + \
                       self.params_num(self.multimodal_encoder.model.decoder.lm.model.embed_tokens) + \
                       self.params_num(self.multimodal_encoder.model.decoder.lm.model.norm) + \
                       self.params_num(self.multimodal_encoder.model.decoder.lm.lm_head) + \
                       self.params_num(self.unet) + \
                       self.params_num(self.vae)

        if self.safety_checker is not None:
            mp_rule["safety_checker"] = device_list[0]
            other_params += self.params_num(self.safety_checker)

        layer_params = self.params_num(self.multimodal_encoder.model.decoder.lm.model.layers[0])
        layer_num = len(self.multimodal_encoder.model.decoder.lm.model.layers)

        total_params = other_params + layer_params * layer_num
        params_per_device = [total_params / len(device_list) for _ in device_list]
        params_per_device[0] -= other_params

        accumulate_params, device_idx = 0, 0
        for idx in range(layer_num):
            if accumulate_params + layer_params > params_per_device[device_idx] and device_idx < len(device_list) - 1:
                accumulate_params = 0
                device_idx += 1

            mp_rule[f"multimodal_encoder.model.decoder.lm.model.layers.{idx}"] = device_list[device_idx]
            accumulate_params += layer_params

        mme_rule = {k.replace("multimodal_encoder.", ""): v for k, v in mp_rule.items() if k.startswith("multimodal_encoder")}
        vsd_rule = {k: v for k, v in mp_rule.items() if not k.startswith("multimodal_encoder")}

        return mme_rule, vsd_rule
