# -*- coding: utf-8 -*-

from PIL import Image
from typing import List, Optional, Tuple
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as TF

from diffusers import UNet2DConditionModel, EulerDiscreteScheduler, AutoencoderKL
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor

from .emu import EmuModel
from .constants import EVA_IMAGE_SIZE, OPENAI_DATASET_MEAN, OPENAI_DATASET_STD, DEFAULT_IMG_PLACEHOLDER
from .mixin import ModelParallelMixin

class EmuVisualGeneration(nn.Module, ModelParallelMixin):

    def __init__(
        self,
        encoder: str,
        scheduler: str,
        unet: str,
        vae: str,
        feature_extractor: Optional[str] = None,
        safety_checker: Optional[str] = None,
        eva_size=EVA_IMAGE_SIZE,
        eva_mean=OPENAI_DATASET_MEAN,
        eva_std=OPENAI_DATASET_STD,
        **kwargs,
    ):

        super().__init__()

        self.emu_model = EmuModel()
        print(self.emu_model.load_state_dict(encoder, strict=False))

        self.unet = UNet2DConditionModel.from_pretrained(unet)
        self.vae = AutoencoderKL.from_pretrained(vae)
        self.scheduler = EulerDiscreteScheduler.from_pretrained(scheduler)

        self.safety_checker = None
        if safety_checker is not None:
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_checker)

        self.feature_extractor = None
        if feature_extractor is not None:
            self.feature_extractor = CLIPImageProcessor.from_pretrained(feature_extractor)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.eval()

        self.transform = TF.Compose([
            TF.Resize((eva_size, eva_size), interpolation=TF.InterpolationMode.BICUBIC),
            TF.ToTensor(),
            TF.Normalize(mean=eva_mean, std=eva_std),
        ])

        self.negative_prompt = {}

    @torch.no_grad()
    def forward(
        self,
        inputs: List[Image.Image | str] | str | Image.Image,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.,
        crop_info: Tuple[int, int] = [0, 0],
        original_size: Tuple[int, int] = [1024, 1024],
    ):
        if not isinstance(inputs, list):
            inputs = [inputs]

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self.emu_model.device
        dtype = self.emu_model.dtype

        do_classifier_free_guidance = guidance_scale > 1.0

        # 1. Encode input prompt
        prompt_embeds = self._prepare_and_encode_inputs(
            inputs,
            device,
            dtype,
            do_classifier_free_guidance,
        )
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
        image = self.decode_latents(latents)

        # 6. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(
            image,
            device,
            dtype
        )

        # 7. Convert to PIL
        image = self.numpy_to_pil(image)
        return image[0], has_nsfw_concept[0] if has_nsfw_concept is not None else image[0]

    def _prepare_and_encode_inputs(
        self,
        inputs: List[str | Image.Image],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        do_classifier_free_guidance: bool = False,
        placeholder: str = DEFAULT_IMG_PLACEHOLDER,
    ):
        has_image, has_text = False, True
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
            image_prompt = image_prompt.type(dtype).to(device)

        if has_image and not has_text:
            text_prompt = None
            prompt = self.emu_model.encode_image(image=image_prompt)
            if do_classifier_free_guidance:
                key = "[NULL_IMAGE]"
                if key not in self.negative_prompt:
                    negative_image = torch.zeros_like(image_prompt)
                    self.negative_prompt[key]= self.emu_model.encode_image(image=negative_image)
                prompt = torch.cat([prompt, self.negative_prompt[key]], dim=0)
        else:
            prompt = self.emu_model.generate_image(text=[text_prompt], image=image_prompt)
            if do_classifier_free_guidance:
                key = ""
                if key not in self.negative_prompt:
                    self.negative_prompt[key] = self.emu_model.generate_image(text=[key])
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
        encoder = kwargs.pop("encoder", None)
        feature_extractor = kwargs.pop("feature_extractor", None)
        safety_checker = kwargs.pop("safety_checker", None)
        scheduler = kwargs.pop("scheduler", None)
        unet = kwargs.pop("unet", None)
        vae = kwargs.pop("vae", None)

        check_if_none = lambda x, y: y if x is None else x

        encoder = check_if_none(encoder, f"{path}/multimodal_encoder/pytorch_model.bin")
        feature_extractor = check_if_none(feature_extractor, f"{path}/feature_extractor")
        safety_checker = check_if_none(safety_checker, f"{path}/safety_checker")
        scheduler = check_if_none(scheduler, f"{path}/scheduler")
        unet = check_if_none(unet, f"{path}/unet")
        vae = check_if_none(vae, f"{path}/vae")

        return cls(
            encoder=encoder,
            feature_extractor=feature_extractor,
            safety_checker=safety_checker,
            scheduler=scheduler,
            unet=unet,
            vae=vae,
            **kwargs,
        )

    def multicuda(
        self,
        device_list: List[str | torch.device],
    ):
        """
            unet:                                     2.8B
            vae:                                      0.xB
            emu_model.visual:                           4B
            emu_model.decoder.lm.project_down:        omit
            emu_model.decoder.lm.project_up:          omit
            emu_model.decoder.lm.model.embed_tokens:  omit
            emu_model.decoder.lm.model.norm:          omit
            emu_model.decoder.lm.lm_head:             omit
            emu_model.decoder.lm.model.layers.[0..59]: 33B (0.55B/layer)
        """
        mp_rule = {
            "unet": device_list[0],
            "vae": device_list[0],
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
                       self.params_num(self.emu_model.decoder.lm.lm_head) + \
                       self.params_num(self.unet) + \
                       self.params_num(self.vae)

        if self.safety_checker is not None:
            mp_rule["safety_checker"] = device_list[0]
            other_params += self.params_num(self.safety_checker)

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
        self.vae.decode = self._forward_hook(self.vae, self.vae.decode, pre=True, post=False)
        return self

    def multito(self, device_list: List[str | torch.device]):
        return self.multicuda(device_list)
