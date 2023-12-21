""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from functools import partial

import torch

from .eva_vit_model import EVAVisionTransformer
from .transformer import LayerNormFp32, LayerNorm


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    drop_path_rate: Optional[float] = None  # drop path rate

    eva_model_name: str = None  # a valid eva model name overrides layers, width, patch_size
    qkv_bias: bool = True
    fusedLN: bool = False
    xattn: bool = False
    postnorm: bool = False
    rope: bool = False
    pt_hw_seq_len: int = 16  # 224/14
    intp_freq: bool = False
    naiveswiglu: bool = False
    subln: bool = False

    freeze: bool = False


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'

    # Coca Cfg
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False

    masked_language_modeling: bool = False
    fusedLN: bool = False
    xattn: bool = False
    attn_mask: bool = True

    max_txt_len: int = 512

    # llm as txt encoder
    llm_name: str = None
    attn_pooler_heads: int = 8

    freeze: bool = False


@dataclass
class MultimodalCfg(CLIPTextCfg):
    name: str = 'CoCa'
    mlp_ratio: int = 4
    dim_head: int = 64
    heads: int = 8
    n_causal: int = 32
    attn_pooler_heads: int = 8

    vl_adapter: str = "attnpool"
    freeze: bool = True


@dataclass
class VLadapterCfg:
    name: str = "cformer"
    n_causal: int = 32


def get_cast_dtype(precision: str):
    cast_dtype = None
    if 'bf16' in precision:
        cast_dtype = torch.bfloat16
    elif 'fp16' in precision:
        cast_dtype = torch.float16
    return cast_dtype


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    if vision_cfg.eva_model_name:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        visual = EVAVisionTransformer(
            img_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            num_classes=embed_dim,
            use_mean_pooling=vision_cfg.global_average_pool,  # False
            init_values=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            embed_dim=vision_cfg.width,
            depth=vision_cfg.layers,
            num_heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            qkv_bias=vision_cfg.qkv_bias,
            drop_path_rate=vision_cfg.drop_path_rate,
            norm_layer=partial(norm_layer, eps=1e-6),
            xattn=vision_cfg.xattn,
            rope=vision_cfg.rope,
            postnorm=vision_cfg.postnorm,
            pt_hw_seq_len=vision_cfg.pt_hw_seq_len,  # 224/14
            intp_freq=vision_cfg.intp_freq,
            naiveswiglu=vision_cfg.naiveswiglu,
            subln=vision_cfg.subln,
        )

    return visual

