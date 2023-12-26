from dataclasses import dataclass
from typing import Optional
import os.path as osp


@dataclass
class CLIPVisionCfg:
    eva_model_name: str = "eva-clip-4b-14-x"

    image_size: int = 448
    patch_size: int = 14
    width: int = 1792
    layers: int = 64
    head_width: int = 112
    mlp_ratio: float = 8.571428571428571

    qkv_bias: bool = True
    drop_path_rate: float = 0.

    init_value: Optional[float] = None
    patch_dropout: float = 0.
    rope: bool = False
    global_average_pool: bool = False

    xattn: bool = False
    postnorm: bool = True
    pt_hw_seq_len: int = 16
    intp_freq: bool = False
    naiveswiglu: bool = False
    subln: bool = False

    n_query: int = 64
    v_query: int = 64


@dataclass
class TextDecoderCfg:
    llama_config_path: str = osp.join(osp.dirname(__file__), "llama_config")
    instruct: bool = False

