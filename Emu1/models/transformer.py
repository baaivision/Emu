import os
import logging
import math
from typing import Callable, Optional
import torch
from torch import nn
from torch.nn import functional as F

try:
    import xformers.ops as xops
except ImportError:
    xops = None
    print("Please 'pip install xformers'")


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token
        logging.info(f"os.getenv('RoPE')={os.getenv('RoPE')}")

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        if self.training and os.getenv('RoPE') == '1':
            return x, patch_indices_keep

        return x


def _in_projection_packed(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor] = None,
):
    """
    https://github.com/pytorch/pytorch/blob/db2a237763eb8693a20788be94f8c192e762baa8/torch/nn/functional.py#L4726
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return F.linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            kv_dim=None,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.,
            xattn=False,
            rope=False
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        self.kv_dim = kv_dim if kv_dim is not None else dim
        self._qkv_same_dim = self.kv_dim == dim

        if self._qkv_same_dim:  # self-attn & cross-attn with qkv same dim
            # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
            self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
            if qkv_bias:
                self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
            else:
                self.in_proj_bias = None
        else:  # cross-attn with qkv different dim
            # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
            self.q_proj_weight = nn.Parameter(torch.randn((dim, dim)) * self.scale)
            self.k_proj_weight = nn.Parameter(torch.randn((dim, self.kv_dim)) * self.scale)
            self.v_proj_weight = nn.Parameter(torch.randn((dim, self.kv_dim)) * self.scale)
            if qkv_bias:
                self.q_proj_bias = nn.Parameter(torch.zeros(dim))
                self.k_proj_bias = nn.Parameter(torch.zeros(dim))
                self.v_proj_bias = nn.Parameter(torch.zeros(dim))
            else:
                self.q_proj_bias = None
                self.k_proj_bias = None
                self.v_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        # self.attn_drop = xtriton.FusedDropoutBias(p=attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        # self.out_proj = xtriton.FusedLinear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)
        # self.out_drop = xtriton.FusedDropoutBias(p=proj_drop)
        self.xattn = xattn
        self.xattn_drop = attn_drop
        self.rope = rope

    def forward(
            self,
            q: torch.Tensor,
            k: Optional[torch.Tensor] = None,
            v: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None):
        """
        if k and v is None, self-attn, else cross-attn
        :param q:
        :param k:
        :param v:
        :param attn_mask:
        :return:
        """
        L, N, C = q.shape
        k_L = L
        # TODO: check cross attn
        if self._qkv_same_dim and k is None:  # self attn
            q, k, v = F.linear(q, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        elif self._qkv_same_dim and k is not None:  # cross-attn with same dim
            k_L = k.shape[0]
            q, k, v = _in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)
        else:  # cross-attn with different dim
            k_L = k.shape[0]
            q = F.linear(q, self.q_proj_weight, self.q_proj_bias)
            k = F.linear(k, self.k_proj_weight, self.k_proj_bias)
            v = F.linear(v, self.k_proj_weight, self.k_proj_bias)
        if self.xattn:
            q = q.contiguous().view(L, N, self.num_heads, -1).transpose(0, 1)
            k = k.contiguous().view(k_L, N, self.num_heads, -1).transpose(0, 1)
            v = v.contiguous().view(k_L, N, self.num_heads, -1).transpose(0, 1)

            x = xops.memory_efficient_attention(
                q, k, v,
                p=self.xattn_drop,
                scale=self.scale if self.logit_scale is None else None,
                attn_bias=xops.LowerTriangularMask() if attn_mask is not None else None,
                # op=xops.MemoryEfficientAttentionFlashAttentionOp
            )
            # if self.head_scale is not None:
            #     x = x.view(N, self.num_heads, L, C) * self.head_scale
            #     x = x.view(-1, L, C)
            # x = x.transpose(0, 1).reshape(L, N, C)
            # x = self.out_proj(x)
            # x = self.out_drop(x)
        else:
            q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
            k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
            v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

            if self.logit_scale is not None:
                attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
                logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
                attn = attn.view(N, self.num_heads, L, L) * logit_scale
                attn = attn.view(-1, L, L)
            else:
                q = q * self.scale
                attn = torch.bmm(q, k.transpose(-1, -2))

            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                    new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                    attn_mask = new_attn_mask
                attn += attn_mask

            attn = attn.softmax(dim=-1)
            # attn = xtriton.softmax(attn)
            attn = self.attn_drop(attn)

            x = torch.bmm(attn, v)

        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class AttentionalPooler(nn.Module):
    """
    randomly initialize N queries
    input: x: [B, n_token, C (context_dim)] as K and V
    output: queries: [B, n_query, d_query (d_model)]
    """

    def __init__(
            self,
            d_model: int,
            context_dim: int,  # KV dim
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNorm,
            xattn=False,
    ):
        super().__init__()

        self.n_head = n_head

        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        # TODO, replace the nn.MHA to my own attention
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)
        self.xattn = xattn

    def forward(self, x: torch.Tensor, attn_mask=None):

        """
        :param x: [B, n_token, C (context_dim)] as K and V
        :param attn_mask: 1, 0
        :return: queries: [B, n_query, d_query (d_model)]
        """

        B, seq_len, C = x.shape
        if attn_mask is None:
            attn_mask = torch.ones(
                (B, seq_len), dtype=torch.bool, device=x.device
            )
        # process attn_mask as nn.MHA's input format
        q_len = self.query.shape[0]
        expanded_mask = attn_mask[:, None, :].expand(B, q_len, seq_len).repeat(self.n_head, 1, 1).to(x.dtype)
        inverted_mask = 1.0 - expanded_mask
        attn_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(x.dtype).min)

        x = self.ln_k(x).permute(1, 0, 2)  # [B, n_seq, C] -> [n_seq, B, C]
        N = x.shape[1]
        q = self.ln_q(self.query)  # [n_query, C_query]
        out = self.attn(self._repeat(q, N), x, x, need_weights=False, attn_mask=attn_mask)[0]
        return out.permute(1, 0, 2)  # [n_seq, B, C] -> [B, n_seq, C]

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)

