"""
Adapted from  https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_qformer.py
"""

import torch
import torch.nn as nn

from transformers.utils import logging
from transformers.models.t5.configuration_t5 import T5Config
from .modeling_t5 import T5ForConditionalGeneration

logger = logging.get_logger(__name__)


class CausalFormer(nn.Module):
    def __init__(
            self,
            args,
            n_causal: int = 32,
            vision_width: int = 768,
            output_dim: int = 512,
    ):
        super(CausalFormer, self).__init__()

        model_config = T5Config.from_pretrained("t5-base")
        model_config.encoder_width = vision_width

        lm = T5ForConditionalGeneration(config=model_config)
        lm.shared = None  # causal tokens dont need such embeddings

        self.cformer = lm.decoder

        for name, param in self.cformer.named_parameters():
            param.data = param.data.bfloat16()

        self.causal_tokens = nn.Parameter(
            torch.zeros(1, n_causal, model_config.d_model)
        )
        self.causal_tokens.data.normal_(mean=0.0, std=0.02)

        self.projection = nn.Linear(model_config.d_model, output_dim)

    def forward(self, img_embeds):
        encoder_attention_mask = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to(img_embeds.device)

        causal_tokens = self.causal_tokens.expand(img_embeds.shape[0], -1, -1).to(img_embeds.device)

        decoder_attention_mask = torch.ones(causal_tokens.size()[:-1], dtype=torch.long).to(img_embeds.device)
        causal_output = self.cformer(
            attention_mask=decoder_attention_mask,
            inputs_embeds=causal_tokens,
            encoder_hidden_states=img_embeds,
            encoder_attention_mask=encoder_attention_mask,
            # output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        causal_output = causal_output.last_hidden_state

        causal_output = self.projection(causal_output)  # [B, n_causal, output_dim]

        return causal_output

    def set_grad_checkpointing(self):
        self.cformer.config.use_cache = False
        self.cformer.gradient_checkpointing_enable()

