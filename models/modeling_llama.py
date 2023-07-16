"""
LLaMA model from transformers, following stanford's Alpaca
"""

MODEL_PATH = "./models/llama_config"

from typing import Dict, List, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn

import transformers
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig

IGNORE_INDEX = -100
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_UNK_TOKEN = "<unk>"

DEFAULT_IMG_START_TOKEN = "[IMG]"
DEFAULT_IMG_END_TOKEN = "[/IMG]"
DEFAULT_IMG_TOKEN = "<image>"
USER_TOKEN = '[USER]'
ASSISTANT_TOKEN = '[ASSISTANT]'


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
        resize_output: bool = True,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg

        if resize_output:
            output_embeddings = model.get_output_embeddings().weight.data
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings[-num_new_tokens:] = output_embeddings_avg


@dataclass
class RegressCausalLMOutputWithPast(CausalLMOutputWithPast):
    llm_loss: Optional[torch.FloatTensor] = None
    regression_loss: Optional[torch.FloatTensor] = None


class LlamaForReg(transformers.LlamaForCausalLM):
    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                regress_mask: torch.Tensor = None,
                img_length: int = None,
                args=None,
                regress_labels=None
                ):
        """
        :param self:
        :param inputs_embeds: shape [B, 1 + n_image + n_token, C]
        :param img_length: length of image tokens, not include the special `[IMG]` token
        :return:
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return RegressCausalLMOutputWithPast(
            llm_loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LLaMAForClsAndRegression(nn.Module):
    def __init__(self, args, model_name_or_path=MODEL_PATH):
        super(LLaMAForClsAndRegression, self).__init__()
        self.args = args

        # self.lm = LlamaForReg.from_pretrained(model_name_or_path)
        self.lm = LlamaForReg(config=LlamaConfig.from_pretrained(model_name_or_path))

        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(
            model_name_or_path,
            model_max_length=2048,
            padding_side="right",
            truncation=True,
            use_fast=False,
        )

        if args.instruct:  # for instruction tuning, [USER] and [ASSISTANT] tokens are added
            special_token_list = [DEFAULT_IMG_START_TOKEN, DEFAULT_IMG_END_TOKEN, DEFAULT_IMG_TOKEN, USER_TOKEN,
                                  ASSISTANT_TOKEN]
        else:
            special_token_list = [DEFAULT_IMG_START_TOKEN, DEFAULT_IMG_END_TOKEN, DEFAULT_IMG_TOKEN]

        special_tokens_dict = dict(
            pad_token=DEFAULT_PAD_TOKEN,
            bos_token=DEFAULT_BOS_TOKEN,
            eos_token=DEFAULT_EOS_TOKEN,
            unk_token=DEFAULT_UNK_TOKEN,
            additional_special_tokens=special_token_list
        )

        if self.tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=special_tokens_dict,
                tokenizer=self.tokenizer,
                model=self.lm,
            )

        self.lm.model.embed_tokens.padding_idx = self.tokenizer.pad_token_id
        print(f"The Special Tokens: {self.tokenizer.special_tokens_map}")
        print(f"Vocab Size: {len(self.tokenizer)}")

        # student head
        self.lm.stu_regress_head = nn.Linear(self.lm.config.hidden_size, self.lm.config.hidden_size, bias=False)

        self.config = self.lm.config
        self.lm.config.d_model = self.lm.config.hidden_size
        self.lm.bfloat16()

        self.prompt = None

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(['<image>'])[0]
        print(f"image_token_id: {self.image_token_id}")

        self.img_token_id = self.tokenizer.convert_tokens_to_ids(['[IMG]'])[0]
        print(f"[IMG] token id: {self.img_token_id}")

        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids(['[/IMG]'])[0]
        print(f"[/IMG] token id: {self.img_end_token_id}")

    def get_num_layers(self):
        return len(self.lm.model.layers)

    def forward(self, image_embeds, text_input, text_mask, text_output=None, output_mask=None):
        """
        Process:
        1. image_embeds & text_tokens as input
        2. prepend [IMG] token to img features or replace <ImagePatches> in <img><ImagePatches></img> with img features
        3. concat image and text features
        4. prepend <BOS> to sequence and append <EOS> to end of sequence
        4. feed into forward and return two losses

        :param image_embeds: [B, n_causal, C], after projected into Language shape
        :param text_input: [B, seq_len]
        :param text_mask: [B, seq_len]
        :return:
        """
        B, n_causal, _ = image_embeds.shape

        # mask [PAD]
        targets = text_input.masked_fill(
            text_input == self.tokenizer.pad_token_id, -100
        )
        # mask <image>
        targets = targets.masked_fill(
            targets == self.image_token_id, -100
        )
        # mask [IMG]
        targets = targets.masked_fill(
            targets == self.img_token_id, -100
        )
        # mask [/IMG]
        targets = targets.masked_fill(
            targets == self.img_end_token_id, -100
        )

        text_embeds = self.lm.model.embed_tokens(text_input)  # [B, seq_len, C]

        all_image_indices = (text_input == self.image_token_id).to(image_embeds.device)

        assert (text_input[all_image_indices].shape[0] == image_embeds.shape[0] * image_embeds.shape[1]), \
            f"{text_input[text_input == self.image_token_id].shape[0]} != {image_embeds.shape[0]}*{image_embeds.shape[1]}"
        assert (image_embeds.shape[-1] == text_embeds.shape[-1]), f"{image_embeds.shape[-1]} != {text_embeds.shape[-1]}"

        image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])

        text_embeds[all_image_indices] = image_embeds

        regress_label_mask = ((text_input == self.image_token_id) + (text_input == self.img_end_token_id)).to(
            image_embeds.device)

        regress_labels = text_embeds[regress_label_mask]
        regress_mask = ((text_input == self.image_token_id) + (text_input == self.img_token_id)).to(image_embeds.device)

        outputs = self.lm(
            inputs_embeds=text_embeds,
            attention_mask=text_mask,
            return_dict=True,
            labels=targets,
            regress_mask=regress_mask,
            img_length=n_causal,
            args=self.args,
            regress_labels=regress_labels.detach()
            # regress_labels=text_embeds
        )

        return outputs

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.lm.gradient_checkpointing_enable()
