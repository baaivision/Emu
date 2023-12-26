"""
LLaMA model from transformers, following stanford's Alpaca
"""
import torch.nn as nn

import transformers
from transformers.models.llama.configuration_llama import LlamaConfig

from .constants import *

        
def add_location_symbols(quantized_size=256, locate_special_token=2, flag_rec_symbol=True):
    custom_sp_symbols = []

    if locate_special_token > 0:
        custom_sp_symbols.append(GRD_SYMBOL)
    
    for symbol in [BOP_SYMBOL, EOP_SYMBOL, BOO_SYMBOL, EOO_SYMBOL, DOM_SYMBOL]:
        custom_sp_symbols.append(symbol)
    
    if flag_rec_symbol:
        custom_sp_symbols.append(REC_SYMBOL)

    for i in range(quantized_size+1):
        token_name = f"<patch_index_{str(i).zfill(4)}>"
        custom_sp_symbols.append(token_name)
    return custom_sp_symbols


class EmuForClsAndRegression(nn.Module):

    def __init__(self, args):
        super(EmuForClsAndRegression, self).__init__()
        self.args = args

        # init a empty lm
        llama_config = LlamaConfig.from_pretrained(args.llama_config_path)
        self.lm = transformers.LlamaForCausalLM(config=llama_config)

        # init tokenizer
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(args.llama_config_path)

        special_tokens_list = [
            DEFAULT_IMG_TOKEN, 
            DEFAULT_IMG_END_TOKEN, 
            DEFAULT_IMAGE_TOKEN, 
            DEFAULT_gIMG_TOKEN,
            DEFAULT_gIMG_END_TOKEN,
            DEFAULT_EOC_TOKEN,
            DEFAULT_VIDEO_TOKEN
        ] + add_location_symbols()

        if args.instruct:
            special_tokens_list += [USER_TOKEN, ASSISTANT_TOKEN]

        special_tokens_dict = dict(
            pad_token=DEFAULT_PAD_TOKEN,
            bos_token=DEFAULT_BOS_TOKEN,
            eos_token=DEFAULT_EOS_TOKEN,
            additional_special_tokens=special_tokens_list
        )

        self.num_new_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.lm.resize_token_embeddings(len(self.tokenizer))
        self.lm.model.embed_tokens.padding_idx = self.tokenizer.pad_token_id

        self.config = self.lm.config
        self.lm.config.d_model = self.lm.config.hidden_size

        self.image_token_id = self.tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_TOKEN])[0]
        self.img_token_id = self.tokenizer.convert_tokens_to_ids([DEFAULT_IMG_TOKEN])[0]
        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids([DEFAULT_IMG_END_TOKEN])[0]
        self.gimg_token_id = self.tokenizer.convert_tokens_to_ids([DEFAULT_gIMG_TOKEN])[0]
        self.gimg_end_token_id = self.tokenizer.convert_tokens_to_ids([DEFAULT_gIMG_END_TOKEN])[0]
        self.eoc_token_id = self.tokenizer.convert_tokens_to_ids([DEFAULT_EOC_TOKEN])[0]

        self.grounding_token_id = self.tokenizer.convert_tokens_to_ids([GRD_SYMBOL])[0]
        self.rec_token_id = self.tokenizer.convert_tokens_to_ids([REC_SYMBOL])[0]

        self.user_token_id = self.assistant_token_id = None
        if args.instruct:
            self.user_token_id = self.tokenizer.convert_tokens_to_ids([USER_TOKEN])[0]
            self.assistant_token_id = self.tokenizer.convert_tokens_to_ids([ASSISTANT_TOKEN])[0]


        print(f"Vocab Size: {len(self.tokenizer)}")
        print(f"The Special Tokens: {self.tokenizer.special_tokens_map}")
        print(f"bos_token_id: {self.tokenizer.bos_token_id}")
        print(f"eos_token_id: {self.tokenizer.eos_token_id}")
        print(f"pad_token_id: {self.tokenizer.pad_token_id}")
        print(f"{DEFAULT_IMAGE_TOKEN} token id: {self.image_token_id}")
        print(f"{DEFAULT_IMG_TOKEN} token id: {self.img_token_id}")
        print(f"{DEFAULT_IMG_END_TOKEN} token id: {self.img_end_token_id}")
        print(f"{DEFAULT_gIMG_TOKEN} token id: {self.gimg_token_id}")
        print(f"{DEFAULT_gIMG_END_TOKEN} token id: {self.gimg_end_token_id}")
        print(f"{DEFAULT_EOC_TOKEN} token id: {self.eoc_token_id}")
        print(f"{GRD_SYMBOL} token id: {self.grounding_token_id}")
        print(f"{REC_SYMBOL} token id: {self.rec_token_id}")

        if args.instruct:
            print(f"{USER_TOKEN} token id: {self.user_token_id}")
            print(f"{ASSISTANT_TOKEN} token id: {self.assistant_token_id}")

    def get_num_layers(self):
        return len(self.lm.model.layers)

