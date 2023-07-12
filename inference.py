import argparse

import json

import torch
import numpy as np
from PIL import Image
from models.modeling_emu import Emu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instruct",
        action='store_true',
        default=False,
        help="Load Emu-I",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default='',
        help="Emu ckpt path",
    )
    args = parser.parse_args()

    return args


def prepare_model(model_name, args):
    with open(f'models/{model_name}.json', "r", encoding="utf8") as f:
        model_cfg = json.load(f)
    print(f"=====> model_cfg: {model_cfg}")

    model = Emu(**model_cfg, cast_dtype=torch.float, args=args)

    if args.instruct:
        print('Patching LoRA...')
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.decoder.lm = get_peft_model(model.decoder.lm, lora_config)

    print(f"=====> loading from ckpt_path {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    ckpt = ckpt if args.instruct else ckpt['module']
    msg = model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f"=====> get model.load_state_dict msg: {msg}")

    return model


def Emu_instruct_caption(img):
    prompt = "You will be presented with an image: [IMG]ImageContent[/IMG]. You will be able to see the image after I provide it to you. Please answer my questions based on the given image." \
             "[USER]: [IMG]<image><image><image><image><image><image><image><image><image><image>" \
             "<image><image><image><image><image><image><image><image><image><image><image><image>" \
             "<image><image><image><image><image><image><image><image><image><image>[/IMG]" \
             "Please provide an accurate and concise description of the given image. " \
             "[ASSISTANT]: The image depicts a photo of"

    print(f"===> caption prompt: {prompt}")

    samples = {"image": img, "prompt": prompt}

    output_text = emu_model.generate(
        samples,
        max_new_tokens=512,
        num_beams=5,
        length_penalty=0.0,
        repetition_penalty=1.0,
    )[0].strip()

    print(f"===> caption output: {output_text}")


def Emu_instruct_vqa(img, question):
    prompt = "You will be presented with an image: [IMG]ImageContent[/IMG]. You will be able to see the image after I provide it to you. Please answer my questions based on the given image." \
             "[USER]: [IMG]<image><image><image><image><image><image><image><image><image><image>" \
             "<image><image><image><image><image><image><image><image><image><image><image><image>" \
             "<image><image><image><image><image><image><image><image><image><image>[/IMG]" \
             f"{question} " \
             "[ASSISTANT]:"

    print(f"===> vqa prompt: {prompt}")

    samples = {"image": img, "prompt": prompt}

    output_text = emu_model.generate(
        samples,
        max_new_tokens=512,
        num_beams=5,
        length_penalty=0.0,
        repetition_penalty=1.0,
    )[0].strip()

    print(f"===> vqa output: {output_text}")


def process_img(img_path, device):
    width, height = 224, 224
    OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
    img = Image.open(img_path).convert("RGB")
    img = img.resize((width, height))
    img = np.array(img) / 255.
    img = (img - OPENAI_DATASET_MEAN) / OPENAI_DATASET_STD
    img = torch.tensor(img).to(device).to(torch.float)
    img = torch.einsum('hwc->chw', img)
    img = img.unsqueeze(0)

    print(f"===> img.shape: {img.shape}")

    return img


if __name__ == '__main__':

    args = parse_args()

    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    emu_model = prepare_model('Emu-14B', args)
    emu_model.to(args.device).to(torch.bfloat16)

    img_path = 'examples/iron_man.jpg'
    question = 'what is the man doing?'
    img = process_img(img_path, args.device)

    # instruct
    Emu_instruct_caption(img)
    Emu_instruct_vqa(img, question)
