import argparse

import json

import torch
import numpy as np
from PIL import Image
from models.modeling_emu import Emu

image_placeholder = "[IMG]" + "<image>" * 32 + "[/IMG]"


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
    msg = model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f"=====> get model.load_state_dict msg: {msg}")

    return model


def Emu_inference(image_list, text_sequence, instruct=True):
    if len(image_list) == 1 and instruct:
        system = 'You will be presented with an image: [IMG]ImageContent[/IMG]. You will be able to see the image after I provide it to you. Please answer my questions based on the given image.'
    else:
        system = ''

    if instruct:
        prompt = f"{system} [USER]: {text_sequence} [ASSISTANT]:".strip()
    else:
        prompt = text_sequence

    print(f"===> prompt: {prompt}")

    samples = {"image": torch.cat(image_list, dim=0), "prompt": prompt}

    output_text = emu_model.generate(
        samples,
        max_new_tokens=512,
        num_beams=5,
        length_penalty=0.0,
        repetition_penalty=1.0,
    )[0].strip()

    print(f"===> output: {output_text}\n")


def Emu_instruct_caption(img):
    system = "You will be presented with an image: [IMG]ImageContent[/IMG]. You will be able to see the image after I provide it to you. Please answer my questions based on the given image."

    prompt = f"{system} [USER]: {image_placeholder}Please provide an accurate and concise description of the given image. [ASSISTANT]: The image depicts a photo of".strip()

    print(f"===> caption prompt: {prompt}")

    samples = {"image": img, "prompt": prompt}

    output_text = emu_model.generate(
        samples,
        max_new_tokens=512,
        num_beams=5,
        length_penalty=0.0,
        repetition_penalty=1.0,
    )[0].strip()

    print(f"===> caption output: {output_text}\n")


def pretrain_example():
    # prepare in-context learning example
    interleaved_sequence = [
        process_img('examples/dog.png', args.device),
        'There are two dogs.',
        process_img('examples/panda.png', args.device),
        'There are three pandas.',
        process_img('examples/sunflower.png', args.device),
    ]
    text_sequence = ''
    image_list = []
    for item in interleaved_sequence:
        if isinstance(item, str):  # text
            text_sequence += item
        else:  # image
            image_list.append(item)
            text_sequence += image_placeholder

    # Pretrained Model Inference
    # -- in-context learning
    Emu_inference(image_list, text_sequence, instruct=False)


def instruct_example():
    # prepare interleaved image-text data as input
    interleaved_sequence = [
        process_img('examples/book1.jpeg', args.device),
        'This is the first image.',
        process_img('examples/book2.jpeg', args.device),
        'This is the second image.',
        process_img('examples/book3.jpeg', args.device),
        'This is the third image.',
        process_img('examples/book4.jpeg', args.device),
        'This is the fourth image.',
        'Describe all images.'
    ]
    text_sequence = ''
    image_list = []
    for item in interleaved_sequence:
        if isinstance(item, str):  # text
            text_sequence += item
        else:  # image
            image_list.append(item)
            text_sequence += image_placeholder

    # prepare image captioning and vqa data
    image = process_img('examples/iron_man.jpg', args.device)
    question = 'what is the man doing?'

    # Instruct Model Inference
    # -- image captioning
    Emu_instruct_caption(image)
    # -- visual question answering
    Emu_inference([image], image_placeholder + question)
    # -- image-text interleaved input, text output
    Emu_inference(image_list, text_sequence)


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
    return img


if __name__ == '__main__':

    args = parse_args()

    # initialize and load model
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    emu_model = prepare_model('Emu-14B', args)
    emu_model.to(args.device).to(torch.bfloat16)

    if args.instruct:
        instruct_example()
    else:
        pretrain_example()
