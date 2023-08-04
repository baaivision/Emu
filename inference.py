import argparse

import json

import torch
from models.modeling_emu import Emu
from utils import process_img, process_video

image_placeholder = "[IMG]" + "<image>" * 32 + "[/IMG]"
image_system_msg = "You will be presented with an image: [IMG]ImageContent[/IMG]. You will be able to see the image after I provide it to you. Please answer my questions based on the given image."
video_system_msg = "You are a helpful assistant and you will be presented with a video consisting of multiple chronological images: [IMG]ImageContent[/IMG]. You will be able to see the video after I provide it to you. Please answer my questions based on the given video."


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


def Emu_inference(image_list, text_sequence, system='', instruct=True, max_new_tokens=128, beam_size=5, length_penalty=0.0):
    if instruct:
        prompt = f"{system} [USER]: {text_sequence} [ASSISTANT]:".strip()
    else:
        prompt = text_sequence

    print(f"===> prompt: {prompt}")

    samples = {"image": torch.cat(image_list, dim=0), "prompt": prompt}

    output_text = emu_model.generate(
        samples,
        max_new_tokens=max_new_tokens,
        num_beams=beam_size,
        length_penalty=length_penalty,
        repetition_penalty=1.0,
    )[0].strip()

    print(f"===> output: {output_text}\n")


def Emu_instruct_caption(img):
    system = image_system_msg

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
    image_text_sequence = [
        process_img(img_path='examples/dog.png', device=args.device),
        'There are two dogs.',
        process_img(img_path='examples/panda.png', device=args.device),
        'There are three pandas.',
        process_img(img_path='examples/sunflower.png', device=args.device),
    ]
    interleaved_sequence_1 = ''
    image_list_1 = []
    for item in image_text_sequence:
        if isinstance(item, str):  # text
            interleaved_sequence_1 += item
        else:  # image
            image_list_1.append(item)
            interleaved_sequence_1 += image_placeholder

    # Pretrained Model Inference
    # -- in-context learning
    Emu_inference(image_list_1, interleaved_sequence_1, instruct=False)


def instruct_example():
    # prepare image captioning and vqa examples
    image = process_img(img_path='examples/iron_man.jpg', device=args.device)
    question = 'what is the man doing?'

    # prepare interleaved image-text input example
    image_text_sequence = [
        process_img(img_path='examples/book1.jpeg', device=args.device),
        'This is the first image.',
        process_img(img_path='examples/book2.jpeg', device=args.device),
        'This is the second image.',
        process_img(img_path='examples/book3.jpeg', device=args.device),
        'This is the third image.',
        process_img(img_path='examples/book4.jpeg', device=args.device),
        'This is the fourth image.',
        'Describe all images.'
    ]
    interleaved_sequence_1 = ''
    image_list_1 = []
    for item in image_text_sequence:
        if isinstance(item, str):  # text
            interleaved_sequence_1 += item
        else:  # image
            image_list_1.append(item)
            interleaved_sequence_1 += image_placeholder

    # prepare video example
    image_list_2, interleaved_sequence_2 = process_video('examples/AppleVR.mp4')
    interleaved_sequence_2 += "What's the woman doing in the video?"

    # Instruct Model Inference
    # -- image captioning
    Emu_instruct_caption(image)
    # -- visual question answering
    Emu_inference([image], image_placeholder + question, system=image_system_msg)
    # -- image-text interleaved input, text output
    Emu_inference(image_list_1, interleaved_sequence_1, system='')
    # -- video understanding
    Emu_inference(image_list_2, interleaved_sequence_2, system=video_system_msg, length_penalty=1.0)


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
