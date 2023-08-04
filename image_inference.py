# -*- coding: utf-8 -*-

import argparse

from PIL import Image
from models.pipeline import EmuGenerationPipeline


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
        help="Emu Decoder ckpt path",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    # NOTE
    # Emu Decoder Pipeline only supports pretrain model
    # Using instruct tuning model as image encoder may cause unpredicted results
    assert args.instruct is False, "Image Generation currently do not support instruct tuning model"

    pipeline = EmuGenerationPipeline.from_pretrained(
        path=args.ckpt_path,
        args=args,
    )
    pipeline = pipeline.bfloat16().cuda()

    # image blend case
    # image_1 = Image.open("examples/sunflower.png")
    # image_2 = Image.open("examples/oil_sunflower.jpg")
    image_1 = Image.open("examples/cat.jpg")
    image_2 = Image.open("examples/tiger.jpg")
    image, safety = pipeline(
        [image_1, image_2],
        height=512,
        width=512,
        guidance_scale=7.5,
    )

    if safety is None or not safety:
        image.save("image_blend_result.jpg")
    else:
        print("ImageBlend Generated Image Has Safety Concern!!!")

    # text-to-image case
    text = "An image of a dog wearing a pair of glasses."
    image, safety = pipeline(
        [text],
        height=512,
        width=512,
        guidance_scale=7.5,
    )

    if safety is None or not safety:
        image.save("text2image_result.jpg")
    else:
        print("T2I Generated Image Has Safety Concern!!!")

    # in-context generation
    image_1 = Image.open("examples/dog.png")
    image_2 = Image.open("examples/sunflower.png")

    image, safety = pipeline(
        [
            "This is the first image: ",
            image_1,
            "This is the second image: ",
            image_2,
            "The animal in the first image surrounded with the plant in the second image: ",
        ],
        height=512,
        width=512,
        guidance_scale=10.,
    )

    if safety is None or not safety:
        image.save("incontext_result.jpg")
    else:
        print("In-context Generated Image Has Safety Concern!!!")
