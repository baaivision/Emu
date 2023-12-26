# -*- coding: utf-8 -*-

# ===================================================
#
#    Author        : Fan Zhang
#    Email         : zhangfan@baai.ac.cn
#    Institute     : Beijing Academy of Artificial Intelligence (BAAI)
#    Create On     : 2023-12-12 18:05
#    Last Modified : 2023-12-22 10:37
#    File Name     : chat_frontend.py
#    Description   :
#
# ===================================================

import json
import io
import time
from PIL import Image
import requests

import gradio as gr

from .meta import ConvMeta, Role, DataMeta
from .utils import extract_frames
from .utils import frontend_logger as logging
from .constants import TERM_OF_USE, CHAT_GUIDANCE, RECOMMEND

CONTROLLER_URL = ""

def submit(
    meta,
    image,
    video,
    text,
    num_frames,
):
    if meta is None:
        meta = ConvMeta()

    meta.pop_error()

    check_text = (text != "" and text is not None)
    check_image = image is not None
    check_video = video is not None

    if check_text + check_image + check_video != 1:
        logging.info(f"{meta.log_id}: invalid input: give multi madality simultaneously for single modality input")
        meta.append(Role.ASSISTANT, DataMeta.build(text=f"Generate Failed: Invalid input number, must give exactly one modality input at a time", is_error=True))
        return meta.format_chatbot(), meta, None, None, ""

    if check_text:
        meta.append(Role.USER, DataMeta.build(text=text))
    elif check_image:
        meta.append(Role.USER, DataMeta.build(image=image))
    elif check_video:
        frames = extract_frames(video, num_frames)
        meta.append(Role.USER, DataMeta.build(frames=frames))

    return meta.format_chatbot(), meta, None, None, ""


def clear_history(meta):
    if meta is None:
        meta = ConvMeta()
    meta.clear()
    return meta.format_chatbot(), meta


def generate(
    meta,
    do_sample,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    length_penalty,
    num_beams,
    repetition_penalty,
):
    if meta is None:
        meta = ConvMeta()

    meta.pop_error()
    meta.pop()

    if len(meta) == 0:
        meta.append(Role.ASSISTANT, DataMeta.build(text=f"Generate Failed: Please enter a valid input", is_error=True))
        return meta.format_chatbot(), meta

    prompt = meta.format_chat()

    prompt_list, image_list = [], {}
    for idx, p in enumerate(prompt):
        if isinstance(p, Image.Image):
            key = f"[<IMAGE{idx}>]"
            prompt_list.append(["IMAGE", key])

            buf = io.BytesIO()
            p.save(buf, format="PNG")
            image_list[key] = (key, io.BytesIO(buf.getvalue()), "image/png")
        else:
            prompt_list.append(["TEXT", p])

    if len(image_list) == 0:
        image_list = None

    logging.info(f"{meta.log_id}: construct chat reqeust with prompt {prompt_list}")

    t0 = time.time()
    try:
        rsp = requests.post(
            CONTROLLER_URL + "/v1/mmc",
            files=image_list,
            data={
                "log_id": meta.log_id,
                "prompt": json.dumps(prompt_list),
                "do_sample": do_sample,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "length_penalty": length_penalty,
                "num_beams": num_beams,
                "repetition_penalty": repetition_penalty,
            },
        )
    except Exception as ex:
        rsp = requests.Response()
        rsp.status_code = 1099
        rsp._content = str(ex).encode()
    t1 = time.time()

    logging.info(f"{meta.log_id}: get response with status code: {rsp.status_code}, time: {(t1-t0)*1000:.3f}ms")

    if rsp.status_code == requests.codes.ok:
        content = json.loads(rsp.text)
        if content["code"] == 0:
            meta.append(Role.ASSISTANT, DataMeta.build(text=content["data"]))
        else:
            meta.append(Role.ASSISTANT, DataMeta.build(text=f"Generate Failed: {content['data']}", is_error=True))
    else:
        meta.append(Role.ASSISTANT, DataMeta.build(text=f"Generate Failed: http failed with code {rsp.status_code}, msg: {rsp.text}", is_error=True))

    return meta.format_chatbot(), meta


def push_examples(examples, meta):
    if meta is None:
        meta = ConvMeta()

    meta.clear()

    image, prompt = examples
    meta.append(Role.USER, DataMeta.build(image=Image.open(image)))
    meta.append(Role.USER, DataMeta.build(text=prompt))

    return meta.format_chatbot(), meta


def build_chat(args):
    global CONTROLLER_URL
    CONTROLLER_URL = args.controller_url

    with gr.Blocks(title="Emu", theme=gr.themes.Default(primary_hue="blue", secondary_hue="blue")) as demo:
        state = gr.State()

        gr.Markdown(CHAT_GUIDANCE)
        gr.Markdown(RECOMMEND)

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    imagebox = gr.Image(type="pil")
                with gr.Row():
                    videobox = gr.Video()

                with gr.Accordion("Parameters", open=True, visible=True) as parameter_row:
                    do_sample = gr.Checkbox(value=False, label="Do Sample", interactive=True)
                    max_new_tokens = gr.Slider(minimum=0, maximum=2048, value=512, step=1, interactive=True, label="Max Output Tokens")
                    temperature = gr.Slider(minimum=0, maximum=1, value=0.7, step=0.05, interactive=True, label="Temperature")
                    top_k = gr.Slider(minimum=1, maximum=5, value=3, step=1, interactive=True, label="Top K")
                    top_p = gr.Slider(minimum=0, maximum=1, value=0.9, step=0.05, interactive=True, label="Top P")
                    length_penalty = gr.Slider(minimum=0, maximum=5, value=2, step=0.1, interactive=True, label="Length Penalty")
                    num_beams = gr.Slider(minimum=1, maximum=10, value=5, step=1, interactive=True, label="Beam Size")
                    repetition_penalty = gr.Slider(minimum=1.0, maximum=10.0, value=1.0, step=0.5, interactive=True, label="Repetition Penalty")
                    num_frames = gr.Number(interactive=True, value=8, maximum=12, label="Num Video Frames")

            with gr.Column(scale=6):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="Emu Chatbot",
                    visible=True,
                    height=1070,
                )

                with gr.Row():
                    with gr.Column(scale=8):
                        textbox = gr.Textbox(
                            show_label=False,
                            placeholder="Enter text and add to prompt",
                            visible=True,
                            container=False,
                        )

                    with gr.Column(scale=1, min_width=60):
                        add_btn = gr.Button(value="Add")

                with gr.Row(visible=True) as button_row:
                    # upvote_btn = gr.Button(value="👍 Upvote", interactive=False)
                    # downvote_btn = gr.Button(value="👎 Downvote", interactive=False)
                    # flag_btn = gr.Button(value="⚠️ Flag", interactive=False)
                    # regenerate_btn = gr.Button(value="🔄 Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️ Clear History")
                    generate_btn = gr.Button(value="Generate")

        with gr.Row():
            examples = gr.Dataset(components=[gr.Image(type="pil", visible=False), gr.Textbox(visible=False)],
                label="Examples",
                samples=[
                    ["./examples/squirrel.jpeg", "What is funny about this image?"],
                    ["./examples/shapes.jpeg", "Look at this sequence of three shapes. What shape should come as the fourth shape? Explain your reasoning with detailed descriptions of the first shapes."],
                ],
            )

        gr.Markdown(TERM_OF_USE)

        clear_btn.click(clear_history, inputs=state, outputs=[chatbot, state])
        textbox.submit(
            submit,
            inputs=[
                state,
                imagebox,
                videobox,
                textbox,
                num_frames,
            ],
            outputs=[
                chatbot,
                state,
                imagebox,
                videobox,
                textbox,
            ],
        )
        add_btn.click(
            submit,
            inputs=[
                state,
                imagebox,
                videobox,
                textbox,
                num_frames,
            ],
            outputs=[
                chatbot,
                state,
                imagebox,
                videobox,
                textbox,
            ],
        )
        generate_btn.click(
            generate,
            inputs=[
                state,
                do_sample,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                length_penalty,
                num_beams,
                repetition_penalty,
            ],
            outputs=[
                chatbot,
                state,
            ],
        )
        examples.click(
            push_examples,
            inputs=[
                examples,
                state,
            ],
            outputs=[
                chatbot,
                state,
            ]
        )

    return demo
