# -*- coding: utf-8 -*-

# ===================================================
#
#    Author        : Fan Zhang
#    Email         : zhangfan@baai.ac.cn
#    Institute     : Beijing Academy of Artificial Intelligence (BAAI)
#    Create On     : 2023-12-11 15:35
#    Last Modified : 2023-12-23 11:03
#    File Name     : generation_frontend.py
#    Description   :
#
# ===================================================

import base64
import json
import io
import time
from PIL import Image
import requests

import gradio as gr

from .emu_constants import EVA_IMAGE_SIZE
from .meta import ConvMeta, Role, DataMeta
from .utils import frontend_logger as logging
from .constants import TERM_OF_USE, GEN_GUIDANCE, RECOMMEND

CONTROLLER_URL = ""

def submit(
    meta,
    enable_grd,
    left,
    top,
    right,
    bottom,
    image,
    text,
):
    if meta is None:
        meta = ConvMeta()

    meta.pop_error()
    if meta.has_gen:
        meta.clear()

    if enable_grd:
        if text == "" and image is None:
            logging.info(f"{meta.log_id}: invalid input: no valid data for grounding input")
            meta.append(Role.ASSISTANT, DataMeta.build(text=f"Input Error: Text or image must be given if enable grounding generation", is_error=True))
            return meta.format_chatbot(), meta, False, 0, 0, EVA_IMAGE_SIZE, EVA_IMAGE_SIZE, None, ""

        meta.append(Role.USER, DataMeta.build(text=text, image=image, coordinate=[left, top, right, bottom]))
    elif image is not None and text != "":
        logging.info(f"{meta.log_id}: invalid input: give text and image simultaneously for single modality input")
        meta.append(Role.ASSISTANT, DataMeta.build(text=f"Input Error: Do not submit text and image data at the same time!!!", is_error=True))
        return meta.format_chatbot(), meta, False, 0, 0, EVA_IMAGE_SIZE, EVA_IMAGE_SIZE, None, ""
    elif image is not None:
        meta.append(Role.USER, DataMeta.build(image=image))
    elif text != "":
        meta.append(Role.USER, DataMeta.build(text=text))
    return meta.format_chatbot(), meta, False, 0, 0, EVA_IMAGE_SIZE, EVA_IMAGE_SIZE, None, ""


def clear_history(meta):
    if meta is None:
        meta = ConvMeta()
    meta.clear()
    return meta.format_chatbot(), meta


def generate(meta, classifier_free_guidance, steps):
    if meta is None:
        meta = ConvMeta()

    meta.pop_error()
    meta.pop()

    if len(meta) == 0:
        meta.append(Role.ASSISTANT, DataMeta.build(text=f"Generate Failed: Please enter a valid input", is_error=True))
        return meta.format_chatbot(), meta

    prompt = meta.format_prompt()

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

    logging.info(f"{meta.log_id}: construct generation reqeust with prompt {prompt_list}")

    t0 = time.time()
    try:
        rsp = requests.post(
            CONTROLLER_URL + "/v1/mmg",
            files=image_list,
            data={
                "log_id": meta.log_id,
                "prompt": json.dumps(prompt_list),
                "classifier_free_guidance": classifier_free_guidance,
                "steps": steps,
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
            image = Image.open(io.BytesIO(base64.b64decode(content["data"])))
            meta.append(Role.ASSISTANT, DataMeta.build(image=image, resize=False))
        else:
            meta.append(Role.ASSISTANT, DataMeta.build(text=f"Generate Failed: {content['data']}", is_error=True))
    else:
        meta.append(Role.ASSISTANT, DataMeta.build(text=f"Generate Failed: http failed with code {rsp.status_code}, msg: {rsp.text}", is_error=True))

    return meta.format_chatbot(), meta

def push_examples(examples, meta):
    if meta is None:
        meta = ConvMeta()

    meta.clear()

    if len(examples) == 1:
        prompt, = examples
        meta.append(Role.USER, DataMeta.build(text=prompt))
    elif len(examples) == 3:
        p1, image, p2 = examples
        if p1 is not None and p1 != "":
            meta.append(Role.USER, DataMeta.build(text=p1))
        meta.append(Role.USER, DataMeta.build(image=Image.open(image)))
        if p2 is not None and p2 != "":
            meta.append(Role.USER, DataMeta.build(text=p2))

    return meta.format_chatbot(), meta


def build_generation(args):
    global CONTROLLER_URL
    CONTROLLER_URL = args.controller_url

    with gr.Blocks(title="Emu", theme=gr.themes.Default(primary_hue="blue", secondary_hue="blue")) as demo:
        state = gr.State()

        gr.Markdown("<font size=5><center><b>This demo</b> can accept a mix of <b><u>_texts_</u></b>, <b><u>_locations_</u></b> and <b><u>_images_</u></b> as input, and generating images in context</center></font>")
        gr.Markdown(GEN_GUIDANCE)
        gr.Markdown(RECOMMEND)
        gr.Markdown("<font size=4>💡<b><u>Tips</b></u>💡:</font> To achieve better generation quality\n \
                       - If subject-driven generation does not follow the given prompt, randomly bind a central bounding box with the input image or text often helps resolve the issue.\n \
                       - In multi-object generation, it is recommended to specify location and object names(phrase) for better results.\n \
                       - The best results are achieved when the aspect ratio of the location box matches that of the original object.")

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    imagebox = gr.Image(type="pil")

                with gr.Row():
                    with gr.Accordion("Grounding Parameters", open=True, visible=True) as grounding_row:
                        enable_grd = gr.Checkbox(label="Enable")
                        left = gr.Slider(minimum=0, maximum=EVA_IMAGE_SIZE, value=0, step=1, interactive=True, label="left")
                        top = gr.Slider(minimum=0, maximum=EVA_IMAGE_SIZE, value=0, step=1, interactive=True, label="top")
                        right = gr.Slider(minimum=0, maximum=EVA_IMAGE_SIZE, value=EVA_IMAGE_SIZE, step=1, interactive=True, label="right")
                        bottom = gr.Slider(minimum=0, maximum=EVA_IMAGE_SIZE, value=EVA_IMAGE_SIZE, step=1, interactive=True, label="bottom")

                with gr.Row():
                    with gr.Accordion("Diffusion Parameters", open=True, visible=True) as parameters_row:
                        cfg = gr.Slider(minimum=1, maximum=30, value=3, step=0.5, interactive=True, label="classifier free guidance")
                        steps = gr.Slider(minimum=1, maximum=100, value=50, step=1, interactive=True, label="steps")

            with gr.Column(scale=6):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="Emu Chatbot",
                    visible=True,
                    height=720,
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
                    # regenerate_btn = gr.Button(value="🔄 Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️ Clear History")
                    generate_btn = gr.Button(value="Generate")

        with gr.Row():
            examples_t2i = gr.Dataset(components=[gr.Textbox(visible=False)],
                label="Text-to-image Examples",
                samples=[
                    ["impressionist painting of an astronaut in a jungle"],
                    ["A poised woman with short, curly hair and a warm smile, dressed in elegant attire, standing in front of a historic stone bridge in a serene park at sunset."],
                ],
            )
        with gr.Row():
            examples_it2i = gr.Dataset(components=[gr.Textbox(visible=False), gr.Image(type="pil", visible=False), gr.Textbox(visible=False)],
                label="Image Editing Examples",
                samples=[
                    ["", "./examples/dog.jpg", "make it oil painting style."],
                    ["An image of", "./examples/emu.png", "wearing a big sunglasses on the beach"]
                ],
            )


        gr.Markdown(TERM_OF_USE)

        clear_btn.click(clear_history, inputs=state, outputs=[chatbot, state])

        textbox.submit(
            submit,
            inputs=[
                state,
                enable_grd,
                left,
                top,
                right,
                bottom,
                imagebox,
                textbox,
            ],
            outputs=[
                chatbot,
                state,
                enable_grd,
                left,
                top,
                right,
                bottom,
                imagebox,
                textbox,
            ],
        )

        add_btn.click(
            submit,
            inputs=[
                state,
                enable_grd,
                left,
                top,
                right,
                bottom,
                imagebox,
                textbox,
            ],
            outputs=[
                chatbot,
                state,
                enable_grd,
                left,
                top,
                right,
                bottom,
                imagebox,
                textbox,
            ],
        )

        generate_btn.click(
            generate,
            inputs=[
                state,
                cfg,
                steps,
            ],
            outputs=[
                chatbot,
                state,
            ]
        )

        examples_t2i.click(
            push_examples,
            inputs=[
                examples_t2i,
                state,
            ],
            outputs=[
                chatbot,
                state,
            ]
        )
        examples_it2i.click(
            push_examples,
            inputs=[
                examples_it2i,
                state,
            ],
            outputs=[
                chatbot,
                state,
            ]
        )


    return demo
