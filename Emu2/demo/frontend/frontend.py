# -*- coding: utf-8 -*-

# ===================================================
#
#    Author        : Fan Zhang
#    Email         : zhangfan@baai.ac.cn
#    Institute     : Beijing Academy of Artificial Intelligence (BAAI)
#    Create On     : 2023-12-11 15:34
#    Last Modified : 2023-12-22 10:48
#    File Name     : frontend.py
#    Description   :
#
# ===================================================

import argparse

import gradio as gr
from libs.generation_frontend import build_generation
from libs.chat_frontend import build_chat

parser = argparse.ArgumentParser()
parser.add_argument("--title", type=str, default='Emu')

parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=9002)
parser.add_argument("--share", action="store_true")
parser.add_argument("--controller-url", type=str, default="http://127.0.0.1:9000")
parser.add_argument("--concurrency-count", type=int, default=8)
parser.add_argument("--disable-chat", action="store_true")
parser.add_argument("--disable-generate", action="store_true")

args = parser.parse_args()


if __name__ == "__main__":
    title = "Emu2: Generative Multimodal Models are In-Context Learners<br> \
            <h2 align='center'> \
            [<a href='https://baaivision.github.io/emu2' target='_blank' rel='noopener'>project page</a>] \
            [<a href='https://github.com/baaivision/Emu' target='_blank' rel='noopener'>code</a>] \
            [<a href='https://arxiv.org/abs/2312.13286' target='_blank' rel='noopener'>paper</a>] \
            </h2> \
            <h3 align='center'> \
            🤗HF models: \
            <a href='https://huggingface.co/BAAI/Emu2' target='_blank' rel='noopener'>Emu2</a> | \
            <a href='https://huggingface.co/BAAI/Emu2-Chat' target='_blank' rel='noopener'>Emu2-Chat</a> | \
            <a href='https://huggingface.co/BAAI/Emu2-Gen' target='_blank' rel='noopener'>Emu2-Gen</a> \
            </h3> \
            <h4 align='center'> \
            [<a href='https://jwolpxeehx.feishu.cn/docx/KskPdU99FomufKx4G9hcQMeQnHv' target='_blank' rel='noopener'>使用说明</a>] \
            [<a href='https://jwolpxeehx.feishu.cn/docx/RYHNd1tvEo8k8Mx9HeMcvvxWnvZ' target='_blank' rel='noopener'>User Guide</a>] \
            </h4> \
            "

    interface_list, tab_names = [], []
    if not args.disable_chat:
        demo_chat = build_chat(args)
        interface_list.append(demo_chat)
        tab_names.append("Multimodal Chat")

    if not args.disable_generate:
        demo_generation = build_generation(args)
        interface_list.append(demo_generation)
        tab_names.append("Multimodal Generation")

    demo_all = gr.TabbedInterface(
        interface_list=interface_list,
        tab_names=tab_names,
        title=title,
        theme=gr.themes.Default(primary_hue="blue", secondary_hue="blue"),
    )

    demo_all.queue(
        concurrency_count=args.concurrency_count,
        status_update_rate=3,
        api_open=False,
    ).launch(
        enable_queue=True,
        server_name=args.host, server_port=args.port,
        share=args.share,
    )
