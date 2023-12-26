# -*- coding: utf-8 -*-

# ===========================================================================================
#
#    Copyright (c) Beijing Academy of Artificial Intelligence (BAAI). All rights reserved.
#
#    Author        : Fan Zhang
#    Email         : zhangfan@baai.ac.cn
#    Institute     : Beijing Academy of Artificial Intelligence (BAAI)
#    Create On     : 2023-12-12 03:00
#    Last Modified : 2023-12-25 04:31
#    File Name     : backend.py
#    Description   :
#
# ===========================================================================================

import argparse
import base64
import json
from queue import Queue
import io
import os
import os.path as osp
import time
from PIL import Image
from threading import BoundedSemaphore, Lock
import traceback

import torch
from flask import Flask, request

from emu.diffusion import EmuVisualGeneration
from emu.chat import EmuChatGeneration
from utils import backend_logger as logging

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=9000)
parser.add_argument("--start-card", type=int, default=0)

parser.add_argument("--disable-chat", action="store_true")
parser.add_argument("--chat-concurrency", type=int, default=1)
parser.add_argument("--chat-gpu-per-instance", type=int, default=1)

parser.add_argument("--disable-generate", action="store_true")
parser.add_argument("--generate-concurrency", type=int, default=1)
parser.add_argument("--generate-gpu-per-instance", type=int, default=1)

parser.add_argument("--model-path", type=str, default="./weight")

args = parser.parse_args()

CACHE_DIR = osp.join(osp.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

class Helper:

    def __init__(
        self,
        model_cls,
        weight_path,
        start_card=0,
        concurrency=1,
        gpu_num=1,
        **kwargs,
    ):
        self.queue = Queue()
        for i in range(concurrency):
            device_list = [
                torch.device(f"cuda:{start_card + i * gpu_num + inner_idx}")
                for inner_idx in range(gpu_num)
            ]
            pipeline = model_cls.from_pretrained(
                weight_path,
                **kwargs,
            ).multito(device_list)
            self.queue.put(pipeline)

        self.sem = BoundedSemaphore(concurrency)
        self.lock = Lock()

    def get(self):
        self.sem.acquire()
        self.lock.acquire()
        pipeline = self.queue.get()
        self.lock.release()

        return pipeline

    def put(self, pipeline):
        self.lock.acquire()
        self.queue.put(pipeline)
        self.lock.release()
        self.sem.release()

app = Flask("Emu2")

device = args.start_card
if not args.disable_generate:
    g_helper = Helper(
        EmuVisualGeneration,
        osp.join(args.model_path, "Emu2-Gen_pytorch_model.bf16.safetensors"),
        device,
        args.generate_concurrency,
        args.generate_gpu_per_instance,
        dtype=torch.bfloat16,
        use_safetensors=True,
    )
    device += args.generate_concurrency * args.generate_gpu_per_instance

    @app.route('/v1/mmg', methods=["POST"])
    def multimodal_generation():
        log_id = request.form.get("log_id", "")
        logging.info(f"{log_id}: receive generation request")

        prompt = json.loads(request.form.get("prompt", ""))
        inputs = []
        for t, p in prompt:
            if t == "TEXT":
                inputs.append(p)
            else:
                inputs.append(Image.open(io.BytesIO(request.files.get(p).stream.read())).convert("RGB"))
                inputs[-1].save(osp.join(CACHE_DIR, f"{log_id}-{p}.png"))

        cfg = float(request.form.get("classifier_free_guidance"))
        steps = int(request.form.get("steps"))

        logging.info(f"{log_id}: generate with hyper-parameters, cfg: {cfg}, steps: {steps}")
        logging.info(f"{log_id}: generate prompt: {inputs}")

        pipeline = g_helper.get()
        res = {"code": 0}
        t0 = time.time()
        try:
            image = pipeline(inputs=inputs, guidance_scale=cfg, num_inference_steps=steps).image
            image.save(osp.join(CACHE_DIR, f"{log_id}-[RESULT].png"))
            buf = io.BytesIO()
            image.save(buf, format="WEBP")
            res["data"] = base64.b64encode(buf.getvalue()).decode('ascii')
        except Exception as ex:
            logging.error(f"{log_id}: generate failed, err msg: {str(ex)}")
            logging.error(traceback.format_exc())
            res["code"] = -1
            res["data"] = str(ex)
        t1 = time.time()
        logging.info(f"{log_id}: generate complete with code {res['code']}, time: {(t1-t0)*1000:.3f}ms")

        g_helper.put(pipeline)
        torch.cuda.empty_cache()
        return json.dumps(res)


if not args.disable_chat:
    c_helper = Helper(
        EmuChatGeneration,
        osp.join(args.model_path, "Emu2-Chat_pytorch_model.bf16.pth"),
        device,
        args.chat_concurrency,
        args.chat_gpu_per_instance,
        instruct=True,
        dtype=torch.bfloat16,
        use_safetensors=False,
    )
    device += (args.chat_concurrency * args.chat_gpu_per_instance)

    @app.route('/v1/mmc', methods=["POST"])
    def multimodal_chat():
        log_id = request.form.get("log_id", "")
        logging.info(f"{log_id}: receive chat request")

        prompt = json.loads(request.form.get("prompt", ""))
        inputs = []
        for t, p in prompt:
            if t == "TEXT":
                inputs.append(p)
            else:
                inputs.append(Image.open(io.BytesIO(request.files.get(p).stream.read())).convert("RGB"))
                inputs[-1].save(osp.join(CACHE_DIR, f"{log_id}-{p}.png"))

        do_sample = True if request.form.get("do_sample", 'False').lower() == "true" else False
        max_new_tokens = int(request.form.get("max_new_tokens", 10))
        temperature = float(request.form.get("temperature", 0.7))
        top_k = int(request.form.get("top_k", 3))
        top_p = float(request.form.get("top_p", 0.9))
        length_penalty = float(request.form.get("length_penalty", 1))
        num_beams = int(request.form.get("num_beams", 5))
        repetition_penalty = float(request.form.get("repetition_penalty", 1.))

        logging.info(f"{log_id}: generate with hyper-parameters, "
            f"do_sample: {do_sample}, "
            f"max_new_tokens: {max_new_tokens}, "
            f"temperature: {temperature}, "
            f"top_k: {top_k}, "
            f"top_p: {top_p}, "
            f"length_penalty: {length_penalty}, "
            f"num_beams: {num_beams}, "
            f"repetition_penalty: {repetition_penalty}."
        )
        logging.info(f"{log_id}: chat prompt: {inputs}")

        pipeline = c_helper.get()
        res = {"code": 0}
        t0 = time.time()
        try:
            response = pipeline(
                inputs=inputs,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
            )
            res["data"] = response
        except Exception as ex:
            logging.error(f"{log_id}: chat failed, err msg: {str(ex)}")
            logging.error(traceback.format_exc())
            res["code"] = -1
            res["data"] = str(ex)
        t1 = time.time()
        logging.info(f"{log_id}: chat complete with code {res['code']}, output: {res['data']}, time: {(t1-t0)*1000:.3f}ms")

        c_helper.put(pipeline)
        torch.cuda.empty_cache()
        return json.dumps(res)


app.run(host="0.0.0.0", port=args.port)
