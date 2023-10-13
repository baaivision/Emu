import base64
import io
import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import os
import json
import torch
import numpy as np
from PIL import Image
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from .vqa_tools.vqa import VQA
from .vqa_tools.vqa_eval import OKVQAEval
from .utils import short_answer


class OKVQADataset(Dataset):
    split_paths = {
        'val': [
            'okvqa/annotations/vqa_val_eval.json', 
            'coco/images/',
        ],
    }
    
    def __init__(self, root, split='val'):
        self.root = root
        self.split = split
        self.ann_path, self.img_path = self.split_paths[split]
        with open(os.path.join(root, self.ann_path), 'r') as f:
            self.samples = json.load(f)
        
        self.anno_files = self.ques_files = None
        self.anno_files = os.path.join(root, 'okvqa/annotations/mscoco_val2014_annotations.json')
        self.ques_files = os.path.join(root, 'okvqa/annotations/OpenEnded_mscoco_val2014_questions.json')
        
        from .. import image_placeholder, image_system_msg
        self.image_placeholder = image_placeholder
        self.image_system_msg = image_system_msg

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = os.path.join(self.root, self.img_path, sample["image"])
        image = Image.open(image_path).convert("RGB")
        
        prompt = self.image_system_msg
        prompt += f" [USER]: {self.image_placeholder} based on the content of the image and common sense, please provide an accurate answer consisting of only one word or phrase. {sample['question']} [ASSISTANT]: the answer is:"

        return {"image": image, "instance_id": index, "prompt": prompt}
        
        
def okvqa_dataloader(root_path, batch_size):
    dataset = OKVQADataset(root=root_path)
    print(f"===> OKVQA num_samples: {len(dataset)}")  # 5000
    
    if dist.is_initialized():
        sampler = DistributedSampler( 
            dataset,
            shuffle=False,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
        )
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        sampler=sampler,
        collate_fn=lambda batch: batch,
        drop_last=False,
    )
    
    inference_kwargs = dict(
        num_beams=5,
        max_new_tokens=20,
        min_length=1,
        length_penalty=-1.0,
        inference_type="generation"
    )
    
    return dataloader, inference_kwargs, {"samples": dataset.samples, "anno_files": dataset.anno_files, "ques_files": dataset.ques_files}



def okvqa_results_processor(results, output_dir, samples, anno_files, ques_files):
    # save predictions
    save_result = []
    for res in results:
        sample = samples[res['instance_id']]
        save_result.append({
            "question_id": sample["question_id"],
            "answer": short_answer(res['prediction'])
        })
    result_file = os.path.join(output_dir + "okvqa_answer.json")
    with open(result_file, "w") as f:
        json.dump(save_result, f)
    
    vqa = VQA(anno_files, ques_files)
    vqa_result = vqa.loadRes(
        resFile=result_file, quesFile=ques_files,
    )

    vqa_scorer = OKVQAEval(vqa, vqa_result, n=2)
    vqa_scorer.evaluate()
    print(f"OKVQA accuracy: {vqa_scorer.accuracy}")
    return vqa_scorer.accuracy
