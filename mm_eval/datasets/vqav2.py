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
from .vqa_tools.vqa_eval import VQAEval
from .utils import short_answer


class VQAv2Dataset(Dataset):
    split_paths = {
        'val': [
            'coco/annotations/vqa_val_eval.json', 
            'coco/images/',
        ],
        'test': [
            'coco/annotations/vqa_test.json', 
            'coco/images/',
        ],
    }
    
    def __init__(self, root, split='test'):
        self.root = root
        self.split = split
        self.ann_path, self.img_path = self.split_paths[split]
        with open(os.path.join(root, self.ann_path), 'r') as f:
            self.samples = json.load(f)
        
        self.anno_files = self.ques_files = None
        if split == 'val':
            self.anno_files = os.path.join(root, 'coco/annotations/v2_mscoco_val2014_annotations.json')
            self.ques_files = os.path.join(root, 'coco/annotations/v2_OpenEnded_mscoco_val2014_questions.json')
        
        from .. import image_placeholder, image_system_msg
        self.image_placeholder = image_placeholder
        self.image_system_msg = image_system_msg

    def __len__(self):
        return len(self.samples) // 1000
    
    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = os.path.join(self.root, self.img_path, sample["image"])
        image = Image.open(image_path).convert("RGB")
        
        prompt = self.image_system_msg
        prompt += f" [USER]: {self.image_placeholder} based on the content of the image and common sense, please provide an accurate answer consisting of only one word or phrase. {sample['question']} [ASSISTANT]: the answer is:"

        return {"image": image, "instance_id": index, "prompt": prompt}

        
def vqav2_dataloader(root_path, batch_size):
    dataset = VQAv2Dataset(root=root_path)
    print(f"===> VQAv2 num_samples: {len(dataset)}")  # 5000
    
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

def vqav2_results_processor(results, output_dir, samples, anno_files, ques_files):
    # save predictions
    save_result = []
    for res in results:
        sample = samples[res['instance_id']]
        save_result.append({
            "question_id": sample["question_id"],
            "answer": short_answer(res['prediction'])
        })
    result_file = os.path.join(output_dir + "vqav2_answer.json")
    with open(result_file, "w") as f:
        json.dump(save_result, f)
        
    if anno_files is None:
        print('VQAv2-test: Please submit the results file to the official evaluation website.')
        return
    
    vqa = VQA(anno_files, ques_files)
    vqa_result = vqa.loadRes(
        resFile=result_file, quesFile=ques_files,
    )
    vqa_scorer = VQAEval(vqa, vqa_result, n=2)
    vqa_scorer.evaluate()
    print(f"VQAv2 accuracy: {vqa_scorer.accuracy}")
    return vqa_scorer.accuracy
