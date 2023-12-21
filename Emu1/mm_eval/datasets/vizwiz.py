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

from .utils import short_answer

def get_split(name):
    if 'train' in name:
        return 'train'
    if 'val' in name:
        return 'val'
    if 'test' in name:
        return 'test'

class VizwizDataset(Dataset):
    split_paths = {
        'val': [
            'vizwiz/annotations/val.json', 
            'vizwiz/images/',
        ],
        'test': [
            'vizwiz/annotations/test.json', 
            'vizwiz/images/',
        ],
    }
    
    def __init__(self, root, split='test'):
        self.root = root
        self.split = split
        self.ann_path, self.img_path = self.split_paths[split]
        with open(os.path.join(root, self.ann_path), 'r') as f:
            self.samples = json.load(f)
        
        from .. import image_placeholder, image_system_msg
        self.image_placeholder = image_placeholder
        self.image_system_msg = image_system_msg

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        image_id = sample["image"]
        image_path = os.path.join(
            self.root, self.img_path, get_split(image_id), image_id,
        )
        image = Image.open(image_path).convert("RGB")
        
        prompt = self.image_system_msg
        prompt += f" [USER]: {self.image_placeholder} based on the content of the image and common sense, please provide an accurate answer consisting of only one word or phrase. {sample['question']} [ASSISTANT]: the answer is:"
        answerability_prompt = self.image_system_msg
        answerability_prompt += f" [USER]: {self.image_placeholder} based on the content of the image and common sense, please provide an accurate answer consisting of only one word or phrase. {sample['question']}, is the answer known? [ASSISTANT]:"

        return {"image": image, "instance_id": index, "prompt": prompt, "answerability_prompt": answerability_prompt}
        
        
def vizwiz_dataloader(root_path, batch_size):
    dataset = VizwizDataset(root=root_path)
    print(f"===> VizWiz num_samples: {len(dataset)}")  # 5000
    
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
        inference_type="vizwiz"
    )
    
    return dataloader, inference_kwargs, {"samples": dataset.samples}

def vizwiz_results_processor(results, output_dir, samples):
    # save predictions
    save_result = []
    for res in results:
        sample = samples[res['instance_id']]
        answerability, prediction = res['prediction']
        answer = 'unanswerable' if answerability == 'no.' else \
                short_answer(prediction)
        save_result.append({
            "image": sample["image"],
            "answer": answer
        })
    result_file = os.path.join(output_dir + "vizwiz_answer.json")
    with open(result_file, "w") as f:
        json.dump(save_result, f)
        
    print('VizWiz-test: Please submit the results file to the official evaluation website.')
    