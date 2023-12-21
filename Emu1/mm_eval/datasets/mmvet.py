import os
import json
import torch
import numpy as np
from PIL import Image
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

image_path = "$YOUR_PATH/mm-vet/images"
ann_path = "$YOUR_PATH/mm-vet/mm-vet.json"

class MMVetDataset(Dataset):
    def __init__(self, image_path, ann_path):
        self.image_path = image_path

        self.annotation = []
        self.read_questions(ann_path)

        self._add_index()
        
        from .. import image_placeholder, image_system_msg
        self.image_placeholder = image_placeholder
        self.image_system_msg = image_system_msg
        
    def _add_index(self, key="index"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)
    
    def read_questions(self, ann_path):
        with open(ann_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        for name, sample in samples.items():
            sample.update({'instance_id': name})
            self.annotation.append(sample)

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, item):
        assert 0 <= item < len(self)
        sample = self.annotation[item]
        image_path = os.path.join(self.image_path, sample["imagename"])
        image = Image.open(image_path).convert('RGB')
        
        question = self.image_placeholder + sample["question"]
        prompt = self.image_system_msg
        prompt += f" [USER]: {question} [ASSISTANT]:"
        

        return {
            "image": image,
            "prompt": prompt,
            "instance_id": sample['instance_id'],
        }

def mmvet_dataloader(batch_size):
    dataset = MMVetDataset(image_path=image_path, ann_path=ann_path)
    print(f"===> MMVeT num_samples: {len(dataset)}")  # 200
    
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
        max_new_tokens=128,
        min_length=1,
        length_penalty=1.0,
        inference_type="generation"
    )
    
    return dataloader, inference_kwargs, {}


def mmvet_results_processor(results, output_dir):
    save_result = {}
    for res in results:
        save_result.update({res['instance_id']: res['prediction']})
    result_file = os.path.join(output_dir + "mmvet_answer.json")

    with open(result_file, "w") as f:
        json.dump(save_result, f)

    print(f"MM-Vet: Saved results for leaderboard evaluation at {result_file}")
