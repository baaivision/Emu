import os
import json
from PIL import Image
from torch.utils.data import Dataset

import io

import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class VisDialDataset(Dataset):
    split_paths = {
        'val': [
            'visdial/visdial_1.0_val.json', 
            'visdial/VisualDialog_val2018',
        ],
        'test': [
            'visdial/visdial_1.0_test.json', 
            'visdial/VisualDialog_test2018',
        ],
    }

    def __init__(self, root, split='val'):
        self.root = root
        self.split = split
        self.ann_path, self.img_path = self.split_paths[split]
        with open(os.path.join(root, self.ann_path), 'r') as f:
            data_dict = json.load(f)
        answers = data_dict['data']['answers']  # 34821, list
        questions = data_dict['data']['questions']  # 45237, list

        self.samples = []
        for dialog_id, dialog in enumerate(data_dict['data']['dialogs']):  # 2064, list
            history = []
            for round_id, round_qa in enumerate(dialog['dialog']):  # 10 round, list
                question = questions[round_qa['question']]  # int
                options = [
                    answers[answer_id] 
                    for answer_id in round_qa['answer_options']  # list of int
                ]
                gt_index = round_qa['gt_index']
                answer = options[gt_index]
                self.samples.append({
                    "image_id": dialog["image_id"],
                    "dialog_id": dialog_id,
                    "round_id": round_id,
                    "question": question,
                    "options": options,
                    "gt_index": gt_index,
                    "answer": answer,
                    "history": history,
                })
                history.append((question, answer))
        
        from .. import image_placeholder, image_system_msg
        self.image_placeholder = image_placeholder
        self.image_system_msg = image_system_msg

    def __len__(self):
        return len(self.samples)  # 20640
    
    def __getitem__(self, index):
        sample = self.samples[index]

        image_path = os.path.join(
            self.root, self.img_path, 
            os.path.basename(self.img_path) + '_{:0>12d}.jpg'.format(sample["image_id"]),
        )
        image = Image.open(image_path).convert("RGB")
        
        prompt = self.image_system_msg
        prompt += f" [USER]: {self.image_placeholder}</s>"
        for i in range(sample["round_id"]):
            question, answer = sample["history"][i]
            prompt += f" [USER]: {question}? [ASSISTANT]: {answer}.</s>"
        question = sample["question"]
        prompt += f" [USER]: {question}? [ASSISTANT]:"
        
        return {"image": image, "prompt": prompt, "options": sample["options"], "instance_id": index}


def visdial_dataloader(root_path, batch_size):
    dataset = VisDialDataset(root=root_path)
    print(f"===> VisDial num_samples: {len(dataset)}")  #
    
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
        n_segments=100,
        inference_type="classification"
    )
    
    return dataloader, inference_kwargs, {"samples": dataset.samples}


def visdial_results_processor(results, output_dir, samples):
    # save predictions
    save_result = []
    for res in results:
        sample = samples[res['instance_id']]
        save_result.append({
            "image_id": sample["image_id"],
            "round_id": sample["round_id"] + 1,
            "ranks": (torch.sort(res['prediction']).indices + 1).tolist(),
            "gt_index": sample["gt_index"], 
        })
    result_file = os.path.join(output_dir + "visdial_answer.json")
    with open(result_file, "w") as f:
        json.dump(save_result, f)
    print(f"visdial answer save to {result_file}, please submit to the official evaluation server.")
