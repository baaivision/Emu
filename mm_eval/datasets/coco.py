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

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

coco_gt_file = "$YOUR_PATH/coco_karpathy_test_gt.json"

class COCODataset(Dataset):
    split_paths = {
        'val': [
            'coco/annotations/coco_karpathy_val.json', 
            'coco/images/',
        ],
        'test': [
            'coco/annotations/coco_karpathy_test.json', 
            'coco/images/',
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
    
    def __getitem__(self, item):
        sample = self.samples[item]
        image_id = int(sample["image"].split("/")[-1].strip(".jpg").split("_")[-1])

        image_path = os.path.join(self.root, self.img_path, sample["image"])
        image = Image.open(image_path).convert("RGB")
        
        prompt = self.image_system_msg
        prompt += f" [USER]: {self.image_placeholder} please provide an accurate and concise description of the given image. [ASSISTANT]:"

        return {"image": image, "instance_id": image_id, "prompt": prompt}
        
        
def coco_dataloader(root_path, batch_size):
    dataset = COCODataset(root=root_path)
    print(f"===> COCO num_samples: {len(dataset)}")  # 5000
    
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
        min_length=8,
        length_penalty=-1.0,
        inference_type="generation"
    )
    
    return dataloader, inference_kwargs, {}


def coco_results_processor(results, output_dir):
    # save predictions
    save_result = []
    for res in results:
        caption = res['prediction'].split('\n')[0].split('. ')[0]
        caption = caption if len(caption) == 0 or caption[-1] != '.' else caption[:-1]
        caption = caption.lower()
        save_result.append({"image_id": res['instance_id'], "caption": caption})
    result_file = os.path.join(output_dir + "coco_answer.json")
    with open(result_file, "w") as f:
        json.dump(save_result, f)
    
    annotation_file = coco_gt_file

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(result_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval

# def coco_eval(result_file):
#     annotation_file = coco_gt_file

#     # create coco object and coco_result object
#     coco = COCO(annotation_file)
#     coco_result = coco.loadRes(result_file)

#     # create coco_eval object by taking coco and coco_result
#     coco_eval = COCOEvalCap(coco, coco_result)

#     # evaluate on a subset of images by setting
#     # coco_eval.params['image_id'] = coco_result.getImgIds()
#     # please remove this line when evaluating the full validation set
#     # coco_eval.params['image_id'] = coco_result.getImgIds()

#     # evaluate results
#     # SPICE will take a few minutes the first time, but speeds up due to caching
#     coco_eval.evaluate()

#     # print output evaluation scores
#     for metric, score in coco_eval.eval.items():
#         print(f"{metric}: {score:.3f}")

#     return coco_eval


# coco_eval("/share/project/qiying/projects/Emu-evaluation/Emu/output/coco_answer.json")