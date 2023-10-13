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

mmbench_datafile = "/share/project/zxs/datasets/benchmarks/mmbench/mmbench_dev_en_20231003.tsv"

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

class MMBenchDataset(Dataset):
    def __init__(self,
                 data_file,
                 sys_prompt='There are several options:'):
        self.df = pd.read_csv(data_file, sep='\t')
        self.raw_len = len(self.df)
        self.sys_prompt = sys_prompt
        
        from .. import image_placeholder
        self.image_placeholder = image_placeholder

        self.circular, self.answer_dict, self.option_num_dict = self.process()

    def __len__(self):
        return len(self.circular) // 100

    def __getitem__(self, idx):
        return self.circular[idx]

    def process(self):
        circular_list, answer_dict, num_dict = [], {}, {}
        for idx in range(self.raw_len):
            index = int(self.df.iloc[idx]['index'])
            image = self.df.iloc[idx]['image']
            image = decode_base64_to_image(image)
            question = self.df.iloc[idx]['question']
            answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[0].keys() else None
            category = self.df.iloc[idx]['category']
            l2_category = self.df.iloc[idx]['l2-category']
            
            option_candidate = ['A', 'B', 'C', 'D', 'E']
            options = {
                cand: self.load_from_df(idx, cand)
                for cand in option_candidate
                if self.load_from_df(idx, cand) is not None
            }
            option_candidate_num = len(options.keys())
            
            for i in range(option_candidate_num):
                shifted_options = {}
                for option in options.keys():
                    shifted_options[self.option_shift(option, option_candidate_num)] = options[option]
                options = shifted_options
                answer = self.option_shift(answer, option_candidate_num)
                
                options_prompt = f'{self.sys_prompt}\n'
                for key in option_candidate[:option_candidate_num]:
                    options_prompt += f'{key}. {options[key]}\n'

                hint = self.load_from_df(idx, 'hint')
                if hint is not None:
                    question_prompt = hint + ' ' + self.image_placeholder + ' ' + question + ' ' + options_prompt + ' Please select an option as answer.'
                else:
                    question_prompt = self.image_placeholder + ' ' + question + ' ' + options_prompt + ' Please answer only the character corresponding to your answer choice (e.g., "A," "B," "C," or "D").'
                
                index_sub = f"{index}_{i}"
                circular_list.append({"image": image, "question": question_prompt, "instance_id": index_sub})
                answer_dict[index_sub] = {"answer": answer, "category": f"{l2_category}/{category}"}
            
            num_dict[index] = option_candidate_num
                
        return circular_list, answer_dict, num_dict
    
    def option_shift(self, option, option_num):
        assert 1 < option_num <= 5
        assert option in ['A', 'B', 'C', 'D', 'E']
        int2option = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
        option2int = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        return int2option[(option2int[option] + 1) % option_num]
    
    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None
        
        
def mmbench_dataloader(batch_size):
    dataset = MMBenchDataset(data_file=mmbench_datafile)
    print(f"===> MMBench num_samples: {len(dataset)}")  # 4000
    
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
        max_new_tokens=5,
        min_length=1,
        length_penalty=-1.0,
    )
    
    return dataloader, inference_kwargs, {"answer_dict": dataset.answer_dict, "option_num_dict": dataset.option_num_dict}


def mmbench_results_processor(results, output_dir, answer_dict, option_num_dict):
    # calculate accuracy
    correct_cnt_dict = {}
    for res in results:
        if answer_dict[res['instance_id']]['answer'] == res['prediction'][0]:
            index = int(res['instance_id'].split('_')[0])
            correct_cnt_dict[index] = correct_cnt_dict[index] + 1 if correct_cnt_dict.get(index) else 1
    
    total_cnt, correct_cnt = 0, 0
    for key in correct_cnt_dict.keys():
        total_cnt += 1
        if correct_cnt_dict[key] == option_num_dict[key]:
            correct_cnt += 1
    accuracy = correct_cnt / total_cnt
    print(f"MM-Bench, Accuracy = {accuracy}")
    
    # save predictions
    save_result = {}
    for res in results:
        save_result.update({res['instance_id']: res['prediction']})
    result_file = os.path.join(output_dir + "mmbench_answer.json")
    with open(result_file, "w") as f:
        json.dump(save_result, f)
    print(f"MM-Bench: Saved results for leaderboard evaluation at {result_file}")
    
    return accuracy