from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import transformers

from .datasets.mmvet import mmvet_dataloader, mmvet_results_processor
from .datasets.mmbench import mmbench_dataloader, mmbench_results_processor
from .datasets.coco import coco_dataloader, coco_results_processor
from .datasets.visdial import visdial_dataloader, visdial_results_processor
from .datasets.okvqa import okvqa_dataloader, okvqa_results_processor
from .datasets.vqav2 import vqav2_dataloader, vqav2_results_processor
from .datasets.vizwiz import vizwiz_dataloader, vizwiz_results_processor

from .models.emu import emu_inference

image_placeholder = "[IMG]" + "<image>" * 32 + "[/IMG]"
image_system_msg = "You will be presented with an image: [IMG]ImageContent[/IMG]. You will be able to see the image after I provide it to you. Please answer my questions based on the given image."

dataloader_dict = {
    "mmvet": mmvet_dataloader,
    "mmbench": mmbench_dataloader,
    "coco": coco_dataloader,
    "visdial": visdial_dataloader,
    "vqav2": vqav2_dataloader,
    "okvqa": okvqa_dataloader,
    "vizwiz": vizwiz_dataloader
}

results_processor_dict = {
    "mmvet": mmvet_results_processor,
    "mmbench": mmbench_results_processor,
    "coco": coco_results_processor,
    "visdial": visdial_results_processor,
    "vqav2": vqav2_results_processor,
    "okvqa": okvqa_results_processor,
    "vizwiz": vizwiz_results_processor
}


@dataclass
class EvalArguments:
    output_path: str = field(default='./output/')
    dataset_name: str = field(default='coco')
    batch_size: int = 1
    root_path: str = field(default='./benchmarks')
    
@dataclass
class ModelArguments:
    instruct: bool = field(default=True)
    ckpt_path: str = field(default="./Emu-instruct.pt")


def evaluate_engine(model=None):
    # init distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank())
    if dist.get_rank() == 0:
        print(f"World Size = {dist.get_world_size()}")
    device = torch.device(f"cuda:{dist.get_rank()}")
    
    # arguments
    model_args = ModelArguments
    eval_args = EvalArguments
    parser = transformers.HfArgumentParser((model_args, eval_args))
    model_args, eval_args = parser.parse_args_into_dataclasses()
    dataset_name = eval_args.dataset_name
    
    # dataloader --> sample: (image, prompt, instance_id) for generation, or (image, prompt, instance_id, options) for classification
    dataloader, inference_kwargs, results_process_kwargs = dataloader_dict[dataset_name](root_path=eval_args.root_path, batch_size=eval_args.batch_size)
    
    # model inference --> (instance_id, prediction)
    results = emu_inference(dataloader, model_args, device, inference_kwargs=inference_kwargs, model=model)
    
    # gather results
    if dist.is_initialized():
        dist.barrier()
        results_gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(results_gathered, results)
    else:
        results_gathered = [results]
    # deduplicate
    results, unique_ids = [], set()
    for result in sum(results_gathered, []):
        instance_id = result["instance_id"]
        if instance_id in unique_ids:
            continue
        results.append(result)
        unique_ids.add(instance_id)
    
    # process results
    metric = None
    if dist.get_rank() == 0:
        results_processor = results_processor_dict[dataset_name]
        metric = results_processor(results, eval_args.output_path, **results_process_kwargs)
    
    return metric
