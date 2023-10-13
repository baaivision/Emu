from tqdm import tqdm
import numpy as np

from dataclasses import dataclass, field
import torch
from inference import prepare_model, parse_args
    
def process_img(img_path=None, img=None, device=torch.device("cuda")):
    assert img_path is not None or img is not None, "you should pass either path to an image or a PIL image object"
    width, height = 224, 224
    OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
    if img_path:
        img = Image.open(img_path).convert("RGB")
    img = img.resize((width, height))
    img = np.array(img) / 255.
    img = (img - OPENAI_DATASET_MEAN) / OPENAI_DATASET_STD
    img = torch.tensor(img).to(device).to(torch.float)
    img = torch.einsum('hwc->chw', img)
    img = img.unsqueeze(0)
    return img

def prepare_generation_data(samples, device):
    images = []
    prompts = []
    for sample in samples:
        images.append(process_img(img=sample["image"], device=device))
        prompts.append(sample["prompt"])
    images = torch.cat(images, dim=0)
    return images, prompts, None

def prepare_prediction_data(samples, device):
    images = []
    prompts = []
    options = []
    
    for sample in samples:
        images.append(process_img(img=sample["image"], device=device))
        prompts.append(sample["prompt"])
        options.append(sample["options"])
        
    images = torch.cat(images, dim=0)
    return images, prompts, options

def prepare_vizwiz_data(samples, device):
    from .. import image_placeholder, image_system_msg
    images = []
    prompts = []
    answerability_prompts = []
    for sample in samples:
        images.append(process_img(img=sample["image"], device=device))
        prompts.append(sample["prompt"])
        answerability_prompts.append(sample["answerability_prompt"])
    images = torch.cat(images, dim=0)
    return images, prompts, answerability_prompts

def emu_prepare_model(model_args):
    model = prepare_model('Emu-14B', model_args)
    return model

def emu_inference(dataloader, model_args, device, inference_kwargs, model=None):
    """
    batches of dataloader contains a list of dict, keys: image, question, and instance_id
    """
    if model is None:
        model_args.device = device
    
        model = emu_prepare_model(model_args)
        model.to(torch.bfloat16).to(device)
    
    inference_type = inference_kwargs.pop("inference_type")
    if inference_type == "generation":
        prepare_data = prepare_generation_data
    elif inference_type == "classification":
        prepare_data = prepare_prediction_data
    elif inference_type == "vizwiz":
        prepare_data = prepare_vizwiz_data
    else:
        raise NotImplementedError("Only support 'generation', 'classification' and 'vizwiz' tasks.")
    
    results = []
    for samples in tqdm(dataloader):
        image, prompt, auxiliary = prepare_data(samples, device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if inference_type == "generation":
                outputs = model.generate(
                    {"image": image, "prompt": prompt},
                    **inference_kwargs,
                )
            elif inference_type == "classification":
                outputs = model.predict(
                    {"image": image, "prompt": prompt}, 
                    auxiliary,
                    **inference_kwargs,
                )
            elif inference_type == "vizwiz":
                from .trie import Trie
                trie = []
                for choice in ['yes.', 'no.']:
                    idxs = model.decoder.tokenizer.encode(choice)[1:]
                    trie.append(idxs + [2])
                trie = Trie(trie)
                def prefix_allowed_tokens_fn(batch_id, input_ids):
                    return trie.get(input_ids.tolist()[1:])
                
                answerabilitys = model.generate(
                    {"image": image, "prompt": auxiliary}, 
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                )
                predictions = model.generate(
                    {"image": image, "prompt": prompt}, 
                    **inference_kwargs,
                )
                outputs = [[a, b] for a, b in zip(answerabilitys, predictions)]

        for sample, output in zip(samples, outputs):
            results.append({
                "instance_id": sample["instance_id"],
                "prediction": output,
            })
    return results
