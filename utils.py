import torch
from PIL import Image
import numpy as np
from decord import VideoReader


def get_index(num_frames, num_segments):
    print(f"===> num_frames: {num_frames}, num_segments: {num_segments}")
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


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


def process_video(video_path=None):
    vr = VideoReader(video_path)
    frame_indices = get_index(len(vr), 8)
    image_list = []
    text_sequence = ''
    from inference import image_placeholder
    for frame_index in frame_indices:
        image = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        image = process_img(img=image)
        image_list.append(image)
        text_sequence += image_placeholder
    return image_list, text_sequence
