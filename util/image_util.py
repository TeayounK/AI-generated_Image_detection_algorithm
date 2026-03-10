from PIL import Image
import torch
import numpy as np
import random

def simple_resize(img: Image.Image, size: int = 256) -> Image.Image:
    """Resize keeping aspect ratio, then center crop if needed."""
    w, h = img.size
    if w < h:
        new_w = size
        new_h = int(size * h / w)
    else:
        new_h = size
        new_w = int(size * w / h)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))

def random_horizontal_flip(img: Image.Image, p: float = 0.5) -> Image.Image:
    if random.random() < p:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL Image (HWC, uint8) → torch tensor (CHW, float32, [0,1])"""
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))
    return tensor

def normalize(tensor: torch.Tensor, mean: list = [0.5, 0.5, 0.5], std: list = [0.5, 0.5, 0.5]) -> torch.Tensor:
    """Normalize tensor in-place: (x - mean) / std per channel"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (tensor - mean) / std

def train_transform(img: Image.Image) -> torch.Tensor:
    img = simple_resize(img, size=256)
    img = random_horizontal_flip(img)
    tensor = to_tensor(img)
    tensor = normalize(tensor)
    return tensor

def val_transform(img: Image.Image) -> torch.Tensor:
    img = simple_resize(img, size=256)
    tensor = to_tensor(img)
    tensor = normalize(tensor)
    return tensor
