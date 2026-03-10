import os
from typing import Tuple, Optional, List, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset

class AIGCDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        split: str = 'train'
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform
        self.split = split

        self.samples: List[Tuple[str, int]] = []

        print(f"[{split.upper()}] Scanning root: {self.root_dir}")
        real_count = fake_count = 0

        for current_dir, subdirs, files in os.walk(self.root_dir):
            is_real_dir = '0_real' in current_dir
            is_fake_dir = '1_fake' in current_dir

            if is_real_dir or is_fake_dir:
                label = 0 if is_real_dir else 1
                count = 0
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_path = os.path.join(current_dir, file)
                        self.samples.append((full_path, label))
                        count += 1
                if label == 0:
                    real_count += count
                else:
                    fake_count += count

        if not self.samples:
            raise ValueError(f"No images found in {self.root_dir}.")

        print(f"Loaded {len(self.samples)} images (Real: {real_count}, Fake: {fake_count})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label