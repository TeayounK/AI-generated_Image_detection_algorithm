from typing import Tuple
import torch
from PIL import Image

def smash(img: Image.Image, patch_size: int = 32, num_patches: int = 192, diversity_threshold: float = 0.33) -> Tuple[torch.Tensor, torch.Tensor]:
    
    """
    Extracts patches from the image, computes texture diversity scores, sorts them,
    and returns the top (rich) and bottom (poor) patches based on the threshold.
    
    Args:
        img: Input PIL image.
        patch_size: Size of each square patch (e.g., 32x32).
        num_patches: Total number of patches to extract.
        diversity_threshold: Fraction for top/bottom selection (e.g., 0.33 for 33%).

    Returns:
        rich_patches: Tensor of rich texture patches [K, C, patch_size, patch_size].
        poor_patches: Tensor of poor texture patches [K, C, patch_size, patch_size].
    """

    # TODO: Implement patch extraction (e.g., unfold), diversity calculation (abs diffs in 4 directions), sort, select.
    pass