from typing import Tuple
from matplotlib import transforms
import torch
from PIL import Image
import numpy as np

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
    # Choose random coordinates for patch extraction
    corrodinate_x = torch.randint(0, img.width - patch_size, (num_patches,))
    corrodinate_y = torch.randint(0, img.height - patch_size, (num_patches,))
    patches = []
    for x, y in zip(corrodinate_x, corrodinate_y):
        # smash the patch by cropping the image at the specified coordinates
        patch = img.crop((x, y, x + patch_size, y + patch_size))
        patches.append(transforms.ToTensor()(patch))
    patches = torch.stack(patches)  # [num_patches, C, patch_size, patch_size]

    # calculate texture diversity scores for each patch
    diversity_scores = torch.zeros(num_patches)
    for _ in range(4):
        # CONCERN: the paper said to add each difference to the score, but it seems more intuitive to average them.
        # we can do .sum(dim=[1,2,3]) instead of .mean(), but then it would be more sensitive to patch size and number of channels. 
        # so, averaging makes it more sense to me to get the average difference for each pixcel.
        if _==0:
            metric = torch.abs(patches[:, :, 1:, :] - patches[:, :, :-1, :]).mean(dim=[1, 2, 3])  # vertical
        elif _==1:
            metric = torch.abs(patches[:, :, :, 1:] - patches[:, :, :, :-1]).mean(dim=[1, 2, 3])  # horizontal
        elif _==2:
            metric = torch.abs(patches[:, :, 1:, 1:] - patches[:, :, :-1, :-1]).mean(dim=[1, 2, 3])  # diagonal
        else:
            metric = torch.abs(patches[:, :, 1:, :-1] - patches[:, :, :-1, 1:]).mean(dim=[1, 2, 3])  # anti-diagonal
        diversity_scores += metric
    diversity_scores /= 4  #OPTIONAL: average the scores from all directions

    # sort patches by diversity scores
    _, sorted_indices = torch.sort(diversity_scores, descending=True)

    # select top and bottom patches based on the threshold
    num_rich = int(num_patches * diversity_threshold)
    num_poor = int(num_patches * diversity_threshold)

    rich_indices = sorted_indices[:num_rich]
    poor_indices = sorted_indices[-num_poor:]

    rich_patches = patches[rich_indices]
    poor_patches = patches[poor_indices]
    # this gives us 2/3 of the patches excluding the middle 1/3.
    # perhaps we want to explore the results to get a better sense of the fraction of patches to select. 
    # For example, we could visualize the diversity scores and see if there is a natural cutoff point that separates rich and poor textures.
    return rich_patches, poor_patches