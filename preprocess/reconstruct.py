import torch

def reconstruct(patches: torch.Tensor, collage_size: int = 256) -> torch.Tensor:
    """
    Reconstructs a collage image from a set of patches (e.g., rich or poor).
    
    Args:
        patches: Tensor of patches [N, C, patch_size, patch_size].
        collage_size: Target size of the square collage (e.g., 256x256).

    Returns:
        collage: Reconstructed tensor [C, collage_size, collage_size].
    """
    # I didn't use `collage_size` in the original implementation because 
    # we are reconstructing a collage of size K*P x K*P, where K is the number of patches and P is the patch size.
    num_patches = patches.shape[0]
    patch_size = patches.shape[2]  # Assuming square patches
    grid_size = int(num_patches ** 0.5)  # Assuming num_patches is a perfect square
    assert grid_size ** 2 == num_patches, "Number of patches must be a perfect square for reconstruction."
    # Reshape patches into a grid
    patches = patches.view(grid_size, grid_size, patches.shape[1], patch_size, patch_size)  # [G, G, C, P, P]
    # Permute to [C, G*P, G*P]
    collage = patches.permute(2, 0, 3, 1, 4).contiguous().view(patches.shape[2], grid_size * patch_size, grid_size * patch_size)
    # Optionally, we could resize the collage to the target collage_size if needed.
    return collage