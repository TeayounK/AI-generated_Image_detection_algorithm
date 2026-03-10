import torch

from preprocess.smash import smash
from preprocess.reconstruct import reconstruct
from model.filters import SRMFilters
from model.fingerprint import FingerprintExtractor

#def preprocess(images, srm: SRMFilters = None, extractor: FingerprintExtractor = None, device = torch.cuda()):
#
#    B = images.size(0)
#    features = []
#    for i in range(B):
#        img = images[i]  # [3, H, W]
#        rich_patches, poor_patches = smash(img)
#        rich_collage = reconstruct(rich_patches)
#        poor_collage = reconstruct(poor_patches)
#        rich_res = srm(rich_collage.unsqueeze(0))  # [1, 30, H, W]
#        poor_res = srm(poor_collage.unsqueeze(0))
#        feat = extractor(rich_res, poor_res)  # [1, feat_dim]
#        features.append(feat)
#
#    return features

def preprocess(images: torch.Tensor, srm: SRMFilters, extractor: FingerprintExtractor, device: torch.device) -> torch.Tensor:
    """
    Vectorized preprocessing for the entire batch.
    
    Args:
        images: Batched input [B, 3, H, W] (e.g., 256x256).
        srm: SRMFilters module on device.
        extractor: FingerprintExtractor module on device.
        device: torch.device('cuda') or cpu.

    Returns:
        features: Batched fingerprints [B, feat_dim] (e.g., 30).
    """
    images = images.to(device)
    B, C, H, W = images.shape
    P = 32  # Patch size from paper
    step = P  # Non-overlapping
    diversity_threshold = 0.33

    # Vectorized Smash: Extract all patches at once
    # Unfold: [B, C*P*P, num_patches_h * num_patches_w]
    patches = torch.unfold(images, kernel_size=(P, P), stride=(step, step))
    num_patches = patches.shape[-1]
    patches = patches.view(B, C, P, P, num_patches).permute(0, 4, 1, 2, 3)  # [B, N, C, P, P]

    # Vectorized diversity scores (Eq. 1: sum abs diffs in 4 directions)
    scores = (
        torch.sum(torch.abs(patches[:, :, :, :-1, :] - patches[:, :, :, 1:, :]), dim=(2,3,4)) +  # horiz
        torch.sum(torch.abs(patches[:, :, :-1, :, :] - patches[:, :, 1:, :, :]), dim=(2,3,4)) +  # vert
        torch.sum(torch.abs(patches[:, :, :-1, :-1, :] - patches[:, :, 1:, 1:, :]), dim=(2,3,4)) +  # diag
        torch.sum(torch.abs(patches[:, :, :-1, 1:, :] - patches[:, :, 1:, :-1, :]), dim=(2,3,4))   # counter-diag
    )  # [B, N]

    # Sort scores descending per image
    sorted_scores, indices = torch.sort(scores, dim=1, descending=True)  # [B, N]
    
    # Select top/bottom K per batch item
    K = int(num_patches * diversity_threshold)
    rich_indices = indices[:, :K]  # [B, K]
    poor_indices = indices[:, -K:]  # [B, K]
    
    # Gather rich/poor patches
    batch_idx = torch.arange(B).unsqueeze(1).expand(-1, K).to(device)  # [B, K]
    rich_patches = patches[batch_idx, rich_indices]  # [B, K, C, P, P]
    poor_patches = patches[batch_idx, poor_indices]  # [B, K, C, P, P]

    # Vectorized Reconstruct: Reshape to collages [B, C, collage_H, collage_W]
    grid_size = int(K ** 0.5)
    K_square = grid_size ** 2
    rich_patches = rich_patches[:, :K_square]  # Truncate to square
    poor_patches = poor_patches[:, :K_square]
    
    # Reshape: [B, grid, grid, C, P, P] → permute → view [B, C, grid*P, grid*P]
    rich_collages = rich_patches.view(B, grid_size, grid_size, C, P, P).permute(0, 3, 1, 4, 2, 5).reshape(B, C, grid_size*P, grid_size*P)
    poor_collages = poor_patches.view(B, grid_size, grid_size, C, P, P).permute(0, 3, 1, 4, 2, 5).reshape(B, C, grid_size*P, grid_size*P)

    # High-Pass Filters (SRM)
    rich_res = srm(rich_collages)  # [B, 30, H, W]
    poor_res = srm(poor_collages)  # [B, 30, H, W]

    # Fingerprint Extraction
    features = extractor(rich_res, poor_res)  # [B, feat_dim]

    return features