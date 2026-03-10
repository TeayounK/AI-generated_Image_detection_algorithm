import torch
import torch.nn as nn

class PatchCraftCNN(nn.Module):
    """
    The CNN classifier network from PatchCraft (cascade of conv blocks + FC).
    """
    def __init__(self, in_features: int = 30, num_classes: int = 1):
        super().__init__()
        # TODO: Define conv blocks (4x: Conv2d + BN + ReLU + MaxPool), adaptive pool, linear FC.
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN.

        Args:
            x: Input features [B, 30, H, W] from fingerprint.

        Returns:
            logits: [B, 1] for binary classification (real/fake).
        """
        # TODO: Pass through conv_blocks, pool, flatten, FC.
        pass