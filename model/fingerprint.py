import torch
import torch.nn as nn

class FingerprintExtractor(nn.Module):
    """
    Extracts fingerprint by processing rich/poor residuals and computing contrast (PatchCraft Sec. 3).
    """
    def __init__(self, in_channels: int = 30):
        super().__init__()
        # TODO: Define layers: 1x1 conv, BatchNorm2d, HardTanh.
        pass

    def forward(self, rich_res: torch.Tensor, poor_res: torch.Tensor) -> torch.Tensor:
        """
        Processes residuals and returns contrast fingerprint.

        Args:
            rich_res: Rich residuals [B, 30, H, W].
            poor_res: Poor residuals [B, 30, H, W].

        Returns:
            fingerprint: Contrast features [B, feature_dim] (e.g., after pooling).
        """
        # TODO: Apply conv+bn+hardtanh to each, subtract, global pool/flatten.
        pass