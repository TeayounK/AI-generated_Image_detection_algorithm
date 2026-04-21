import torch
import torch.nn as nn
import torch.nn.functional as F

class SRMFilters(nn.Module):
    """
    Applies the 30 SRM high-pass filters from the Rich Models paper (Fig. 2).
    """
    def __init__(self):
        super().__init__()
        # Define the 30 kernels as tensors (from classes like 1st/2nd/3rd order, square, edge).
        
        kernels = [
            # a
            torch.tensor([
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=torch.float32),
            torch.tensor([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=torch.float32),
            torch.tensor([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, -1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=torch.float32),
            torch.tensor([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0]
            ], dtype=torch.float32),
            torch.tensor([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=torch.float32),
            torch.tensor([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=torch.float32),
            torch.tensor([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=torch.float32),
            torch.tensor([
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=torch.float32),
            # b
            torch.tensor([
                [0, 0, -1, 0, 0],
                [0, 0, 3, 0, 0],
                [0, 0, -3, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=torch.float32),
            torch.tensor([
                [0, 0, 0, 0, -1],
                [0, 0, 0, 3, 0],
                [0, 0, -3, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=torch.float32),
            torch.tensor([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -3, 3, -1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=torch.float32),
            torch.tensor([
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, -3, 0, 0],
                [0, 0, 0, 3, 0],
                [0, 0, 0, 0, -1]
            ], dtype=torch.float32),
            torch.tensor([
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, -3, 0, 0],
                [0, 0, 3, 0, 0],
                [0, 0, -1, 0, 0]
            ], dtype=torch.float32),
            torch.tensor([
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, -3, 0, 0],
                [0, 3, 0, 0, 0],
                [-1, 0, 0, 0, 0]
            ], dtype=torch.float32),
            torch.tensor([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [-1, 3, -3, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=torch.float32),
            torch.tensor([
                [-1, 0, 0, 0, 0],
                [0, 3, 0, 0, 0],
                [0, 0, -3, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1]
            ], dtype=torch.float32),
            # c
            torch.tensor([
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, -2, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=torch.float32),
            torch.tensor([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=torch.float32),
            torch.tensor([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, -2, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=torch.float32),
            torch.tensor([
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, -2, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0]
            ], dtype=torch.float32),
            # d
            torch.tensor([
                [0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=torch.float32),
            torch.tensor([
                [0, 0, 0, 0, 0],
                [0, 0, 2, -1, 0],
                [0, 0, -4, 2, 0],
                [0, 0, 2, -1, 0],
                [0, 0, 0, 0, 0]
            ], dtype=torch.float32),
            torch.tensor([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0]
            ], dtype=torch.float32),
            torch.tensor([
                [0, 0, 0, 0, 0],
                [0, -1, 2, 0, 0],
                [0, 2, -4, 0, 0],
                [0, -1, 2, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=torch.float32),
            # e
            torch.tensor([
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=torch.float32),
            torch.tensor([
                [0, 0, -2, 2, -1],
                [0, 0, 8, -6, 2],
                [0, 0, -12, 8, -2],
                [0, 0, 8, -6, 2],
                [0, 0, -2, 2, -1]
            ], dtype=torch.float32),
            torch.tensor([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1]
            ], dtype=torch.float32),
            torch.tensor([
                [-1, 2, -2, 0, 0],
                [2, -6, 8, 0, 0],
                [-2, 8, -12, 0, 0],
                [2, -6, 8, 0, 0],
                [-1, 2, -2, 0, 0]
            ], dtype=torch.float32),
            # f
            torch.tensor([
                [0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0]
            ], dtype=torch.float32),
            # g
            torch.tensor([
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1]
            ], dtype=torch.float32),
        ]

        kernels = torch.stack(kernels)   # [30, 5, 5]
        kernels = kernels.unsqueeze(1)   # [30, 1, 5, 5]
        self.register_buffer('kernels', kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies filters to input (e.g., collage).

        Args:
            x: Input tensor [B, C, H, W] (e.g., grayscale or RGB; convert if needed).

        Returns:
            residuals: Filtered residuals [B, 30, H, W].
        """
        # Convert to grayscale if RGB
        if x.shape[1] == 3:
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        # Apply all kernels in one batched conv2d, padding=2 to preserve spatial size for 5x5 kernels
        return torch.nn.functional.conv2d(x, self.kernels, padding=2)


class LearnableSRMFilters(nn.Module):
    """
    Learnable variant of SRMFilters. Same input/output shape ([B, C, H, W] -> [B, num_filters, H, W]),
    but the 30 5x5 kernels are trainable parameters instead of fixed buffers.

    Args:
        num_filters: Number of output filter maps (default 30 to match SRMFilters).
        init_from_srm: If True, initialize weights from the hardcoded SRM kernels as a warm start.
                       If False, use default Kaiming init (random).
    """
    def __init__(self, num_filters: int = 30, init_from_srm: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=5,
            padding=2,
            bias=False,
        )

        if init_from_srm:
            if num_filters != 30:
                raise ValueError("init_from_srm=True requires num_filters=30 to match the SRM kernel count.")
            with torch.no_grad():
                self.conv.weight.copy_(SRMFilters().kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 3:
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        return self.conv(x)