import torch
import torch.nn as nn

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