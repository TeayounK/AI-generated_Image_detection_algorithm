import torch
import torch.nn as nn

class LearnableConvBlock(nn.Module):
   
    def __init__(
        self,
        in_channels: int = 30,
        out_channels: int = 30,
        kernel_size: int = 3,
        use_bias: bool = False
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd so padding keeps spatial size.")

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=use_bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.Hardtanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], got shape {tuple(x.shape)}")
        return self.block(x)





class FingerprintExtractor(nn.Module):
    """


    Input:
        rich_feat: [B, C, H, W]
        poor_feat: [B, C, H, W]

    Output:
        fingerprint: [B, C, H, W]
    """
    def __init__(self, mode: str = "subtract"):
        super().__init__()
        valid_modes = {"subtract", "abs_subtract", "concat"}
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}")
        self.mode = mode

    def forward(self, rich_feat: torch.Tensor, poor_feat: torch.Tensor) -> torch.Tensor:
        if rich_feat.shape != poor_feat.shape:
            raise ValueError(
                f"rich_feat and poor_feat must have same shape, "
                f"got {tuple(rich_feat.shape)} vs {tuple(poor_feat.shape)}"
            )

        if self.mode == "subtract":
            return rich_feat - poor_feat

        if self.mode == "abs_subtract":
            return torch.abs(rich_feat - poor_feat)

 
        return torch.cat([rich_feat, poor_feat], dim=1)

class PatchCraftFeatureModule(nn.Module):
    """
    SRM 출력 -> learnable conv block -> fingerprint

    Input:
        rich_srm: [B, 30, H, W]
        poor_srm: [B, 30, H, W]

    Output:
        fingerprint: [B, C, H, W]
    """
    def __init__(
        self,
        srm_channels: int = 30,
        feat_channels: int = 30,
        kernel_size: int = 3,
        fingerprint_mode: str = "subtract"
    ):
        super().__init__()

        self.learnable_block = LearnableConvBlock(
            in_channels=srm_channels,
            out_channels=feat_channels,
            kernel_size=kernel_size
        )

        self.fingerprint = FingerprintExtractor(mode=fingerprint_mode)

    def forward(self, rich_srm: torch.Tensor, poor_srm: torch.Tensor) -> torch.Tensor:
        rich_feat = self.learnable_block(rich_srm)   # [B, C, H, W]
        poor_feat = self.learnable_block(poor_srm)   # [B, C, H, W]

        fp = self.fingerprint(rich_feat, poor_feat)  # [B, C, H, W]
        return fp
    


#     B, C, H, W = 4, 30, 256, 256

# rich_srm = torch.randn(B, C, H, W)
# poor_srm = torch.randn(B, C, H, W)

# feature_module = PatchCraftFeatureModule(
#     srm_channels=30,
#     feat_channels=32,
#     kernel_size=3,
#     fingerprint_mode="subtract"
# )

# fingerprint = feature_module(rich_srm, poor_srm)

# print("fingerprint shape:", fingerprint.shape)
# # expected: [4, 32, 256, 256]