import torch
import torch.nn as nn
import torch.optim as optim


class ConvBNReLU(nn.Module):
    """
    Basic block: Conv -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class PatchCraftClassifier(nn.Module):
    """
    Classifier structure based on the paper's supplement Table 5.

    Structure:
        4 x (Conv + BN + ReLU)
        AvgPool
        2 x (Conv + BN + ReLU)
        AvgPool
        2 x (Conv + BN + ReLU)
        AvgPool
        2 x (Conv + BN + ReLU)
        AdaptiveAvgPool
        Flatten
        FC

    Notes:
    - The paper says each conv stage uses 32 kernels/channels.
    - The final FC produces one logit for binary classification.
    """
    def __init__(self, in_channels=32, feature_channels=32, num_classes=1):
        super().__init__()

        self.features = nn.Sequential(
            # Before first pooling: 4 conv blocks
            ConvBNReLU(in_channels, feature_channels),
            ConvBNReLU(feature_channels, feature_channels),
            ConvBNReLU(feature_channels, feature_channels),
            ConvBNReLU(feature_channels, feature_channels),

            nn.AvgPool2d(kernel_size=2, stride=2),

            # After first pooling: 2 conv blocks
            ConvBNReLU(feature_channels, feature_channels),
            ConvBNReLU(feature_channels, feature_channels),

            nn.AvgPool2d(kernel_size=2, stride=2),

            # After second pooling: 2 conv blocks
            ConvBNReLU(feature_channels, feature_channels),
            ConvBNReLU(feature_channels, feature_channels),

            nn.AvgPool2d(kernel_size=2, stride=2),

            # Final conv stage: 2 conv blocks
            ConvBNReLU(feature_channels, feature_channels),
            ConvBNReLU(feature_channels, feature_channels),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(feature_channels, num_classes)

    def forward(self, x):
        """
        x: shape (B, C, H, W)
        returns:
            logits: shape (B,) if num_classes=1
        """
        x = self.features(x)          # (B, 32, 1, 1)
        x = torch.flatten(x, 1)       # (B, 32)
        x = self.classifier(x)        # (B, 1)
        return x.squeeze(1)           # (B,)
    



# model = PatchCraftClassifier(in_channels=32)
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)

# x = torch.randn(8, 32, 256, 256)
# y = torch.randint(0, 2, (8,)).float()

# optimizer.zero_grad()
# logits = model(x)
# loss = criterion(logits, y)
# loss.backward()
# optimizer.step()

# print("loss =", loss.item())


# model.eval()

# with torch.no_grad():
#     x = torch.randn(4, 32, 256, 256)
#     logits = model(x)
#     probs = torch.sigmoid(logits)
#     preds = (probs > 0.5).long()

# print("logits:", logits)
# print("probs:", probs)
# print("preds:", preds)