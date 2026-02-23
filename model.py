import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Reusable building blocks
# ──────────────────────────────────────────────────────────────────────────────


class ConvBNReLU(nn.Module):
    """Two consecutive Conv → BN → ReLU layers (the standard UNet conv block)."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.block = nn.Sequential(
            # First convolution
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            # Second convolution
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """
    Encoder path: two downsampling stages.

    Each stage applies ConvBNReLU then MaxPool(2×2).
    Returns the feature map BEFORE pooling (skip tensor) and the pooled tensor.
    """

    def __init__(self):
        super().__init__()
        # Stage 1: 3  → 64  channels
        self.enc1 = ConvBNReLU(3, 64)
        # Stage 2: 64 → 128 channels
        self.enc2 = ConvBNReLU(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Stage 1
        s1 = self.enc1(x)  # skip connection tensor  (B, 64,  H,   W)
        x = self.pool(s1)  # downsampled             (B, 64,  H/2, W/2)

        # Stage 2
        s2 = self.enc2(x)  # skip connection tensor  (B, 128, H/2, W/2)
        x = self.pool(s2)  # downsampled             (B, 128, H/4, W/4)

        return x, s1, s2


class Bottleneck(nn.Module):
    """Central bottleneck (lowest resolution)."""

    def __init__(self):
        super().__init__()
        self.conv = ConvBNReLU(128, 256)  # 128 → 256 channels

    def forward(self, x):
        return self.conv(x)  # (B, 256, H/4, W/4)


class Decoder(nn.Module):
    """
    Decoder path: two upsampling stages with skip connections.

    Each stage uses TransposedConv to upsample, then concatenates the
    corresponding encoder skip tensor, then applies ConvBNReLU.
    """

    def __init__(self):
        super().__init__()
        # Up-sample 1: 256 → 128, then concat with s2 (128) → ConvBNReLU(256→128)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = ConvBNReLU(256, 128)  # 128 (up) + 128 (skip) = 256 in

        # Up-sample 2: 128 → 64,  then concat with s1 (64)  → ConvBNReLU(128→64)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBNReLU(128, 64)  # 64  (up) + 64  (skip) = 128 in

    def forward(self, x, s2, s1):
        # Up-sample 1  +  skip from encoder stage 2
        x = self.up1(x)  # (B, 128, H/2, W/2)
        x = torch.cat([x, s2], dim=1)  # (B, 256, H/2, W/2)
        x = self.dec1(x)  # (B, 128, H/2, W/2)

        # Up-sample 2  +  skip from encoder stage 1
        x = self.up2(x)  # (B, 64,  H,   W)
        x = torch.cat([x, s1], dim=1)  # (B, 128, H,   W)
        x = self.dec2(x)  # (B, 64,  H,   W)

        return x


# ──────────────────────────────────────────────────────────────────────────────
# Full U-Net model
# ──────────────────────────────────────────────────────────────────────────────


class UNet(nn.Module):
    """
    Binary U-Net for foreground/background segmentation.

    Forward pass:
        x          – (B, 3, H, W) input image batch
    Returns:
        (B, 1, H, W) sigmoid output in [0, 1]
    """

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.bottleneck = Bottleneck()
        self.decoder = Decoder()
        # Output head: reduce to 1 channel and squash to [0, 1]
        self.head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encoder
        x, s1, s2 = self.encoder(x)
        # Bottleneck
        x = self.bottleneck(x)
        # Decoder
        x = self.decoder(x, s2, s1)
        # Output head
        return self.head(x)


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check when run directly
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = UNet()
    dummy = torch.randn(2, 3, 256, 256)
    output = model(dummy)
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {output.shape}")  # expected: (2, 1, 256, 256)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
