import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """A standard block of two convolutional layers used throughout the U-Net."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DualBranchFusion(nn.Module):
    def __init__(self, num_classes=2):
        """
        Dual-Encoder U-Net Architecture:
        - Optical Branch: Learns high-resolution edges, color, and multispectral reflectance.
        - SAR Branch: Learns structural density, geometry, and surface roughness.
        - Decoder: Fuses both feature sets using skip connections for precise boundary segmentation.
        """
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        # 1. Optical Encoder Branch
        self.opt_enc1 = DoubleConv(3, 32)
        self.opt_enc2 = DoubleConv(32, 64)
        self.opt_enc3 = DoubleConv(64, 128)

        # 2. SAR Encoder Branch
        self.sar_enc1 = DoubleConv(2, 32)
        self.sar_enc2 = DoubleConv(32, 64)
        self.sar_enc3 = DoubleConv(64, 128)

        # 3. Fusion Bottleneck (Concatenates both branches at their lowest resolution)
        self.bottleneck = DoubleConv(128 + 128, 256)

        # 4. Decoder (Upsampling and Skip Connections)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128 + 128 + 128, 128)  # fused bottleneck + opt3 + sar3

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64 + 64 + 64, 64)  # fused dec3 + opt2 + sar2

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32 + 32 + 32, 32)  # fused dec2 + opt1 + sar1

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, optical, sar):
        # -- Encoding Stage 1 (Output: 256x256) --
        opt1 = self.opt_enc1(optical)
        sar1 = self.sar_enc1(sar)

        # -- Encoding Stage 2 (Output: 128x128) --
        opt2 = self.opt_enc2(self.pool(opt1))
        sar2 = self.sar_enc2(self.pool(sar1))

        # -- Encoding Stage 3 (Output: 64x64) --
        opt3 = self.opt_enc3(self.pool(opt2))
        sar3 = self.sar_enc3(self.pool(sar2))

        # -- Bottleneck (Output: 32x32) --
        # Pool stage 3 and fuse the features
        fused_bottleneck = torch.cat([self.pool(opt3), self.pool(sar3)], dim=1)
        b = self.bottleneck(fused_bottleneck)

        # -- Decoding Stage 3 (Upsample to 64x64) --
        up3 = self.up3(b)
        # Skip connection: inject the high-res Optical and SAR features from Stage 3
        up3_fused = torch.cat([up3, opt3, sar3], dim=1)
        dec3 = self.dec3(up3_fused)

        # -- Decoding Stage 2 (Upsample to 128x128) --
        up2 = self.up2(dec3)
        up2_fused = torch.cat([up2, opt2, sar2], dim=1)
        dec2 = self.dec2(up2_fused)

        # -- Decoding Stage 1 (Upsample to 256x256) --
        up1 = self.up1(dec2)
        up1_fused = torch.cat([up1, opt1, sar1], dim=1)
        dec1 = self.dec1(up1_fused)

        # -- Final Pixel Classification --
        return self.final_conv(dec1)
