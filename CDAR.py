""" Cascaded Dual Attention Refinement"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LocalSpatialAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels // reduction, 1)
        self.conv2 = nn.Conv2d(channels // reduction, channels, 1)

    def forward(self, x):
        # Local spatial attention with 3x3 convolution
        attn = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        attn = self.conv1(attn)
        attn = F.relu(attn)
        attn = self.conv2(attn)
        attn = torch.sigmoid(attn)
        return x * attn


class GlobalContextBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Global context modeling
        y = self.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialRefinementModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        return residual + out


class CascadeUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local_attn = LocalSpatialAttention(channels)
        self.global_context = GlobalContextBlock(channels)
        self.fusion = nn.Conv2d(channels * 2, channels, 1)
        self.refinement = SpatialRefinementModule(channels)

    def forward(self, x):
        local_feat = self.local_attn(x)
        global_feat = self.global_context(x)
        # Feature fusion
        fused = self.fusion(torch.cat([local_feat, global_feat], dim=1))
        # Spatial refinement
        out = self.refinement(fused)
        return out


class ProgressiveUpsampling(nn.Module):
    def __init__(self, channels, scale):
        super().__init__()
        self.scale = scale
        self.ups = nn.ModuleList()

        for _ in range(int(math.log2(scale))):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels, 4 * channels, 3, 1, 1),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )

    def forward(self, x):
        for up in self.ups:
            x = up(x)
        return x


class CDAR(nn.Module):
    def __init__(self, in_channels=3, channels=64, num_cascade_units=6, scale=4):
        super().__init__()

        # Shallow feature extraction
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Deep feature extraction with cascade units
        self.cascade_units = nn.ModuleList([
            CascadeUnit(channels) for _ in range(num_cascade_units)
        ])

        # Global residual learning
        self.global_residual = nn.Conv2d(channels, channels, 3, 1, 1)

        # Progressive upsampling
        self.upsampling = ProgressiveUpsampling(channels, scale)

        # Final reconstruction
        self.tail = nn.Conv2d(channels, in_channels, 3, 1, 1)

    def forward(self, x):
        # Initial feature extraction
        head = self.head(x)

        # Deep feature extraction
        feat = head
        for unit in self.cascade_units:
            feat = unit(feat) + head

        # Global residual connection
        feat = self.global_residual(feat) + head

        # Upsampling and reconstruction
        feat = self.upsampling(feat)
        out = self.tail(feat)

        return out
