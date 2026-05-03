import math

import torch
import torch.nn.functional as F
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """連続時刻 t in [0, 1] をベクトルに変換する層。

    DiffusionでもFlow Matchingでも同じモデルを使えるように、
    時刻は整数ではなく [0, 1] のfloatとして扱う。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2:
            t = t.squeeze(1)

        half_dim = self.dim // 2
        device = t.device

        # Transformer系でよく使われるsin/cosの時間埋め込み。
        frequencies = torch.exp(-math.log(10_000) * torch.arange(half_dim, device=device) / max(half_dim - 1, 1))
        angles = t[:, None] * frequencies[None, :] * 1000.0
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return embedding


class ResidualBlock(nn.Module):
    """時間埋め込みを足し込むシンプルなResidual Block。"""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_channels)

        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        # time_emb: [B, C] -> [B, C, 1, 1]
        h = h + self.time_proj(time_emb)[:, :, None, None]

        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class TimeConditionedUNet(nn.Module):
    """MNIST用の小さな時間条件付きU-Net。

    入力:
        x: [B, 1, 28, 28]
        t: [B] か [B, 1]。値域は [0, 1]

    出力:
        [B, 1, 28, 28]

    Diffusionでは「ノイズ予測」、Flow Matchingでは「速度場予測」として同じ出力を使う。
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 64, time_dim: int = 256):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.block1 = ResidualBlock(base_channels, base_channels, time_dim)
        self.down1 = ResidualBlock(base_channels, base_channels * 2, time_dim)

        self.block2 = ResidualBlock(base_channels * 2, base_channels * 2, time_dim)
        self.down2 = ResidualBlock(base_channels * 2, base_channels * 4, time_dim)

        self.middle = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)

        self.up2 = ResidualBlock(base_channels * 4 + base_channels * 2, base_channels * 2, time_dim)
        self.up1 = ResidualBlock(base_channels * 2 + base_channels, base_channels, time_dim)

        self.out_norm = nn.GroupNorm(num_groups=8, num_channels=base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(t)

        x = self.in_conv(x)

        skip1 = self.block1(x, time_emb)  # [B, C, 28, 28]
        x = F.avg_pool2d(skip1, kernel_size=2)  # [B, C, 14, 14]
        x = self.down1(x, time_emb)

        skip2 = self.block2(x, time_emb)  # [B, 2C, 14, 14]
        x = F.avg_pool2d(skip2, kernel_size=2)  # [B, 2C, 7, 7]
        x = self.down2(x, time_emb)

        x = self.middle(x, time_emb)

        x = F.interpolate(x, size=skip2.shape[-2:], mode="nearest")
        x = torch.cat([x, skip2], dim=1)
        x = self.up2(x, time_emb)

        x = F.interpolate(x, size=skip1.shape[-2:], mode="nearest")
        x = torch.cat([x, skip1], dim=1)
        x = self.up1(x, time_emb)

        return self.out_conv(F.silu(self.out_norm(x)))
