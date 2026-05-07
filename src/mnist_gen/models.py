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

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        time_dim: int = 256,
        num_classes: int = 0,
        depth: int = 2,
        channel_mults: tuple[int, ...] | None = None,
    ):
        super().__init__()

        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        if channel_mults is None:
            channel_mults = tuple(2**i for i in range(depth + 1))
        elif len(channel_mults) != depth + 1:
            raise ValueError(f"channel_mults must have length depth+1 ({depth + 1}), got {len(channel_mults)}")

        self.depth = depth
        self.channel_mults = tuple(channel_mults)

        self.num_classes = num_classes
        self.null_label_idx = num_classes  # 無条件トークンとして使うインデックス

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        if num_classes > 0:
            # +1 は CFG 用の "無条件" 埋め込み
            self.label_emb = nn.Embedding(num_classes + 1, time_dim)
            self.label_mlp = nn.Sequential(
                nn.Linear(time_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            self.label_emb = None
            self.label_mlp = None

        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        channels = [base_channels * m for m in self.channel_mults]

        self.skip_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        for i in range(depth):
            self.skip_blocks.append(ResidualBlock(channels[i], channels[i], time_dim))
            self.down_blocks.append(ResidualBlock(channels[i], channels[i + 1], time_dim))

        self.middle = ResidualBlock(channels[-1], channels[-1], time_dim)

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(depth)):
            # skip との concat 後を channels[i] に縮約
            self.up_blocks.append(ResidualBlock(channels[i + 1] + channels[i], channels[i], time_dim))

        self.out_norm = nn.GroupNorm(num_groups=8, num_channels=base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        time_emb = self.time_mlp(t)

        if self.label_emb is not None:
            if y is None:
                y = torch.full((x.size(0),), self.null_label_idx, device=x.device, dtype=torch.long)
            time_emb = time_emb + self.label_mlp(self.label_emb(y))

        x = self.in_conv(x)

        skips: list[torch.Tensor] = []
        for skip_block, down_block in zip(self.skip_blocks, self.down_blocks, strict=True):
            skip = skip_block(x, time_emb)
            skips.append(skip)
            x = F.avg_pool2d(skip, kernel_size=2)
            x = down_block(x, time_emb)

        x = self.middle(x, time_emb)

        for up_block, skip in zip(self.up_blocks, reversed(skips), strict=True):
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = up_block(x, time_emb)

        return self.out_conv(F.silu(self.out_norm(x)))
