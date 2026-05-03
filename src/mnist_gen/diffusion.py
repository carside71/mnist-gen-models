from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DiffusionSchedule:
    """DDPMで使う係数をまとめたクラス。"""

    timesteps: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor

    @classmethod
    def create(
        cls,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | str = "cpu",
    ) -> "DiffusionSchedule":
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        return cls(
            timesteps=timesteps,
            betas=betas,
            alphas=alphas,
            alpha_bars=alpha_bars,
        )


def extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """時刻ごとの係数を [B, 1, 1, 1] の形にして取り出す。"""

    out = values.gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(x_shape) - 1)))


def q_sample(
    x_start: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """元画像 x_start に時刻 t の量だけノイズを足す。"""

    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alpha_bar = torch.sqrt(extract(schedule.alpha_bars, t, x_start.shape))
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - extract(schedule.alpha_bars, t, x_start.shape))

    x_t = sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise
    return x_t, noise


def diffusion_loss(
    model: nn.Module,
    x_start: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    """DDPMの基本的なノイズ予測損失。"""

    batch_size = x_start.size(0)
    device = x_start.device

    t = torch.randint(0, schedule.timesteps, (batch_size,), device=device)
    x_t, noise = q_sample(x_start, t, schedule)

    # モデルには [0, 1] に正規化した時刻を渡す。
    t_normalized = t.float() / (schedule.timesteps - 1)
    predicted_noise = model(x_t, t_normalized)

    return F.mse_loss(predicted_noise, noise)


@torch.no_grad()
def sample_ddpm(
    model: nn.Module,
    schedule: DiffusionSchedule,
    shape: tuple[int, int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """DDPMの逆過程で画像を生成する。"""

    model.eval()
    x = torch.randn(shape, device=device)

    for i in reversed(range(schedule.timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        t_normalized = t.float() / (schedule.timesteps - 1)

        predicted_noise = model(x, t_normalized)

        beta_t = extract(schedule.betas, t, x.shape)
        alpha_t = extract(schedule.alphas, t, x.shape)
        alpha_bar_t = extract(schedule.alpha_bars, t, x.shape)

        # DDPMの平均。モデルが予測したノイズを使って1ステップ前へ戻る。
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise
        )

        if i == 0:
            x = mean
        else:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * noise

    return x.clamp(-1.0, 1.0)
