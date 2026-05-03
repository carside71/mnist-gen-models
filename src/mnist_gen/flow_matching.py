import torch
import torch.nn.functional as F
from torch import nn


def flow_matching_loss(model: nn.Module, x_data: torch.Tensor) -> torch.Tensor:
    """線形パスを使うFlow Matchingの損失。

    x0: 標準正規分布からのノイズ
    x1: データ画像
    t : [0, 1] の一様乱数

    x_t = (1 - t) x0 + t x1
    v   = x1 - x0

    モデルは x_t と t から速度 v を予測する。
    """

    batch_size = x_data.size(0)
    device = x_data.device

    x0 = torch.randn_like(x_data)
    x1 = x_data

    t = torch.rand(batch_size, device=device)
    t_view = t[:, None, None, None]

    x_t = (1.0 - t_view) * x0 + t_view * x1
    target_velocity = x1 - x0

    predicted_velocity = model(x_t, t)

    return F.mse_loss(predicted_velocity, target_velocity)


@torch.no_grad()
def sample_flow(
    model: nn.Module,
    shape: tuple[int, int, int, int],
    device: torch.device,
    steps: int = 100,
) -> torch.Tensor:
    """学習した速度場をEuler法で積分して画像を生成する。"""

    model.eval()

    x = torch.randn(shape, device=device)
    dt = 1.0 / steps

    for i in range(steps):
        t = torch.full((shape[0],), i / steps, device=device)
        velocity = model(x, t)
        x = x + dt * velocity

    return x.clamp(-1.0, 1.0)
