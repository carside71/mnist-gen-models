import torch
import torch.nn.functional as F
from torch import nn


def flow_matching_loss(
    model: nn.Module,
    x_data: torch.Tensor,
    labels: torch.Tensor | None = None,
    p_uncond: float = 0.1,
) -> torch.Tensor:
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

    if labels is not None and getattr(model, "label_emb", None) is not None:
        drop_mask = torch.rand(batch_size, device=device) < p_uncond
        y = torch.where(drop_mask, torch.full_like(labels, model.null_label_idx), labels)
        predicted_velocity = model(x_t, t, y)
    else:
        predicted_velocity = model(x_t, t)

    return F.mse_loss(predicted_velocity, target_velocity)


@torch.no_grad()
def sample_flow(
    model: nn.Module,
    shape: tuple[int, int, int, int],
    device: torch.device,
    steps: int = 100,
    labels: torch.Tensor | None = None,
    guidance_scale: float = 0.0,
) -> torch.Tensor:
    """学習した速度場をEuler法で積分して画像を生成する。CFG対応。"""

    model.eval()

    x = torch.randn(shape, device=device)
    dt = 1.0 / steps

    use_cfg = labels is not None and guidance_scale > 0.0 and getattr(model, "label_emb", None) is not None
    null_y = torch.full_like(labels, model.null_label_idx) if use_cfg else None

    for i in range(steps):
        t = torch.full((shape[0],), i / steps, device=device)
        if use_cfg:
            v_cond = model(x, t, labels)
            v_uncond = model(x, t, null_y)
            velocity = (1.0 + guidance_scale) * v_cond - guidance_scale * v_uncond
        else:
            velocity = model(x, t, labels)
        x = x + dt * velocity

    return x.clamp(-1.0, 1.0)
