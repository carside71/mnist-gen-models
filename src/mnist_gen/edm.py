"""EDM (Karras et al. 2022) 流の連続 σ-space 拡散ユーティリティ。

教師訓練 (train_edm.py)、教師サンプラ (sample_edm.py)、
consistency distillation (train_consistency_distillation.py)、
CD サンプラ (sample_consistency.py) から共有する。
"""

import torch
from torch import nn


def karras_sigmas(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    """Karras (ρ=7) 風 σ 等間隔離散化。

    返り値は **昇順** [σ_min, ..., σ_max] の長さ num_steps+1。
    σ_i = (σ_min^(1/ρ) + i/N · (σ_max^(1/ρ) − σ_min^(1/ρ)))^ρ, i=0..N
    """
    i = torch.arange(num_steps + 1, device=device).float() / num_steps
    return (sigma_min ** (1 / rho) + i * (sigma_max ** (1 / rho) - sigma_min ** (1 / rho))) ** rho


def edm_preconditioning(
    sigma: torch.Tensor,
    sigma_data: float,
    sigma_min: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """EDM 前処理係数 (c_skip, c_out, c_in, c_noise)。

    境界条件: σ=σ_min で c_skip=1, c_out=0 → f(x, σ_min) = x。
    """
    sd2 = sigma_data * sigma_data
    c_skip = sd2 / ((sigma - sigma_min) ** 2 + sd2)
    c_out = sigma_data * (sigma - sigma_min) / torch.sqrt(sigma * sigma + sd2)
    c_in = 1.0 / torch.sqrt(sigma * sigma + sd2)
    c_noise = 0.25 * torch.log(sigma)
    return c_skip, c_out, c_in, c_noise


def f_edm(
    model: nn.Module,
    x: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float,
    sigma_min: float,
    labels: torch.Tensor | None,
) -> torch.Tensor:
    """EDM 前処理付きデノイザ D_φ(x, σ) = c_skip x + c_out F_φ(c_in x; c_noise)。

    x: [B, C, H, W], sigma: [B] か [B, 1, 1, 1]。
    """
    sigma_b = sigma.view(-1, 1, 1, 1)
    c_skip, c_out, c_in, c_noise = edm_preconditioning(sigma_b, sigma_data, sigma_min)
    out = model(c_in * x, c_noise.flatten(), labels)
    return c_skip * x + c_out * out


def sample_log_normal_sigma(
    batch_size: int,
    p_mean: float,
    p_std: float,
    device: torch.device,
) -> torch.Tensor:
    """log σ ~ N(P_mean, P_std²) の σ サンプリング (EDM 標準, Karras 2022)。"""
    return torch.exp(torch.randn(batch_size, device=device) * p_std + p_mean)


def edm_loss(
    model: nn.Module,
    x: torch.Tensor,
    sigma_data: float,
    sigma_min: float,
    p_mean: float,
    p_std: float,
    labels: torch.Tensor | None = None,
    p_uncond: float = 0.0,
) -> torch.Tensor:
    """EDM の重み付き x₀ 予測損失 (Karras et al. 2022 Eq.5)。

    L = E[ λ(σ) · ‖D_φ(x + σ·z, σ) − x‖² ], λ(σ) = (σ²+σ_d²)/(σ·σ_d)²。
    CFG 学習用に確率 p_uncond でラベルを null トークンへ置き換える。
    """
    batch_size = x.size(0)
    device = x.device
    sigma = sample_log_normal_sigma(batch_size, p_mean, p_std, device)
    z = torch.randn_like(x)
    x_sigma = x + sigma.view(-1, 1, 1, 1) * z

    if labels is not None and getattr(model, "label_emb", None) is not None and p_uncond > 0.0:
        drop = torch.rand(batch_size, device=device) < p_uncond
        y = torch.where(drop, torch.full_like(labels, model.null_label_idx), labels)
    else:
        y = labels

    D = f_edm(model, x_sigma, sigma, sigma_data, sigma_min, y)
    weight = (sigma * sigma + sigma_data * sigma_data) / (sigma * sigma_data) ** 2
    return (weight.view(-1, 1, 1, 1) * (D - x) ** 2).mean()


@torch.no_grad()
def heun_sample(
    model: nn.Module,
    shape: tuple[int, int, int, int],
    device: torch.device,
    sigma_grid: torch.Tensor,
    sigma_data: float,
    sigma_min: float,
    labels: torch.Tensor | None = None,
    guidance_scale: float = 0.0,
    null_label_idx: int | None = None,
) -> torch.Tensor:
    """EDM Heun 2nd-order deterministic サンプラ (Karras 2022 Algorithm 1 簡略版)。

    sigma_grid は昇順 [σ_min, ..., σ_max]。逆順に走って σ_min まで降下。
    最終ステップ (σ_next ≤ σ_min) は Heun 補正を省いて Euler のみ（D の評価が
    定義上 σ=σ_min で c_out=0 になり 0 除算になりかねないため）。
    """
    model.eval()
    batch_size = shape[0]
    sigmas = sigma_grid.flip(0)  # 降順 [σ_max, ..., σ_min]
    x = sigmas[0] * torch.randn(shape, device=device)

    use_cfg = (
        labels is not None
        and guidance_scale > 0.0
        and null_label_idx is not None
        and getattr(model, "label_emb", None) is not None
    )
    null_y = torch.full_like(labels, null_label_idx) if use_cfg else None

    def denoise(x_in: torch.Tensor, sigma_in: torch.Tensor) -> torch.Tensor:
        if use_cfg:
            d_cond = f_edm(model, x_in, sigma_in, sigma_data, sigma_min, labels)
            d_uncond = f_edm(model, x_in, sigma_in, sigma_data, sigma_min, null_y)
            return (1.0 + guidance_scale) * d_cond - guidance_scale * d_uncond
        return f_edm(model, x_in, sigma_in, sigma_data, sigma_min, labels)

    for i in range(len(sigmas) - 1):
        sigma_cur = torch.full((batch_size,), sigmas[i].item(), device=device)
        sigma_next = torch.full((batch_size,), sigmas[i + 1].item(), device=device)
        sigma_cur_b = sigma_cur.view(-1, 1, 1, 1)
        sigma_next_b = sigma_next.view(-1, 1, 1, 1)

        d = (x - denoise(x, sigma_cur)) / sigma_cur_b
        x_euler = x + (sigma_next_b - sigma_cur_b) * d

        if sigmas[i + 1].item() <= sigma_min:
            x = x_euler
        else:
            d_prime = (x_euler - denoise(x_euler, sigma_next)) / sigma_next_b
            x = x + 0.5 * (sigma_next_b - sigma_cur_b) * (d + d_prime)

    return x.clamp(-1.0, 1.0)
