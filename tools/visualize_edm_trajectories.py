"""EDM の生成軌道と前向き拡散軌道を PCA で可視化するツール。"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mnist_gen.data import DATASET_SPECS, get_raw_dataset
from mnist_gen.edm import f_edm, karras_sigmas
from mnist_gen.models import TimeConditionedUNet
from mnist_gen.utils import get_device, load_model_weights, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize EDM trajectories via PCA.")
    parser.add_argument("--checkpoint", type=str, default="/workspace/outputs/edm/exp_01/checkpoints/best.pt")
    parser.add_argument("--out-path", type=str, default="/workspace/outputs/edm/trajectories/edm_traj.png")
    parser.add_argument("--dataset", type=str, default=None, choices=[None, "mnist", "cifar10"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--num-gen", type=int, default=16)
    parser.add_argument("--num-data", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=18, help="EDM Heun の離散ステップ数。")
    parser.add_argument("--sigma-min", type=float, default=None)
    parser.add_argument("--sigma-max", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--sigma-data", type=float, default=None)
    parser.add_argument("--labels", type=int, nargs="+", default=None)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--dim", type=int, choices=[2, 3], default=2)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model(
    checkpoint_path: str,
    device: torch.device,
    base_channels_arg: int | None,
    num_classes_arg: int | None,
    dataset_arg: str | None,
    depth_arg: int | None,
) -> tuple[TimeConditionedUNet, int, str, int, int, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("model_config", {})
    edm_config = checkpoint.get("edm_config", {})
    base_channels = base_channels_arg or model_config.get("base_channels", 64)
    depth = depth_arg or model_config.get("depth", 2)
    num_classes = num_classes_arg if num_classes_arg is not None else model_config.get("num_classes", 0)
    dataset = dataset_arg or model_config.get("dataset", "mnist")
    spec = DATASET_SPECS[dataset]
    in_channels = model_config.get("in_channels", spec["in_channels"])
    image_size = model_config.get("image_size", spec["image_size"])
    model = TimeConditionedUNet(
        in_channels=in_channels,
        base_channels=base_channels,
        num_classes=num_classes,
        depth=depth,
    ).to(device)
    load_model_weights(model, checkpoint_path, device)
    model.eval()
    return model, num_classes, dataset, in_channels, image_size, edm_config


def assign_labels(
    target_labels: list[int] | None, n: int, device: torch.device, num_classes: int
) -> torch.Tensor | None:
    if num_classes <= 0 or target_labels is None:
        return None
    labels = [target_labels[i % len(target_labels)] for i in range(n)]
    return torch.tensor(labels, device=device, dtype=torch.long)


def _normalize_snapshot(x: torch.Tensor, sigma_val: float, sigma_data: float) -> np.ndarray:
    """EDM の c_in(σ) = 1/√(σ²+σ_d²) で点をスケール正規化して flatten。

    σ ∈ [0.002, 80] に渡って点のスケールが大きく変わるため、生の値で PCA を取ると
    主成分が σ の大きさ軸に固定されて構造が潰れる。c_in をかけて単位スケールに揃える。
    """
    c_in = 1.0 / ((sigma_val * sigma_val + sigma_data * sigma_data) ** 0.5)
    return (x * c_in).detach().cpu().reshape(x.size(0), -1).numpy()


@torch.no_grad()
def generate_trajectories(
    model: TimeConditionedUNet,
    sigma_grid: torch.Tensor,
    sigma_data: float,
    sigma_min: float,
    n: int,
    labels: torch.Tensor | None,
    guidance_scale: float,
    device: torch.device,
    in_channels: int,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """EDM Heun 逆過程の軌道を [n, S+1, 784] で返す。時間方向はノイズ→画像。
    各点は c_in(σ) で正規化してから記録する。"""
    sigmas = sigma_grid.flip(0)  # 降順 [σ_max, ..., σ_min]
    use_cfg = (
        labels is not None
        and guidance_scale > 0.0
        and getattr(model, "label_emb", None) is not None
    )
    null_y = torch.full_like(labels, model.null_label_idx) if use_cfg else None

    def denoise(x_in: torch.Tensor, sigma_in: torch.Tensor) -> torch.Tensor:
        if use_cfg:
            d_cond = f_edm(model, x_in, sigma_in, sigma_data, sigma_min, labels)
            d_uncond = f_edm(model, x_in, sigma_in, sigma_data, sigma_min, null_y)
            return (1.0 + guidance_scale) * d_cond - guidance_scale * d_uncond
        return f_edm(model, x_in, sigma_in, sigma_data, sigma_min, labels)

    x = sigmas[0] * torch.randn(n, in_channels, image_size, image_size, device=device)
    snapshots = [_normalize_snapshot(x, sigmas[0].item(), sigma_data)]

    for i in range(len(sigmas) - 1):
        sigma_cur = torch.full((n,), sigmas[i].item(), device=device)
        sigma_next = torch.full((n,), sigmas[i + 1].item(), device=device)
        sigma_cur_b = sigma_cur.view(-1, 1, 1, 1)
        sigma_next_b = sigma_next.view(-1, 1, 1, 1)

        d = (x - denoise(x, sigma_cur)) / sigma_cur_b
        x_euler = x + (sigma_next_b - sigma_cur_b) * d

        if sigmas[i + 1].item() <= sigma_min:
            x = x_euler
        else:
            d_prime = (x_euler - denoise(x_euler, sigma_next)) / sigma_next_b
            x = x + 0.5 * (sigma_next_b - sigma_cur_b) * (d + d_prime)

        snapshots.append(_normalize_snapshot(x, sigmas[i + 1].item(), sigma_data))

    traj_arr = np.stack(snapshots, axis=1)  # [n, S+1, D]
    label_arr = labels.detach().cpu().numpy() if labels is not None else None
    return traj_arr, label_arr


def sample_data_images(
    dataset_name: str, data_dir: str, target_labels: list[int] | None, m: int, seed: int
) -> tuple[torch.Tensor, np.ndarray]:
    dataset = get_raw_dataset(dataset_name, data_dir, train=True)
    targets = torch.as_tensor(dataset.targets)
    if target_labels is not None:
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for lb in target_labels:
            mask |= targets == lb
        indices = torch.nonzero(mask, as_tuple=False).flatten()
    else:
        indices = torch.arange(len(dataset))

    if len(indices) == 0:
        raise ValueError("指定ラベルに該当する画像がデータセットに存在しません。")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(indices), size=m, replace=(m > len(indices)))
    chosen_indices = indices[torch.from_numpy(chosen)]

    images, labels = [], []
    for idx in chosen_indices.tolist():
        img, lb = dataset[idx]
        images.append(img)
        labels.append(lb)
    x1 = torch.stack(images, dim=0)
    return x1, np.array(labels)


def make_data_trajectories(
    x1: torch.Tensor, sigma_grid: torch.Tensor, sigma_data: float, device: torch.device
) -> np.ndarray:
    """前向き拡散による軌道を [m, S+1, D] で返す。時間方向はノイズ→画像。

    EDM 形式: x_σ = x + σ·z。固定ノイズで σ を降順に適用。
    各点は c_in(σ) で正規化してから記録する（generate_trajectories と同じ規約）。
    """
    sigmas = sigma_grid.flip(0)  # 降順 [σ_max, ..., σ_min]
    x1_dev = x1.to(device)
    fixed_noise = torch.randn_like(x1_dev)

    out = []
    for s in sigmas.tolist():
        x_sigma = x1_dev + float(s) * fixed_noise
        out.append(_normalize_snapshot(x_sigma, float(s), sigma_data))
    return np.stack(out, axis=1)


def fit_pca_and_plot(
    gen_traj: np.ndarray,
    gen_labels: np.ndarray | None,
    data_traj: np.ndarray,
    data_labels: np.ndarray,
    dim: int,
    pca_dim: int,
    out_path: Path,
) -> None:
    n, T, D = gen_traj.shape
    m = data_traj.shape[0]

    noise_feat = data_traj[:, 0, :]
    real_feat = data_traj[:, -1, :]
    fit_feat = np.concatenate([noise_feat, real_feat], axis=0)
    NOISE_LABEL = -1
    fit_labels = np.concatenate(
        [
            np.full(m, NOISE_LABEL),
            data_labels if data_labels is not None else np.full(m, 0),
        ]
    )

    pca_n = max(dim, min(pca_dim, fit_feat.shape[0], fit_feat.shape[1]))
    pca = PCA(n_components=pca_n)
    pca.fit(fit_feat)

    all_states = np.concatenate([gen_traj.reshape(n * T, D), data_traj.reshape(m * T, D)], axis=0)
    all_pca = pca.transform(all_states)

    unique = np.unique(fit_labels)
    lda_used = False
    if len(unique) >= dim + 1:
        lda = LinearDiscriminantAnalysis(n_components=dim)
        lda.fit(pca.transform(fit_feat), fit_labels)
        proj = lda.transform(all_pca)
        explained = lda.explained_variance_ratio_
        lda_used = True
    else:
        proj = all_pca[:, :dim]
        explained = pca.explained_variance_ratio_[:dim]

    gen_proj = proj[: n * T].reshape(n, T, dim)
    data_proj = proj[n * T :].reshape(m, T, dim)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d") if dim == 3 else fig.add_subplot(111)

    all_labels: list[int] = []
    if gen_labels is not None:
        all_labels.extend(gen_labels.tolist())
    if data_labels is not None:
        all_labels.extend(data_labels.tolist())
    unique_labels = sorted(set(all_labels)) if all_labels else []
    cmap = plt.get_cmap("tab10")
    label_to_color = {lb: cmap(i % 10) for i, lb in enumerate(unique_labels)}
    default_color = "tab:gray"

    def plot_line(ax, pts, color, alpha, linestyle, label=None, marker_alpha=None):
        ma = alpha if marker_alpha is None else marker_alpha
        if dim == 3:
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                color=color,
                alpha=alpha,
                linewidth=1.0,
                linestyle=linestyle,
                label=label,
            )
            ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color=color, marker="o", s=20, alpha=ma)
            ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color=color, marker="*", s=60, alpha=ma)
        else:
            ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=alpha, linewidth=1.0, linestyle=linestyle, label=label)
            ax.scatter(pts[0, 0], pts[0, 1], color=color, marker="o", s=20, alpha=ma)
            ax.scatter(pts[-1, 0], pts[-1, 1], color=color, marker="*", s=60, alpha=ma)

    seen_label_keys: set[tuple[int, str]] = set()

    def legend_label(lb: int | None, kind: str) -> str | None:
        key = (-1 if lb is None else int(lb), kind)
        if key in seen_label_keys:
            return None
        seen_label_keys.add(key)
        suffix = "gen" if kind == "gen" else "data"
        return f"label {lb} ({suffix})" if lb is not None else f"unlabeled ({suffix})"

    for i in range(n):
        lb = int(gen_labels[i]) if gen_labels is not None else None
        color = label_to_color.get(lb, default_color) if lb is not None else default_color
        plot_line(ax, gen_proj[i], color=color, alpha=0.85, linestyle="-", label=legend_label(lb, "gen"))
    for j in range(m):
        lb = int(data_labels[j]) if data_labels is not None else None
        color = label_to_color.get(lb, default_color) if lb is not None else default_color
        plot_line(
            ax, data_proj[j], color=color, alpha=0.25, linestyle="--", label=legend_label(lb, "data"), marker_alpha=0.35
        )

    axis_prefix = "LD" if lda_used else "PC"
    ax.set_xlabel(f"{axis_prefix}1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"{axis_prefix}2 ({explained[1] * 100:.1f}%)")
    if dim == 3:
        ax.set_zlabel(f"{axis_prefix}3 ({explained[2] * 100:.1f}%)")
    method = f"PCA({pca_n})→LDA" if lda_used else "PCA"
    ax.set_title(f"EDM trajectories ({method})\nO=σ_max (noise),  *=σ_min")
    ax.legend(loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    model, num_classes, dataset_name, in_channels, image_size, edm_config = load_model(
        args.checkpoint, device, args.base_channels, args.num_classes, args.dataset, args.depth
    )
    data_dir = args.data_dir or f"/workspace/datasets/{dataset_name}"

    sigma_min = args.sigma_min if args.sigma_min is not None else edm_config.get("sigma_min", 0.002)
    sigma_max = args.sigma_max if args.sigma_max is not None else edm_config.get("sigma_max", 80.0)
    rho = args.rho if args.rho is not None else edm_config.get("rho", 7.0)
    sigma_data = args.sigma_data if args.sigma_data is not None else edm_config.get("sigma_data", 0.5)

    sigma_grid = karras_sigmas(args.num_steps, sigma_min, sigma_max, rho, device)

    target_labels = args.labels
    if target_labels is not None and num_classes > 0:
        for lb in target_labels:
            if not 0 <= lb < num_classes:
                raise ValueError(f"--labels の値は 0..{num_classes - 1} の範囲で指定してください: {lb}")

    gen_label_tensor = assign_labels(target_labels, args.num_gen, device, num_classes)
    gen_traj, gen_labels = generate_trajectories(
        model,
        sigma_grid,
        sigma_data,
        sigma_min,
        args.num_gen,
        gen_label_tensor,
        args.guidance_scale,
        device,
        in_channels,
        image_size,
    )
    print(f"generated trajectories: {gen_traj.shape}")

    x1, data_labels = sample_data_images(dataset_name, data_dir, target_labels, args.num_data, args.seed)
    data_traj = make_data_trajectories(x1, sigma_grid, sigma_data, device)
    print(f"data trajectories: {data_traj.shape}")

    out_path = Path(args.out_path)
    out_path = out_path.with_name(f"{out_path.stem}_guided{args.guidance_scale}_dim{args.dim}{out_path.suffix}")
    fit_pca_and_plot(gen_traj, gen_labels, data_traj, data_labels, args.dim, args.pca_dim, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
