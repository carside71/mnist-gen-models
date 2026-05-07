"""DDPM の生成軌道と前向き拡散軌道を PCA で可視化するツール。"""

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
from mnist_gen.diffusion import DiffusionSchedule, extract, q_sample
from mnist_gen.models import TimeConditionedUNet
from mnist_gen.utils import get_device, load_model_weights, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize DDPM trajectories via PCA.")
    parser.add_argument("--checkpoint", type=str, default="/workspace/outputs/diffusion/exp_02/checkpoints/best.pt")
    parser.add_argument("--out-path", type=str, default="/workspace/outputs/diffusion/trajectories/diffusion_traj.png")
    parser.add_argument("--dataset", type=str, default=None, choices=[None, "mnist", "cifar10"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--num-gen", type=int, default=16)
    parser.add_argument("--num-data", type=int, default=16)
    parser.add_argument(
        "--steps", type=int, default=50, help="軌道として記録する点数（DDPM の総ステップから等間隔に間引く）。"
    )
    parser.add_argument("--timesteps", type=int, default=1000, help="DDPM スケジュールの総ステップ数。")
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
) -> tuple[TimeConditionedUNet, int, str, int, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("model_config", {})
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
    return model, num_classes, dataset, in_channels, image_size


def assign_labels(
    target_labels: list[int] | None, n: int, device: torch.device, num_classes: int
) -> torch.Tensor | None:
    if num_classes <= 0 or target_labels is None:
        return None
    labels = [target_labels[i % len(target_labels)] for i in range(n)]
    return torch.tensor(labels, device=device, dtype=torch.long)


def _record_indices(total: int, steps: int) -> np.ndarray:
    """0..total を steps+1 点に等間隔サンプリングした整数インデックス。"""
    return np.unique(np.linspace(0, total, steps + 1).round().astype(int))


@torch.no_grad()
def generate_trajectories(
    model: TimeConditionedUNet,
    schedule: DiffusionSchedule,
    n: int,
    steps: int,
    labels: torch.Tensor | None,
    guidance_scale: float,
    device: torch.device,
    in_channels: int,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """DDPM 逆過程の軌道を [n, S+1, 784] で返す。時間方向はノイズ→画像。"""
    T = schedule.timesteps
    rec = _record_indices(T, steps)  # 0..T を含む昇順
    rec_set = set(int(v) for v in rec.tolist())

    x = torch.randn(n, in_channels, image_size, image_size, device=device)
    use_cfg = labels is not None and guidance_scale > 0.0 and getattr(model, "label_emb", None) is not None
    null_y = torch.full_like(labels, model.null_label_idx) if use_cfg else None

    # 「残りステップ数」が rec に含まれる時点でスナップショットを取る。
    # 初期 x（=純ノイズ, 残り T ステップ）と最終 x（残り 0）が必ず含まれる。
    snapshots: dict[int, np.ndarray] = {}
    if T in rec_set:
        snapshots[T] = x.detach().cpu().reshape(n, -1).numpy()

    for i in reversed(range(T)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        t_norm = t.float() / (T - 1)

        if use_cfg:
            eps_cond = model(x, t_norm, labels)
            eps_uncond = model(x, t_norm, null_y)
            predicted_noise = (1.0 + guidance_scale) * eps_cond - guidance_scale * eps_uncond
        else:
            predicted_noise = model(x, t_norm, labels)

        beta_t = extract(schedule.betas, t, x.shape)
        alpha_t = extract(schedule.alphas, t, x.shape)
        alpha_bar_t = extract(schedule.alpha_bars, t, x.shape)

        mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise)
        if i == 0:
            x = mean
        else:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * noise

        remaining = i  # ステップ後に残るステップ数
        if remaining in rec_set:
            snapshots[remaining] = x.detach().cpu().reshape(n, -1).numpy()

    # ノイズ→画像 の順に並べる: 残り T → 残り 0
    ordered = [snapshots[int(v)] for v in sorted(rec_set, reverse=True)]
    traj_arr = np.stack(ordered, axis=1)  # [n, S+1, 784]
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
    x1: torch.Tensor, schedule: DiffusionSchedule, steps: int, device: torch.device
) -> np.ndarray:
    """前向き拡散による軌道を [m, S+1, 784] で返す。時間方向はノイズ→画像。"""
    T = schedule.timesteps
    rec = _record_indices(T, steps)
    # 残りステップ数の降順 = ノイズ→画像
    rec_desc = sorted({int(v) for v in rec.tolist()}, reverse=True)

    m = x1.size(0)
    x1_dev = x1.to(device)
    fixed_noise = torch.randn_like(x1_dev)

    out = []
    for r in rec_desc:
        # r は「残りステップ数」 = q_sample に渡す t（0 で原画像、T-1 でほぼ純ノイズ）
        t_idx = min(max(r - 1, 0), T - 1) if r > 0 else 0
        if r == T:
            xt = fixed_noise  # 完全ノイズ点
        else:
            t = torch.full((m,), t_idx, device=device, dtype=torch.long)
            xt, _ = q_sample(x1_dev, t, schedule, noise=fixed_noise)
        out.append(xt.detach().cpu().reshape(m, -1).numpy())
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
    ax.set_title(f"Diffusion (DDPM) trajectories ({method})\nO=t=0 (noise),  *=t=1")
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

    model, num_classes, dataset_name, in_channels, image_size = load_model(
        args.checkpoint, device, args.base_channels, args.num_classes, args.dataset, args.depth
    )
    data_dir = args.data_dir or f"/workspace/datasets/{dataset_name}"
    schedule = DiffusionSchedule.create(timesteps=args.timesteps, device=device)

    target_labels = args.labels
    if target_labels is not None and num_classes > 0:
        for lb in target_labels:
            if not 0 <= lb < num_classes:
                raise ValueError(f"--labels の値は 0..{num_classes - 1} の範囲で指定してください: {lb}")

    gen_label_tensor = assign_labels(target_labels, args.num_gen, device, num_classes)
    gen_traj, gen_labels = generate_trajectories(
        model,
        schedule,
        args.num_gen,
        args.steps,
        gen_label_tensor,
        args.guidance_scale,
        device,
        in_channels,
        image_size,
    )
    print(f"generated trajectories: {gen_traj.shape}")

    x1, data_labels = sample_data_images(dataset_name, data_dir, target_labels, args.num_data, args.seed)
    data_traj = make_data_trajectories(x1, schedule, args.steps, device)
    print(f"data trajectories: {data_traj.shape}")

    out_path = Path(args.out_path)
    out_path = out_path.with_name(f"{out_path.stem}_guided{args.guidance_scale}_dim{args.dim}{out_path.suffix}")
    fit_pca_and_plot(gen_traj, gen_labels, data_traj, data_labels, args.dim, args.pca_dim, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
