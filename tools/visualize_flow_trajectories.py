"""Flow Matching の生成軌道とデータ側直線内挿軌道を PCA で可視化するツール。"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mnist_gen.data import _mnist_transform
from mnist_gen.models import TimeConditionedUNet
from mnist_gen.utils import get_device, load_model_weights, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Flow Matching trajectories via PCA.")
    parser.add_argument("--checkpoint", type=str, default="/workspace/outputs/flow/checkpoints/best.pt")
    parser.add_argument("--out-path", type=str, default="/workspace/outputs/flow/trajectories/flow_traj.png")
    parser.add_argument("--data-dir", type=str, default="/workspace/datasets/mnist")
    parser.add_argument("--num-gen", type=int, default=16)
    parser.add_argument("--num-data", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--labels",
        type=int,
        nargs="+",
        default=None,
        help="対象ラベル（複数可）。例: --labels 1 3 5。省略時は全ラベル。",
    )
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--dim", type=int, choices=[2, 3], default=2)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model(
    checkpoint_path: str, device: torch.device, base_channels_arg: int | None, num_classes_arg: int | None
) -> tuple[TimeConditionedUNet, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("model_config", {})
    base_channels = base_channels_arg or model_config.get("base_channels", 64)
    num_classes = num_classes_arg if num_classes_arg is not None else model_config.get("num_classes", 0)
    model = TimeConditionedUNet(base_channels=base_channels, num_classes=num_classes).to(device)
    load_model_weights(model, checkpoint_path, device)
    model.eval()
    return model, num_classes


def assign_labels(
    target_labels: list[int] | None, n: int, device: torch.device, num_classes: int
) -> torch.Tensor | None:
    if num_classes <= 0 or target_labels is None:
        return None
    labels = [target_labels[i % len(target_labels)] for i in range(n)]
    return torch.tensor(labels, device=device, dtype=torch.long)


@torch.no_grad()
def generate_trajectories(
    model: TimeConditionedUNet,
    n: int,
    steps: int,
    labels: torch.Tensor | None,
    guidance_scale: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray | None]:
    """生成軌道を返す。shape: [n, steps+1, 784]。labels が None でなければ第2戻り値はラベル配列。"""
    x = torch.randn(n, 1, 28, 28, device=device)
    use_cfg = labels is not None and guidance_scale > 0.0 and getattr(model, "label_emb", None) is not None
    null_y = torch.full_like(labels, model.null_label_idx) if use_cfg else None

    traj = [x.detach().cpu().reshape(n, -1).numpy()]
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n,), i / steps, device=device)
        if use_cfg:
            v_cond = model(x, t, labels)
            v_uncond = model(x, t, null_y)
            velocity = (1.0 + guidance_scale) * v_cond - guidance_scale * v_uncond
        else:
            velocity = model(x, t, labels)
        x = x + dt * velocity
        traj.append(x.detach().cpu().reshape(n, -1).numpy())

    traj_arr = np.stack(traj, axis=1)  # [n, steps+1, 784]
    label_arr = labels.detach().cpu().numpy() if labels is not None else None
    return traj_arr, label_arr


def sample_data_images(
    data_dir: str, target_labels: list[int] | None, m: int, seed: int
) -> tuple[torch.Tensor, np.ndarray]:
    dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=_mnist_transform())
    targets = dataset.targets
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

    images = []
    labels = []
    for idx in chosen_indices.tolist():
        img, lb = dataset[idx]
        images.append(img)
        labels.append(lb)
    x1 = torch.stack(images, dim=0)  # [m, 1, 28, 28], in [-1, 1]
    return x1, np.array(labels)


def make_data_trajectories(x1: torch.Tensor, steps: int) -> np.ndarray:
    """直線内挿軌道を返す。shape: [m, steps+1, 784]。"""
    m = x1.size(0)
    x0 = torch.randn_like(x1)
    ts = torch.linspace(0.0, 1.0, steps + 1).view(1, steps + 1, 1, 1, 1)
    x0_b = x0.unsqueeze(1)
    x1_b = x1.unsqueeze(1)
    xt = (1.0 - ts) * x0_b + ts * x1_b  # [m, steps+1, 1, 28, 28]
    return xt.reshape(m, steps + 1, -1).numpy()


def fit_pca_and_plot(
    gen_traj: np.ndarray,
    gen_labels: np.ndarray | None,
    data_traj: np.ndarray,
    data_labels: np.ndarray,
    dim: int,
    out_path: Path,
) -> None:
    n, T, D = gen_traj.shape
    m = data_traj.shape[0]

    flat = np.concatenate([gen_traj.reshape(n * T, D), data_traj.reshape(m * T, D)], axis=0)
    pca = PCA(n_components=dim)
    proj = pca.fit_transform(flat)
    gen_proj = proj[: n * T].reshape(n, T, dim)
    data_proj = proj[n * T :].reshape(m, T, dim)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d") if dim == 3 else fig.add_subplot(111)

    def plot_line(ax, pts, color, alpha, label=None):
        if dim == 3:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, alpha=alpha, linewidth=1.0, label=label)
            ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color=color, marker="o", s=20)
            ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color=color, marker="*", s=60)
        else:
            ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=alpha, linewidth=1.0, label=label)
            ax.scatter(pts[0, 0], pts[0, 1], color=color, marker="o", s=20)
            ax.scatter(pts[-1, 0], pts[-1, 1], color=color, marker="*", s=60)

    for i in range(n):
        plot_line(ax, gen_proj[i], color="tab:blue", alpha=0.7, label="generated" if i == 0 else None)
    for j in range(m):
        plot_line(ax, data_proj[j], color="tab:orange", alpha=0.7, label="data interp" if j == 0 else None)

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")
    if dim == 3:
        ax.set_zlabel(f"PC3 ({var[2] * 100:.1f}%)")
    ax.set_title("Flow Matching trajectories (PCA)\nO=t=0 (noise),  *=t=1")
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

    model, num_classes = load_model(args.checkpoint, device, args.base_channels, args.num_classes)

    target_labels = args.labels
    if target_labels is not None and num_classes > 0:
        for lb in target_labels:
            if not 0 <= lb < num_classes:
                raise ValueError(f"--labels の値は 0..{num_classes - 1} の範囲で指定してください: {lb}")

    gen_label_tensor = assign_labels(target_labels, args.num_gen, device, num_classes)
    gen_traj, gen_labels = generate_trajectories(
        model, args.num_gen, args.steps, gen_label_tensor, args.guidance_scale, device
    )
    print(f"generated trajectories: {gen_traj.shape}")

    x1, data_labels = sample_data_images(args.data_dir, target_labels, args.num_data, args.seed)
    data_traj = make_data_trajectories(x1, args.steps)
    print(f"data trajectories: {data_traj.shape}")

    out_path = Path(args.out_path)
    fit_pca_and_plot(gen_traj, gen_labels, data_traj, data_labels, args.dim, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
