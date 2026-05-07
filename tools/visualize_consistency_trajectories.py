"""Consistency Model の生成軌道と前向き拡散軌道を PCA で可視化するツール。"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visualize_diffusion_trajectories import (  # type: ignore
    assign_labels,
    fit_pca_and_plot,
    make_data_trajectories,
    sample_data_images,
)

from mnist_gen.data import DATASET_SPECS
from mnist_gen.diffusion import DiffusionSchedule, extract
from mnist_gen.models import TimeConditionedUNet
from mnist_gen.utils import get_device, set_seed
from train_consistency_distillation import f_consistency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Consistency Model trajectories via PCA.")
    parser.add_argument("--checkpoint", type=str, default="/workspace/outputs/consistency/checkpoints/best.pt")
    parser.add_argument(
        "--out-path",
        type=str,
        default="/workspace/outputs/consistency/trajectories/consistency_traj.png",
    )
    parser.add_argument("--dataset", type=str, default=None, choices=[None, "mnist", "cifar10"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--num-gen", type=int, default=16)
    parser.add_argument("--num-data", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8, help="multistep CM のサンプリングステップ数")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--num-bins", type=int, default=None)
    parser.add_argument("--sigma-data", type=float, default=None)
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--no-ema", dest="use_ema", action="store_false")
    parser.add_argument("--labels", type=int, nargs="+", default=None)
    parser.add_argument("--dim", type=int, choices=[2, 3], default=2)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_consistency_model(
    checkpoint_path: str,
    device: torch.device,
    use_ema: bool,
    base_channels_arg: int | None,
    depth_arg: int | None,
    num_classes_arg: int | None,
    dataset_arg: str | None,
    timesteps_arg: int | None,
    num_bins_arg: int | None,
    sigma_data_arg: float | None,
) -> tuple[TimeConditionedUNet, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("model_config", {})
    diffusion_config = checkpoint.get("diffusion_config", {})
    consistency_config = checkpoint.get("consistency_config", {})

    base_channels = base_channels_arg or model_config.get("base_channels", 64)
    depth = depth_arg or model_config.get("depth", 2)
    num_classes = num_classes_arg if num_classes_arg is not None else model_config.get("num_classes", 0)
    dataset = dataset_arg or model_config.get("dataset", "mnist")
    spec = DATASET_SPECS[dataset]
    in_channels = model_config.get("in_channels", spec["in_channels"])
    image_size = model_config.get("image_size", spec["image_size"])
    timesteps = timesteps_arg or diffusion_config.get("timesteps", 1000)
    num_bins = num_bins_arg or consistency_config.get("num_bins", 18)
    sigma_data = sigma_data_arg if sigma_data_arg is not None else consistency_config.get("sigma_data", 0.5)

    model = TimeConditionedUNet(
        in_channels=in_channels,
        base_channels=base_channels,
        num_classes=num_classes,
        depth=depth,
    ).to(device)

    state_key = "model_ema" if (use_ema and "model_ema" in checkpoint) else "model"
    model.load_state_dict(checkpoint[state_key])
    model.eval()

    info = {
        "num_classes": num_classes,
        "dataset": dataset,
        "in_channels": in_channels,
        "image_size": image_size,
        "timesteps": timesteps,
        "num_bins": num_bins,
        "sigma_data": sigma_data,
        "state_key": state_key,
    }
    return model, info


@torch.no_grad()
def generate_consistency_trajectories(
    model: TimeConditionedUNet,
    schedule: DiffusionSchedule,
    t_idx_grid: torch.Tensor,
    sigma_data: float,
    steps: int,
    n: int,
    labels: torch.Tensor | None,
    device: torch.device,
    in_channels: int,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """multistep CM の軌道を [n, S+1, D] で返す。S+1 個のスナップショット = (純ノイズ, x0_step1, x_t1, x0_step2, ...)。
    可視化のため「ノイズ → 中間 → 最終 x0」の順に並べる。"""
    T = schedule.timesteps
    snapshots: list[np.ndarray] = []

    x = torch.randn(n, in_channels, image_size, image_size, device=device)
    snapshots.append(x.detach().cpu().reshape(n, -1).numpy())  # 純ノイズ

    t_hi = torch.full((n,), T - 1, device=device, dtype=torch.long)
    x0 = f_consistency(model, x, t_hi, schedule, sigma_data, labels)
    snapshots.append(x0.detach().cpu().reshape(n, -1).numpy())  # 1 ショット推定

    if steps > 1:
        intermediate = list(reversed(t_idx_grid.tolist()))[1:-1]
        if len(intermediate) > 0:
            chosen = [
                intermediate[i] for i in torch.linspace(0, len(intermediate) - 1, steps - 1).round().long().tolist()
            ]
            for ti in chosen:
                t_cur = torch.full((n,), int(ti), device=device, dtype=torch.long)
                ab = extract(schedule.alpha_bars, t_cur, x0.shape)
                z = torch.randn_like(x0)
                x_t = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * z
                snapshots.append(x_t.detach().cpu().reshape(n, -1).numpy())  # 再ノイズ
                x0 = f_consistency(model, x_t, t_cur, schedule, sigma_data, labels)
                snapshots.append(x0.detach().cpu().reshape(n, -1).numpy())  # 再投影

    traj = np.stack(snapshots, axis=1)  # [n, S+1, D]
    label_arr = labels.detach().cpu().numpy() if labels is not None else None
    return traj, label_arr


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    model, info = load_consistency_model(
        args.checkpoint,
        device,
        args.use_ema,
        args.base_channels,
        args.depth,
        args.num_classes,
        args.dataset,
        args.timesteps,
        args.num_bins,
        args.sigma_data,
    )
    print(f"loaded weights from key: {info['state_key']}")

    dataset_name = info["dataset"]
    num_classes = info["num_classes"]
    in_channels = info["in_channels"]
    image_size = info["image_size"]
    timesteps = info["timesteps"]
    num_bins = info["num_bins"]
    sigma_data = info["sigma_data"]

    data_dir = args.data_dir or f"/workspace/datasets/{dataset_name}"
    schedule = DiffusionSchedule.create(timesteps=timesteps, device=device)
    t_idx_grid = torch.linspace(0, timesteps - 1, num_bins + 1, device=device).round().long()

    target_labels = args.labels
    if target_labels is not None and num_classes > 0:
        for lb in target_labels:
            if not 0 <= lb < num_classes:
                raise ValueError(f"--labels の値は 0..{num_classes - 1} の範囲で指定してください: {lb}")

    gen_label_tensor = assign_labels(target_labels, args.num_gen, device, num_classes)
    gen_traj, gen_labels = generate_consistency_trajectories(
        model,
        schedule,
        t_idx_grid,
        sigma_data,
        args.steps,
        args.num_gen,
        gen_label_tensor,
        device,
        in_channels,
        image_size,
    )
    print(f"generated trajectories: {gen_traj.shape}")

    # データ軌跡数を gen_traj の時間長に合わせる
    data_steps = gen_traj.shape[1] - 1
    x1, data_labels = sample_data_images(dataset_name, data_dir, target_labels, args.num_data, args.seed)
    data_traj = make_data_trajectories(x1, schedule, data_steps, device)
    print(f"data trajectories: {data_traj.shape}")

    out_path = Path(args.out_path)
    out_path = out_path.with_name(f"{out_path.stem}_steps{args.steps}_dim{args.dim}{out_path.suffix}")
    fit_pca_and_plot(gen_traj, gen_labels, data_traj, data_labels, args.dim, args.pca_dim, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
