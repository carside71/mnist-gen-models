import argparse
from pathlib import Path

import torch

from mnist_gen.data import DATASET_SPECS
from mnist_gen.diffusion import DiffusionSchedule, extract
from mnist_gen.models import TimeConditionedUNet
from mnist_gen.utils import get_device, save_samples_grid, set_seed
from train_consistency_distillation import f_consistency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample images from a distilled consistency model.")

    parser.add_argument("--checkpoint", type=str, default="/workspace/outputs/consistency/checkpoints/best.pt")
    parser.add_argument(
        "--out-path",
        type=str,
        default="/workspace/outputs/consistency/samples/consistency_samples.png",
    )
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1, help="multistep CM のサンプリングステップ数")
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--no-ema", dest="use_ema", action="store_false")

    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--num-bins", type=int, default=None)
    parser.add_argument("--sigma-data", type=float, default=None)
    parser.add_argument("--dataset", type=str, default=None, choices=[None, "mnist", "cifar10"])
    parser.add_argument("--label", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


@torch.no_grad()
def sample_consistency(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    shape: tuple[int, int, int, int],
    device: torch.device,
    t_idx_grid: torch.Tensor,
    sigma_data: float,
    steps: int,
    labels: torch.Tensor | None,
) -> torch.Tensor:
    """multistep consistency sampling。steps=1 で 1 ショット。"""
    model.eval()
    T = schedule.timesteps
    B = shape[0]

    # 初期は最も大きい σ に対応する純ノイズ点。√(1-ᾱ_T-1)/√ᾱ_T-1 ≈ √(1-ᾱ)/√ᾱ ≫ 1
    # CM の定式上、入力は σ(t)·z スケール想定だが、ここでは VP の x_T を使う。
    t_hi = torch.full((B,), T - 1, device=device, dtype=torch.long)
    x = torch.randn(shape, device=device)
    # 教師 schedule で x ~ N(0, I) 上から開始: 数式上は x_T = √ᾱ_{T-1} · 0 + √(1-ᾱ_{T-1}) · z
    # ᾱ_{T-1} ≈ 0 なので近似的に √(1-ᾱ) ≈ 1 で OK
    x0 = f_consistency(model, x, t_hi, schedule, sigma_data, labels)

    if steps <= 1:
        return x0.clamp(-1.0, 1.0)

    # 中間ビン点 t_idx_grid[1..steps-1]（降順）で再ノイズ→再投影
    # t_idx_grid[0]=0, t_idx_grid[-1]=T-1
    intermediate = list(reversed(t_idx_grid.tolist()))[1:-1]
    if len(intermediate) == 0:
        return x0.clamp(-1.0, 1.0)
    chosen = [intermediate[i] for i in torch.linspace(0, len(intermediate) - 1, steps - 1).round().long().tolist()]

    for ti in chosen:
        t_cur = torch.full((B,), int(ti), device=device, dtype=torch.long)
        ab = extract(schedule.alpha_bars, t_cur, x0.shape)
        z = torch.randn_like(x0)
        x_t = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * z
        x0 = f_consistency(model, x_t, t_cur, schedule, sigma_data, labels)

    return x0.clamp(-1.0, 1.0)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    print(f"device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint.get("model_config", {})
    diffusion_config = checkpoint.get("diffusion_config", {})
    consistency_config = checkpoint.get("consistency_config", {})

    base_channels = args.base_channels or model_config.get("base_channels", 64)
    depth = args.depth or model_config.get("depth", 2)
    timesteps = args.timesteps or diffusion_config.get("timesteps", 1000)
    num_classes = args.num_classes if args.num_classes is not None else model_config.get("num_classes", 0)
    num_bins = args.num_bins or consistency_config.get("num_bins", 18)
    sigma_data = args.sigma_data if args.sigma_data is not None else consistency_config.get("sigma_data", 0.5)
    dataset = args.dataset or model_config.get("dataset", "mnist")
    spec = DATASET_SPECS[dataset]
    in_channels = model_config.get("in_channels", spec["in_channels"])
    image_size = model_config.get("image_size", spec["image_size"])

    model = TimeConditionedUNet(
        in_channels=in_channels,
        base_channels=base_channels,
        num_classes=num_classes,
        depth=depth,
    ).to(device)

    state_key = "model_ema" if (args.use_ema and "model_ema" in checkpoint) else "model"
    model.load_state_dict(checkpoint[state_key])
    model.eval()
    print(f"loaded weights from key: {state_key}")

    schedule = DiffusionSchedule.create(timesteps=timesteps, device=device)
    t_idx_grid = torch.linspace(0, timesteps - 1, num_bins + 1, device=device).round().long()

    labels = None
    out_path = Path(args.out_path)
    if num_classes > 0:
        if args.label is not None:
            if not 0 <= args.label < num_classes:
                raise ValueError(f"--label は 0..{num_classes - 1} の範囲で指定してください")
            labels = torch.full((args.num_samples,), args.label, device=device, dtype=torch.long)
            out_path = out_path.with_name(f"{out_path.stem}_steps{args.steps}_label{args.label}{out_path.suffix}")
        else:
            labels = torch.arange(args.num_samples, device=device, dtype=torch.long) % num_classes
            out_path = out_path.with_name(f"{out_path.stem}_steps{args.steps}{out_path.suffix}")
    else:
        out_path = out_path.with_name(f"{out_path.stem}_steps{args.steps}{out_path.suffix}")

    samples = sample_consistency(
        model=model,
        schedule=schedule,
        shape=(args.num_samples, in_channels, image_size, image_size),
        device=device,
        t_idx_grid=t_idx_grid,
        sigma_data=sigma_data,
        steps=args.steps,
        labels=labels,
    )

    save_samples_grid(samples, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
