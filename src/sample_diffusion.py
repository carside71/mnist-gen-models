import argparse
from pathlib import Path

import torch

from mnist_gen.diffusion import DiffusionSchedule, sample_ddpm
from mnist_gen.models import TimeConditionedUNet
from mnist_gen.utils import get_device, load_model_weights, save_samples_grid, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample images from a trained diffusion model.")

    parser.add_argument("--checkpoint", type=str, default="/workspace/outputs/diffusion/checkpoints/best.pt")
    parser.add_argument("--out-path", type=str, default="/workspace/outputs/diffusion/samples/diffusion_samples.png")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--label", type=int, default=None, help="生成したい数字 (0-9)。省略時は 0..9 を循環")
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    print(f"device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)

    model_config = checkpoint.get("model_config", {})
    diffusion_config = checkpoint.get("diffusion_config", {})

    base_channels = args.base_channels or model_config.get("base_channels", 64)
    timesteps = args.timesteps or diffusion_config.get("timesteps", 1000)
    num_classes = args.num_classes if args.num_classes is not None else model_config.get("num_classes", 0)

    model = TimeConditionedUNet(base_channels=base_channels, num_classes=num_classes).to(device)
    load_model_weights(model, args.checkpoint, device)

    schedule = DiffusionSchedule.create(timesteps=timesteps, device=device)

    labels = None
    out_path = Path(args.out_path)
    if num_classes > 0:
        if args.label is not None:
            if not 0 <= args.label < num_classes:
                raise ValueError(f"--label は 0..{num_classes - 1} の範囲で指定してください")
            labels = torch.full((args.num_samples,), args.label, device=device, dtype=torch.long)
            out_path = out_path.with_name(f"{out_path.stem}_label{args.label}{out_path.suffix}")
        else:
            labels = torch.arange(args.num_samples, device=device, dtype=torch.long) % num_classes

    samples = sample_ddpm(
        model=model,
        schedule=schedule,
        shape=(args.num_samples, 1, 28, 28),
        device=device,
        labels=labels,
        guidance_scale=args.guidance_scale,
    )

    save_samples_grid(samples, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
