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

    model = TimeConditionedUNet(base_channels=base_channels).to(device)
    load_model_weights(model, args.checkpoint, device)

    schedule = DiffusionSchedule.create(timesteps=timesteps, device=device)

    samples = sample_ddpm(
        model=model,
        schedule=schedule,
        shape=(args.num_samples, 1, 28, 28),
        device=device,
    )

    save_samples_grid(samples, Path(args.out_path))
    print(f"saved: {args.out_path}")


if __name__ == "__main__":
    main()
