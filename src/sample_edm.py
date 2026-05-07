import argparse
from pathlib import Path

import torch

from mnist_gen.data import DATASET_SPECS
from mnist_gen.edm import heun_sample, karras_sigmas
from mnist_gen.models import TimeConditionedUNet
from mnist_gen.utils import get_device, load_model_weights, save_samples_grid, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample images from a trained EDM diffusion model.")

    parser.add_argument("--checkpoint", type=str, default="/workspace/outputs/edm/checkpoints/best.pt")
    parser.add_argument("--out-path", type=str, default="/workspace/outputs/edm/samples/edm_samples.png")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=18, help="Heun サンプラの離散ステップ数")
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--sigma-min", type=float, default=None)
    parser.add_argument("--sigma-max", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--sigma-data", type=float, default=None)
    parser.add_argument("--dataset", type=str, default=None, choices=[None, "mnist", "cifar10"])
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
    edm_config = checkpoint.get("edm_config", {})

    base_channels = args.base_channels or model_config.get("base_channels", 64)
    depth = args.depth or model_config.get("depth", 2)
    num_classes = args.num_classes if args.num_classes is not None else model_config.get("num_classes", 0)
    sigma_min = args.sigma_min if args.sigma_min is not None else edm_config.get("sigma_min", 0.002)
    sigma_max = args.sigma_max if args.sigma_max is not None else edm_config.get("sigma_max", 80.0)
    rho = args.rho if args.rho is not None else edm_config.get("rho", 7.0)
    sigma_data = args.sigma_data if args.sigma_data is not None else edm_config.get("sigma_data", 0.5)
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
    load_model_weights(model, args.checkpoint, device)

    sigma_grid = karras_sigmas(args.num_steps, sigma_min, sigma_max, rho, device)

    labels = None
    out_path = Path(args.out_path)
    if num_classes > 0:
        if args.label is not None:
            if not 0 <= args.label < num_classes:
                raise ValueError(f"--label は 0..{num_classes - 1} の範囲で指定してください")
            labels = torch.full((args.num_samples,), args.label, device=device, dtype=torch.long)
            out_path = out_path.with_name(
                f"{out_path.stem}_guided{args.guidance_scale}_label{args.label}{out_path.suffix}"
            )
        else:
            labels = torch.arange(args.num_samples, device=device, dtype=torch.long) % num_classes
            out_path = out_path.with_name(f"{out_path.stem}_guided{args.guidance_scale}{out_path.suffix}")

    null_label_idx = model.null_label_idx if num_classes > 0 else None

    samples = heun_sample(
        model=model,
        shape=(args.num_samples, in_channels, image_size, image_size),
        device=device,
        sigma_grid=sigma_grid,
        sigma_data=sigma_data,
        sigma_min=sigma_min,
        labels=labels,
        guidance_scale=args.guidance_scale,
        null_label_idx=null_label_idx,
    )

    save_samples_grid(samples, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
