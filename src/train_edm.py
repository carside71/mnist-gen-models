import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from mnist_gen.data import DATASET_SPECS, get_train_val_dataloaders
from mnist_gen.edm import edm_loss, f_edm
from mnist_gen.models import TimeConditionedUNet
from mnist_gen.utils import ensure_dir, get_device, save_checkpoint, save_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an EDM-style diffusion model (Karras et al. 2022).")

    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--data-dir", type=str, default="/workspace/datasets/mnist")
    parser.add_argument("--out-dir", type=str, default="/workspace/outputs/edm")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2, help="U-Net のダウンサンプリング段数")
    parser.add_argument("--num-classes", type=int, default=10, help="0 で無条件モデル")
    parser.add_argument("--p-uncond", type=float, default=0.1, help="CFG学習時のラベルドロップ確率")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)

    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--p-mean", type=float, default=-1.2, help="log σ サンプル分布の平均 (EDM)")
    parser.add_argument("--p-std", type=float, default=1.2, help="log σ サンプル分布の標準偏差 (EDM)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    print(f"device: {device}")

    out_dir = Path(args.out_dir)
    checkpoint_dir = ensure_dir(out_dir / "checkpoints")
    save_config(args, out_dir / "config.json")

    spec = DATASET_SPECS[args.dataset]

    train_loader, val_loader = get_train_val_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    model = TimeConditionedUNet(
        in_channels=spec["in_channels"],
        base_channels=args.base_channels,
        num_classes=args.num_classes,
        depth=args.depth,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")

        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True) if args.num_classes > 0 else None

            loss = edm_loss(
                model,
                images,
                sigma_data=args.sigma_data,
                sigma_min=args.sigma_min,
                p_mean=args.p_mean,
                p_std=args.p_std,
                labels=labels,
                p_uncond=args.p_uncond,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / total_count

        model.eval()
        val_total = 0.0
        val_count = 0
        torch.manual_seed(epoch)
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True) if args.num_classes > 0 else None
                loss = edm_loss(
                    model,
                    images,
                    sigma_data=args.sigma_data,
                    sigma_min=args.sigma_min,
                    p_mean=args.p_mean,
                    p_std=args.p_std,
                    labels=labels,
                    p_uncond=0.0,
                )
                val_total += loss.item() * images.size(0)
                val_count += images.size(0)
        val_loss = val_total / val_count

        with torch.no_grad():
            probe_B = min(64, args.batch_size)
            probe_shape = (probe_B, spec["in_channels"], spec["image_size"], spec["image_size"])
            x_init = args.sigma_max * torch.randn(probe_shape, device=device)
            sigma_init = torch.full((probe_B,), args.sigma_max, device=device)
            if args.num_classes > 0:
                probe_labels = torch.arange(probe_B, device=device, dtype=torch.long) % args.num_classes
            else:
                probe_labels = None
            probe_x0 = f_edm(model, x_init, sigma_init, args.sigma_data, args.sigma_min, probe_labels)
            one_shot_var = probe_x0.var().item()
            one_shot_abs_mean = probe_x0.abs().mean().item()

        print(
            f"epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"one_shot_var={one_shot_var:.4f} one_shot_abs_mean={one_shot_abs_mean:.4f}"
        )

        extra = {
            "model_config": {
                "base_channels": args.base_channels,
                "depth": args.depth,
                "num_classes": args.num_classes,
                "in_channels": spec["in_channels"],
                "image_size": spec["image_size"],
                "dataset": args.dataset,
            },
            "edm_config": {
                "sigma_min": args.sigma_min,
                "sigma_max": args.sigma_max,
                "rho": args.rho,
                "sigma_data": args.sigma_data,
                "p_mean": args.p_mean,
                "p_std": args.p_std,
            },
            "train_loss": train_loss,
        }

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=val_loss,
            extra=extra,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                extra=extra,
            )

if __name__ == "__main__":
    main()
