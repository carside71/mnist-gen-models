import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from mnist_gen.data import get_mnist_dataloader
from mnist_gen.diffusion import DiffusionSchedule, diffusion_loss
from mnist_gen.models import TimeConditionedUNet
from mnist_gen.utils import ensure_dir, get_device, save_checkpoint, save_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DDPM-style diffusion model on MNIST.")

    parser.add_argument("--data-dir", type=str, default="/workspace/datasets/mnist")
    parser.add_argument("--out-dir", type=str, default="outputs/diffusion")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    print(f"device: {device}")

    out_dir = Path(args.out_dir)
    checkpoint_dir = ensure_dir(out_dir / "checkpoints")
    save_config(args, out_dir / "config.json")

    dataloader = get_mnist_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=True,
    )

    model = TimeConditionedUNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    schedule = DiffusionSchedule.create(timesteps=args.timesteps, device=device)

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        progress = tqdm(dataloader, desc=f"epoch {epoch}/{args.epochs}")

        for images, _ in progress:
            images = images.to(device, non_blocking=True)

            loss = diffusion_loss(model, images, schedule)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            progress.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = total_loss / total_count
        print(f"epoch {epoch}: loss={epoch_loss:.6f}")

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=epoch_loss,
            extra={
                "model_config": {
                    "base_channels": args.base_channels,
                },
                "diffusion_config": {
                    "timesteps": args.timesteps,
                },
            },
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=epoch_loss,
                extra={
                    "model_config": {
                        "base_channels": args.base_channels,
                    },
                    "diffusion_config": {
                        "timesteps": args.timesteps,
                    },
                },
            )


if __name__ == "__main__":
    main()
