import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from mnist_gen.data import DATASET_SPECS, get_train_val_dataloaders
from mnist_gen.edm import f_edm, karras_sigmas
from mnist_gen.models import TimeConditionedUNet
from mnist_gen.utils import ensure_dir, get_device, save_checkpoint, save_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill a pretrained EDM teacher into a consistency model (Song et al., 2023)."
    )

    parser.add_argument(
        "--teacher-checkpoint",
        type=str,
        default="/workspace/outputs/edm/exp_01/checkpoints/best.pt",
    )
    parser.add_argument("--dataset", type=str, default=None, choices=[None, "mnist", "cifar10"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="/workspace/outputs/consistency")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)

    parser.add_argument("--num-steps", type=int, default=18, help="CD の離散時刻数 N (論文 Algorithm 2)")
    parser.add_argument("--sigma-min", type=float, default=None, help="None なら教師 ckpt から継承")
    parser.add_argument("--sigma-max", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--sigma-data", type=float, default=None)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "pseudo_huber"])

    # 教師ckptの構成を上書きしたい場合
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)

    return parser.parse_args()


def f_consistency(
    model: torch.nn.Module,
    x: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float,
    sigma_min: float,
    labels: torch.Tensor | None,
) -> torch.Tensor:
    """学生の consistency 関数。論文 §3.3 で f_θ は EDM 前処理付きデノイザと同形。"""
    return f_edm(model, x, sigma, sigma_data, sigma_min, labels)


def teacher_denoise(
    teacher: torch.nn.Module,
    x: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float,
    sigma_min: float,
    labels: torch.Tensor | None,
) -> torch.Tensor:
    """EDM 教師の x₀ 予測 D_φ(x, σ)。教師と学生は同じ EDM 前処理を共有する。"""
    return f_edm(teacher, x, sigma, sigma_data, sigma_min, labels)


def euler_step(
    x_hi: torch.Tensor,
    sigma_hi: torch.Tensor,
    sigma_lo: torch.Tensor,
    teacher: torch.nn.Module,
    sigma_data: float,
    sigma_min: float,
    labels: torch.Tensor | None,
) -> torch.Tensor:
    """論文の Φ(·;φ) を Euler で実装。dx/dσ = (x − D_φ)/σ。"""
    sigma_hi_b = sigma_hi.view(-1, 1, 1, 1)
    sigma_lo_b = sigma_lo.view(-1, 1, 1, 1)
    D = teacher_denoise(teacher, x_hi, sigma_hi, sigma_data, sigma_min, labels)
    d = (x_hi - D) / sigma_hi_b
    return x_hi + (sigma_lo_b - sigma_hi_b) * d


def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ep, p in zip(ema_model.parameters(), model.parameters(), strict=True):
            ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
        for eb, b in zip(ema_model.buffers(), model.buffers(), strict=True):
            eb.data.copy_(b.data)


def consistency_loss(pred: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "mse":
        return F.mse_loss(pred, target)
    # Pseudo-Huber (Karras 系)
    c = 0.00054 * (pred[0].numel() ** 0.5)
    return (torch.sqrt((pred - target) ** 2 + c * c) - c).mean()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    teacher_ckpt = torch.load(args.teacher_checkpoint, map_location=device)
    if "edm_config" not in teacher_ckpt:
        raise ValueError(
            "EDM 教師チェックポイントが必要です (edm_config が見つかりません)。"
            " train_edm.py で学習した checkpoint を --teacher-checkpoint に指定してください。"
        )

    t_model_cfg = teacher_ckpt.get("model_config", {})
    t_edm_cfg = teacher_ckpt["edm_config"]

    base_channels = args.base_channels or t_model_cfg.get("base_channels", 64)
    depth = args.depth or t_model_cfg.get("depth", 2)
    num_classes = args.num_classes if args.num_classes is not None else t_model_cfg.get("num_classes", 0)
    dataset = args.dataset or t_model_cfg.get("dataset", "mnist")
    spec = DATASET_SPECS[dataset]
    in_channels = t_model_cfg.get("in_channels", spec["in_channels"])
    image_size = t_model_cfg.get("image_size", spec["image_size"])

    sigma_min = args.sigma_min if args.sigma_min is not None else t_edm_cfg["sigma_min"]
    sigma_max = args.sigma_max if args.sigma_max is not None else t_edm_cfg["sigma_max"]
    rho = args.rho if args.rho is not None else t_edm_cfg["rho"]
    sigma_data = args.sigma_data if args.sigma_data is not None else t_edm_cfg["sigma_data"]

    data_dir = args.data_dir or f"/workspace/datasets/{dataset}"

    out_dir = Path(args.out_dir)
    checkpoint_dir = ensure_dir(out_dir / "checkpoints")
    save_config(args, out_dir / "config.json")

    train_loader, val_loader = get_train_val_dataloaders(
        dataset=dataset,
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    teacher = TimeConditionedUNet(
        in_channels=in_channels,
        base_channels=base_channels,
        num_classes=num_classes,
        depth=depth,
    ).to(device)
    teacher.load_state_dict(teacher_ckpt["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = TimeConditionedUNet(
        in_channels=in_channels,
        base_channels=base_channels,
        num_classes=num_classes,
        depth=depth,
    ).to(device)
    # 教師重みで warm-start (論文 Algorithm 2)
    student.load_state_dict(teacher_ckpt["model"])

    student_ema = copy.deepcopy(student)
    for p in student_ema.parameters():
        p.requires_grad_(False)
    student_ema.eval()

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    # Karras 風 σ-grid。昇順 [σ_min, ..., σ_max], 長さ N+1。
    sigma_grid = karras_sigmas(args.num_steps, sigma_min, sigma_max, rho, device)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss = 0.0
        total_count = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")

        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True) if num_classes > 0 else None
            B = images.size(0)

            # 論文 Algorithm 2: n ∈ [1, N-1], σ_{n+1} (high), σ_n (low)
            n_idx = torch.randint(1, args.num_steps, (B,), device=device)
            sigma_hi = sigma_grid[n_idx + 1]
            sigma_lo = sigma_grid[n_idx]

            # EDM forward: x_σ = x + σ·z
            z = torch.randn_like(images)
            x_hi = images + sigma_hi.view(-1, 1, 1, 1) * z

            with torch.no_grad():
                x_lo_hat = euler_step(x_hi, sigma_hi, sigma_lo, teacher, sigma_data, sigma_min, labels)

            pred = f_consistency(student, x_hi, sigma_hi, sigma_data, sigma_min, labels)
            with torch.no_grad():
                target = f_consistency(student_ema, x_lo_hat, sigma_lo, sigma_data, sigma_min, labels)

            loss = consistency_loss(pred, target, args.loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            update_ema(student_ema, student, args.ema_decay)

            total_loss += loss.item() * B
            total_count += B
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / total_count

        student.eval()
        val_total = 0.0
        val_count = 0
        torch.manual_seed(epoch)
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True) if num_classes > 0 else None
                B = images.size(0)

                n_idx = torch.randint(1, args.num_steps, (B,), device=device)
                sigma_hi = sigma_grid[n_idx + 1]
                sigma_lo = sigma_grid[n_idx]

                z = torch.randn_like(images)
                x_hi = images + sigma_hi.view(-1, 1, 1, 1) * z
                x_lo_hat = euler_step(x_hi, sigma_hi, sigma_lo, teacher, sigma_data, sigma_min, labels)

                pred = f_consistency(student, x_hi, sigma_hi, sigma_data, sigma_min, labels)
                target = f_consistency(student_ema, x_lo_hat, sigma_lo, sigma_data, sigma_min, labels)
                v = consistency_loss(pred, target, args.loss)
                val_total += v.item() * B
                val_count += B
        val_loss = val_total / val_count

        with torch.no_grad():
            probe_B = min(64, args.batch_size)
            probe_shape = (probe_B, in_channels, image_size, image_size)
            x_init = sigma_max * torch.randn(probe_shape, device=device)
            sigma_init = torch.full((probe_B,), sigma_max, device=device)
            if num_classes > 0:
                probe_labels = torch.arange(probe_B, device=device, dtype=torch.long) % num_classes
            else:
                probe_labels = None
            probe_x0 = f_consistency(student_ema, x_init, sigma_init, sigma_data, sigma_min, probe_labels)
            one_shot_var = probe_x0.var().item()
            one_shot_abs_mean = probe_x0.abs().mean().item()

        print(
            f"epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"one_shot_var={one_shot_var:.4f} one_shot_abs_mean={one_shot_abs_mean:.4f}"
        )

        extra = {
            "model_config": {
                "base_channels": base_channels,
                "depth": depth,
                "num_classes": num_classes,
                "in_channels": in_channels,
                "image_size": image_size,
                "dataset": dataset,
            },
            "edm_config": {
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "rho": rho,
                "sigma_data": sigma_data,
            },
            "consistency_config": {
                "num_steps": args.num_steps,
                "ema_decay": args.ema_decay,
                "loss": args.loss,
                "teacher_checkpoint": args.teacher_checkpoint,
            },
            "train_loss": train_loss,
            "model_ema": student_ema.state_dict(),
        }

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model=student,
            optimizer=optimizer,
            epoch=epoch,
            loss=val_loss,
            extra=extra,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model=student,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                extra=extra,
            )


if __name__ == "__main__":
    main()
