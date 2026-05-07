import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from mnist_gen.data import DATASET_SPECS, get_train_val_dataloaders
from mnist_gen.diffusion import DiffusionSchedule, extract, q_sample
from mnist_gen.models import TimeConditionedUNet
from mnist_gen.utils import ensure_dir, get_device, save_checkpoint, save_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill a pretrained DDPM teacher into a consistency model (Song et al., 2023)."
    )

    parser.add_argument(
        "--teacher-checkpoint",
        type=str,
        default="/workspace/outputs/diffusion/checkpoints/best.pt",
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

    parser.add_argument("--num-bins", type=int, default=18, help="CD の離散時刻数 N")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "pseudo_huber"])

    # 教師ckptの構成を上書きしたい場合
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)

    return parser.parse_args()


def cd_skip_out(sigma_t: torch.Tensor, sigma_data: float) -> tuple[torch.Tensor, torch.Tensor]:
    """EDM 流の skip/out 係数。f(x,0)=x_0 となる境界条件を満たす。"""
    sd2 = sigma_data * sigma_data
    c_skip = sd2 / (sigma_t * sigma_t + sd2)
    c_out = sigma_t * sigma_data / torch.sqrt(sigma_t * sigma_t + sd2)
    return c_skip, c_out


def sigma_from_index(t_idx: torch.Tensor, schedule: DiffusionSchedule, x_shape: torch.Size) -> torch.Tensor:
    """DDPM の分散保存形に対応する σ(t) = √(1-ᾱ_t)/√(ᾱ_t)。"""
    alpha_bar = extract(schedule.alpha_bars, t_idx, x_shape)
    return torch.sqrt((1.0 - alpha_bar) / alpha_bar)


def f_consistency(
    model: torch.nn.Module,
    x: torch.Tensor,
    t_idx: torch.Tensor,
    schedule: DiffusionSchedule,
    sigma_data: float,
    labels: torch.Tensor | None,
) -> torch.Tensor:
    """skip parameterized consistency function f_θ(x, t) = c_skip x + c_out F_θ(x, t)。"""
    sigma_t = sigma_from_index(t_idx, schedule, x.shape)
    c_skip, c_out = cd_skip_out(sigma_t, sigma_data)
    t_norm = t_idx.float() / (schedule.timesteps - 1)
    out = model(x, t_norm, labels)
    return c_skip * x + c_out * out


def ddim_step_index(
    x_hi: torch.Tensor,
    eps: torch.Tensor,
    t_hi_idx: torch.Tensor,
    t_lo_idx: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    """DDIM η=0 の 1 ステップ。t_hi → t_lo へ遷移。"""
    ab_hi = extract(schedule.alpha_bars, t_hi_idx, x_hi.shape)
    ab_lo = extract(schedule.alpha_bars, t_lo_idx, x_hi.shape)
    x0_pred = (x_hi - torch.sqrt(1.0 - ab_hi) * eps) / torch.sqrt(ab_hi)
    return torch.sqrt(ab_lo) * x0_pred + torch.sqrt(1.0 - ab_lo) * eps


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
    t_model_cfg = teacher_ckpt.get("model_config", {})
    t_diff_cfg = teacher_ckpt.get("diffusion_config", {})

    base_channels = args.base_channels or t_model_cfg.get("base_channels", 64)
    depth = args.depth or t_model_cfg.get("depth", 2)
    num_classes = args.num_classes if args.num_classes is not None else t_model_cfg.get("num_classes", 0)
    timesteps = args.timesteps or t_diff_cfg.get("timesteps", 1000)
    dataset = args.dataset or t_model_cfg.get("dataset", "mnist")
    spec = DATASET_SPECS[dataset]
    in_channels = t_model_cfg.get("in_channels", spec["in_channels"])
    image_size = t_model_cfg.get("image_size", spec["image_size"])

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
    # 教師重みで warm-start
    student.load_state_dict(teacher_ckpt["model"])

    student_ema = copy.deepcopy(student)
    for p in student_ema.parameters():
        p.requires_grad_(False)
    student_ema.eval()

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    schedule = DiffusionSchedule.create(timesteps=timesteps, device=device)

    # 等間隔のビン境界 t_idx[0]=0, t_idx[N]=T-1
    t_idx_grid = torch.linspace(0, timesteps - 1, args.num_bins + 1, device=device).round().long()

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

            # サンプルごとに n ∈ [0, N-1] を一様サンプル → t_hi = t_idx[n+1], t_lo = t_idx[n]
            n_idx = torch.randint(0, args.num_bins, (B,), device=device)
            t_hi = t_idx_grid[n_idx + 1]
            t_lo = t_idx_grid[n_idx]

            x_hi, _ = q_sample(images, t_hi, schedule)

            with torch.no_grad():
                t_hi_norm = t_hi.float() / (timesteps - 1)
                eps_teacher = teacher(x_hi, t_hi_norm, labels)
                x_lo_hat = ddim_step_index(x_hi, eps_teacher, t_hi, t_lo, schedule)

            pred = f_consistency(student, x_hi, t_hi, schedule, args.sigma_data, labels)
            with torch.no_grad():
                target = f_consistency(student_ema, x_lo_hat, t_lo, schedule, args.sigma_data, labels)

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

                n_idx = torch.randint(0, args.num_bins, (B,), device=device)
                t_hi = t_idx_grid[n_idx + 1]
                t_lo = t_idx_grid[n_idx]

                x_hi, _ = q_sample(images, t_hi, schedule)
                t_hi_norm = t_hi.float() / (timesteps - 1)
                eps_teacher = teacher(x_hi, t_hi_norm, labels)
                x_lo_hat = ddim_step_index(x_hi, eps_teacher, t_hi, t_lo, schedule)

                pred = f_consistency(student, x_hi, t_hi, schedule, args.sigma_data, labels)
                target = f_consistency(student_ema, x_lo_hat, t_lo, schedule, args.sigma_data, labels)
                v = consistency_loss(pred, target, args.loss)
                val_total += v.item() * B
                val_count += B
        val_loss = val_total / val_count

        print(f"epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        extra = {
            "model_config": {
                "base_channels": base_channels,
                "depth": depth,
                "num_classes": num_classes,
                "in_channels": in_channels,
                "image_size": image_size,
                "dataset": dataset,
            },
            "diffusion_config": {
                "timesteps": timesteps,
            },
            "consistency_config": {
                "num_bins": args.num_bins,
                "ema_decay": args.ema_decay,
                "sigma_data": args.sigma_data,
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
