import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torchvision.utils import save_image


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_config(args: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    loss: float,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }

    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()

    if extra is not None:
        payload.update(extra)

    torch.save(payload, path)


def load_model_weights(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def save_samples_grid(samples: torch.Tensor, path: str | Path) -> None:
    """[-1, 1] の画像テンソルをPNGのグリッドとして保存する。"""

    path = Path(path)
    ensure_dir(path.parent)

    samples = samples.detach().cpu().clamp(-1.0, 1.0)
    samples = (samples + 1.0) / 2.0

    nrow = int(math.sqrt(samples.size(0)))
    save_image(samples, path, nrow=max(nrow, 1))
