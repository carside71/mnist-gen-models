import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def _mnist_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )


def get_mnist_dataloader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    train: bool = True,
) -> DataLoader:
    """MNIST DataLoaderを作る。

    画像は [0, 1] から [-1, 1] に変換する。
    生成モデルでは、このスケールにしておくとノイズとの整合が取りやすい。
    """

    dataset = datasets.MNIST(
        root=data_dir,
        train=train,
        download=True,
        transform=_mnist_transform(),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )


def get_mnist_train_val_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """MNIST の train split を train/val に分割した2つの DataLoader を返す。

    test split は最終評価用に温存し、ここでは触らない。
    """

    full_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=_mnist_transform(),
    )

    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader
