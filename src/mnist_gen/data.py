import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


DATASET_SPECS = {
    "mnist": {"in_channels": 1, "image_size": 28, "num_classes": 10},
    "cifar10": {"in_channels": 3, "image_size": 32, "num_classes": 10},
}


def _mnist_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )


def _cifar10_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )


def _transform_for(dataset: str):
    if dataset == "mnist":
        return _mnist_transform()
    if dataset == "cifar10":
        return _cifar10_transform()
    raise ValueError(f"unknown dataset: {dataset}")


def _dataset_class(dataset: str):
    if dataset == "mnist":
        return datasets.MNIST
    if dataset == "cifar10":
        return datasets.CIFAR10
    raise ValueError(f"unknown dataset: {dataset}")


def get_raw_dataset(dataset: str, data_dir: str, train: bool = True) -> Dataset:
    """指定された dataset の `Dataset` オブジェクトを返す (DataLoader でラップしない)。

    可視化ツールなど、生サンプルへ直接アクセスしたい用途向け。
    画像は [-1, 1] にスケールされる。
    """

    cls = _dataset_class(dataset)
    return cls(
        root=data_dir,
        train=train,
        download=True,
        transform=_transform_for(dataset),
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

    return _split_train_val_loaders(full_dataset, batch_size, num_workers, val_ratio, seed)


def get_cifar10_dataloader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    train: bool = True,
) -> DataLoader:
    """CIFAR10 DataLoader を作る。MNIST と同様 [-1, 1] にスケールする。"""

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=_cifar10_transform(),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )


def get_cifar10_train_val_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """CIFAR10 の train split を train/val に分割した2つの DataLoader を返す。"""

    full_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=_cifar10_transform(),
    )

    return _split_train_val_loaders(full_dataset, batch_size, num_workers, val_ratio, seed)


def _split_train_val_loaders(
    full_dataset: Dataset,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
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


def get_dataloader(
    dataset: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    train: bool = True,
) -> DataLoader:
    if dataset == "mnist":
        return get_mnist_dataloader(data_dir, batch_size, num_workers, train=train)
    if dataset == "cifar10":
        return get_cifar10_dataloader(data_dir, batch_size, num_workers, train=train)
    raise ValueError(f"unknown dataset: {dataset}")


def get_train_val_dataloaders(
    dataset: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    if dataset == "mnist":
        return get_mnist_train_val_dataloaders(
            data_dir, batch_size, num_workers, val_ratio=val_ratio, seed=seed
        )
    if dataset == "cifar10":
        return get_cifar10_train_val_dataloaders(
            data_dir, batch_size, num_workers, val_ratio=val_ratio, seed=seed
        )
    raise ValueError(f"unknown dataset: {dataset}")
