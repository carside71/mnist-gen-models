from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )

    dataset = datasets.MNIST(
        root=data_dir,
        train=train,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )
