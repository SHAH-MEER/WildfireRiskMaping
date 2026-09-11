"""
PyTorch Dataset/DataLoader for the Wildfire Prediction Dataset.
Class folder layout (ImageFolder-compatible): data/{train,valid,test}/{wildfire,nowildfire}/*.jpg
"""
from pathlib import Path

import torch
from PIL import ImageFile
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# A handful of images in the dataset are truncated by a few bytes; tolerate
# that instead of crashing training (PIL's default is to reject them).
ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloaders(batch_size: int = 32, num_workers: int = 2, data_dir: Path = DATA_DIR):
    train_ds = datasets.ImageFolder(data_dir / "train", transform=get_transforms(train=True))
    valid_ds = datasets.ImageFolder(data_dir / "valid", transform=get_transforms(train=False))
    test_ds = datasets.ImageFolder(data_dir / "test", transform=get_transforms(train=False))

    # ImageFolder assigns class indices alphabetically: nowildfire=0, wildfire=1
    assert train_ds.class_to_idx == valid_ds.class_to_idx == test_ds.class_to_idx

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader, train_ds.class_to_idx


if __name__ == "__main__":
    train_loader, valid_loader, test_loader, class_to_idx = get_dataloaders()
    print("class_to_idx:", class_to_idx)
    print("train batches:", len(train_loader), "| valid:", len(valid_loader), "| test:", len(test_loader))
    x, y = next(iter(train_loader))
    print("batch shape:", x.shape, "labels:", y[:8])
