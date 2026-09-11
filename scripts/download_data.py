"""
Step 1: Pull the Wildfire Prediction Dataset from Kaggle.

Requires a Kaggle API token at ~/.kaggle/kaggle.json (kaggle.com -> Account -> Create New Token).

Usage:
    python scripts/download_data.py
"""
import os
import shutil
import zipfile
from pathlib import Path

DATASET = "abdelghaniaaba/wildfire-prediction-dataset"
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DATA_DIR = ROOT / "data"


def download_and_extract():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {DATASET} ...")
    api.dataset_download_files(DATASET, path=str(RAW_DIR), quiet=False, force=False)

    zip_path = next(RAW_DIR.glob("*.zip"), None)
    if zip_path is None:
        raise FileNotFoundError(f"No zip file found in {RAW_DIR} after download.")

    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)

    organize_splits()


def organize_splits():
    """
    The Kaggle archive extracts to a nested folder containing train/valid/test,
    each with wildfire/nowildfire class subfolders. Flatten that into data/{split}/{class}.
    """
    candidates = list(RAW_DIR.rglob("train"))
    train_dir = next((c for c in candidates if c.is_dir()), None)
    if train_dir is None:
        print("Could not auto-locate a 'train' folder under data/raw — inspect it manually.")
        return

    source_root = train_dir.parent
    for split in ("train", "valid", "test"):
        src = source_root / split
        dst = DATA_DIR / split
        if src.is_dir() and not dst.exists():
            shutil.move(str(src), str(dst))
            print(f"Moved {src} -> {dst}")

    print("Dataset organized under data/{train,valid,test}/{wildfire,nowildfire}/")


if __name__ == "__main__":
    download_and_extract()
