"""
Step 2-3: Fine-tune ResNet18 on the wildfire dataset with early stopping.
Runs on CPU or GPU (auto-detected) — intended primarily for a Colab T4.

Usage:
    python scripts/train.py --epochs 15 --batch-size 32 --patience 3
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from dataset import get_dataloaders
from model import build_model, trainable_summary

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return avg_loss, acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader, valid_loader, _, class_to_idx = get_dataloaders(
        batch_size=args.batch_size, data_dir=Path(args.data_dir)
    )
    print("class_to_idx:", class_to_idx)

    model = build_model().to(device)
    print(trainable_summary(model))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )

    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    best_f1, epochs_no_improve, history = 0.0, 0, []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_f1 = run_epoch(model, valid_loader, criterion, optimizer, device, train=False)
        dt = time.time() - t0

        print(f"epoch {epoch:02d} ({dt:.0f}s) | train loss {train_loss:.4f} acc {train_acc:.3f} f1 {train_f1:.3f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.3f} f1 {val_f1:.3f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
                         "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1})

        if val_f1 > best_f1:
            best_f1, epochs_no_improve = val_f1, 0
            torch.save({"model_state": model.state_dict(), "class_to_idx": class_to_idx}, MODELS_DIR / "best_model.pt")
            print(f"  -> new best (val f1={val_f1:.3f}), saved to models/best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    (OUTPUTS_DIR / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"done. best val f1: {best_f1:.3f}")


if __name__ == "__main__":
    main()
