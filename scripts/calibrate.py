"""
Step 4: Temperature scaling calibration.

Fits a single scalar temperature T on the validation set logits to minimize NLL,
then reports Brier score and a reliability diagram before/after scaling.

Usage:
    python scripts/calibrate.py --checkpoint models/best_model.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import build_model

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs" / "calibration"


def collect_logits(model, loader, device, wildfire_idx):
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x).cpu()
            all_logits.append(logits)
            all_labels.append((y == wildfire_idx).long())
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    temperature = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=100)
    nll = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperature.item()


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def reliability_diagram(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10, title: str = "", path: Path = None):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(probs, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_acc, bin_conf, bin_count = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)
        else:
            bin_acc.append(labels[mask].mean())
            bin_conf.append(probs[mask].mean())
        bin_count.append(mask.sum())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.bar(bins[:-1], bin_acc, width=1 / n_bins, align="edge", edgecolor="black", alpha=0.7, label="empirical")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("empirical frequency")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path)
    plt.close(fig)


def calibrate(checkpoint_path: Path, data_dir: Path = ROOT / "data", batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    wildfire_idx = class_to_idx["wildfire"]

    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state"])

    _, valid_loader, test_loader, _ = get_dataloaders(batch_size=batch_size, data_dir=data_dir)

    # Fit temperature on validation set
    val_logits, val_labels = collect_logits(model, valid_loader, device, wildfire_idx)
    temperature = fit_temperature(val_logits, val_labels)
    print(f"fitted temperature: {temperature:.4f}")

    # Evaluate calibration on the test set (uncalibrated vs calibrated)
    test_logits, test_labels = collect_logits(model, test_loader, device, wildfire_idx)
    labels_np = test_labels.numpy()

    probs_raw = F.softmax(test_logits, dim=1)[:, wildfire_idx].numpy()
    probs_calibrated = F.softmax(test_logits / temperature, dim=1)[:, wildfire_idx].numpy()

    brier_before = brier_score(probs_raw, labels_np)
    brier_after = brier_score(probs_calibrated, labels_np)
    print(f"brier score before calibration: {brier_before:.4f}")
    print(f"brier score after calibration:  {brier_after:.4f}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    reliability_diagram(probs_raw, labels_np, title="Before calibration",
                         path=OUTPUTS_DIR / "reliability_before.png")
    reliability_diagram(probs_calibrated, labels_np, title="After temperature scaling",
                         path=OUTPUTS_DIR / "reliability_after.png")

    result = {"temperature": temperature, "brier_before": brier_before, "brier_after": brier_after}
    (OUTPUTS_DIR / "calibration_report.json").write_text(json.dumps(result, indent=2))

    # Persist temperature alongside the model checkpoint for inference use
    ckpt["temperature"] = temperature
    torch.save(ckpt, checkpoint_path)
    print(f"saved temperature into {checkpoint_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "models" / "best_model.pt"))
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    args = parser.parse_args()
    calibrate(Path(args.checkpoint), Path(args.data_dir))
