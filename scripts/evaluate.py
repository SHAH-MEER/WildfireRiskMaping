"""
Step 3: Evaluate the trained model on the held-out test split.
Reports accuracy, F1, ROC-AUC, and a confusion matrix.

Usage:
    python scripts/evaluate.py --checkpoint models/best_model.pt
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

from dataset import get_dataloaders
from model import build_model

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"


def evaluate(checkpoint_path: Path, data_dir: Path = ROOT / "data", batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, _, test_loader, class_to_idx = get_dataloaders(batch_size=batch_size, data_dir=data_dir)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    wildfire_idx = class_to_idx["wildfire"]

    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)[:, wildfire_idx].cpu()
            preds = logits.argmax(1).cpu()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    binary_labels = [1 if l == wildfire_idx else 0 for l in all_labels]
    binary_preds = [1 if p == wildfire_idx else 0 for p in all_preds]

    acc = accuracy_score(binary_labels, binary_preds)
    f1 = f1_score(binary_labels, binary_preds)
    auc = roc_auc_score(binary_labels, all_probs)
    cm = confusion_matrix(binary_labels, binary_preds).tolist()
    report = classification_report(binary_labels, binary_preds, target_names=["nowildfire", "wildfire"])

    print(f"accuracy:  {acc:.4f}")
    print(f"f1:        {f1:.4f}")
    print(f"roc-auc:   {auc:.4f}")
    print(f"confusion matrix (rows=true, cols=pred, order=[nowildfire, wildfire]):\n{cm}")
    print(report)

    OUTPUTS_DIR.mkdir(exist_ok=True)
    result = {"accuracy": acc, "f1": f1, "roc_auc": auc, "confusion_matrix": cm,
              "classification_report": report, "class_to_idx": class_to_idx}
    (OUTPUTS_DIR / "test_metrics.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "models" / "best_model.pt"))
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    args = parser.parse_args()
    evaluate(Path(args.checkpoint), Path(args.data_dir))
