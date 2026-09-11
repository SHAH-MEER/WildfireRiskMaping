"""
Step 5: Grad-CAM interpretability overlays.

Usage (generate sample overlays for a handful of test images):
    python scripts/gradcam.py --checkpoint models/best_model.pt --num-samples 8
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import get_transforms, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from model import build_model

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "outputs" / "gradcam_samples"


def load_model_for_gradcam(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["class_to_idx"], ckpt.get("temperature", 1.0)


def preprocess_image(pil_img: Image.Image):
    """Returns (input_tensor[1,C,H,W], rgb_float[H,W,3] in [0,1]) for Grad-CAM overlay."""
    pil_img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    rgb_float = np.array(pil_img).astype(np.float32) / 255.0
    tensor = get_transforms(train=False)(pil_img).unsqueeze(0)
    return tensor, rgb_float


def generate_gradcam(model, input_tensor, rgb_float, target_category: int, device: torch.device):
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor.to(device),
                         targets=[ClassifierOutputTarget(target_category)])[0]
    visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    return visualization, grayscale_cam


def run_samples(checkpoint_path: Path, data_dir: Path, num_samples: int = 8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_to_idx, _ = load_model_for_gradcam(checkpoint_path, device)
    wildfire_idx = class_to_idx["wildfire"]

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    test_dir = data_dir / "test"
    picked = []
    for cls in ("wildfire", "nowildfire"):
        cls_dir = test_dir / cls
        files_ = sorted(cls_dir.glob("*"))[: num_samples // 2]
        picked.extend([(f, cls) for f in files_])

    for path, cls in picked:
        img = Image.open(path)
        tensor, rgb_float = preprocess_image(img)
        vis, _ = generate_gradcam(model, tensor, rgb_float, wildfire_idx, device)
        out_path = SAMPLES_DIR / f"{cls}_{path.stem}_gradcam.png"
        Image.fromarray(vis).save(out_path)
        print(f"saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "models" / "best_model.pt"))
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    parser.add_argument("--num-samples", type=int, default=8)
    args = parser.parse_args()
    run_samples(Path(args.checkpoint), Path(args.data_dir), args.num_samples)
