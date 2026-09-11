"""
Step 6-7: Gradio app — upload a satellite image, get a calibrated wildfire
risk probability plus a Grad-CAM overlay explaining the prediction.

Self-contained (no imports from scripts/) so this folder can be pushed as-is
to a Hugging Face Space: it expects best_model.pt in the same directory.

Run locally:
    python app/app.py
"""
import os
from pathlib import Path

import gradio as gr
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

# Tolerate near-complete/truncated uploaded images instead of erroring.
ImageFile.LOAD_TRUNCATED_IMAGES = True

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", APP_DIR / "best_model.pt"))
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def build_model(num_classes: int = 2) -> nn.Module:
    return timm.create_model("resnet18", pretrained=False, num_classes=num_classes)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {MODEL_PATH}. "
            "Train on Colab (notebooks/train_colab.ipynb), run scripts/calibrate.py, "
            "then copy best_model.pt next to app.py."
        )
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model = build_model().to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    class_to_idx = ckpt["class_to_idx"]
    temperature = ckpt.get("temperature", 1.0)
    return model, class_to_idx, temperature


MODEL, CLASS_TO_IDX, TEMPERATURE = load_model()
WILDFIRE_IDX = CLASS_TO_IDX["wildfire"]
TARGET_LAYERS = [MODEL.layer4[-1]]


def predict(image: Image.Image):
    if image is None:
        return None, None

    pil_img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    rgb_float = np.array(pil_img).astype(np.float32) / 255.0
    input_tensor = _transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(input_tensor)
        calibrated_probs = F.softmax(logits / TEMPERATURE, dim=1)[0]
        risk_prob = calibrated_probs[WILDFIRE_IDX].item()

    cam = GradCAM(model=MODEL, target_layers=TARGET_LAYERS)
    grayscale_cam = cam(input_tensor=input_tensor,
                         targets=[ClassifierOutputTarget(WILDFIRE_IDX)])[0]
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    label = {"Wildfire risk": risk_prob, "No wildfire": 1 - risk_prob}
    return label, overlay


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Satellite image (350x350, Canadian terrain)"),
    outputs=[
        gr.Label(label="Calibrated risk probability"),
        gr.Image(label="Grad-CAM overlay (highlights regions driving the prediction)"),
    ],
    title="Wildfire Risk Mapping",
    description=(
        "Upload a satellite image tile to get a temperature-calibrated wildfire risk "
        "probability and a Grad-CAM overlay showing which regions drove the prediction. "
        "Trained on Canadian terrain only — see the README for limitations."
    ),
)

if __name__ == "__main__":
    demo.launch()
