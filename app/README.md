---
title: Wildfire Risk Mapping
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.27.0
app_file: app.py
pinned: false
license: other
---

# Wildfire Risk Mapping

Upload a satellite image tile to get a **temperature-calibrated wildfire risk
probability** plus a **Grad-CAM overlay** showing which regions drove the
prediction.

- **Model:** ResNet18 (timm), fine-tuned on the
  [Wildfire Prediction Dataset](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset)
  (Canadian satellite imagery, ~42,850 images).
- **Calibration:** temperature scaling fit on a held-out validation set.
- **Interpretability:** Grad-CAM on the last convolutional block.

**Limitations:** trained exclusively on Canadian terrain — will not
generalize to other biomes without retraining. Single-image classification,
not a temporal/multi-band forecasting system, and not a substitute for
official fire risk assessments.

Full write-up, training pipeline, and code: see the project repository.
