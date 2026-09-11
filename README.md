# Wildfire Risk Mapping

A wildfire risk classifier from satellite imagery that outputs a **calibrated risk
probability** plus a **Grad-CAM interpretability overlay**, deployed as an
interactive Gradio app.

## Dataset

[Wildfire Prediction Dataset (Satellite Images)](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset)
— Canadian satellite imagery, ~42,850 images at 350x350 px, labeled Wildfire /
No Wildfire, pre-split into train / valid / test. See the Kaggle page for
dataset licensing.

## Model & approach

- **Backbone:** ResNet18 (`timm`), ImageNet-pretrained. Early layers frozen;
  only the last residual block (`layer4`) and the classifier head are
  fine-tuned.
- **Calibration:** Raw softmax outputs are checked for calibration (reliability
  diagram, Brier score) and corrected with single-parameter temperature
  scaling fit on the validation set.
- **Interpretability:** Grad-CAM overlays (`pytorch-grad-cam`) on the last
  convolutional block, shown alongside every prediction.
- **Why this matters:** a bare classifier only says "risk" or "no risk". The
  calibration layer means the output probability can be trusted as an actual
  likelihood rather than an arbitrary confidence score, and the Grad-CAM
  overlay lets a reviewer check *why* the model flagged an image, instead of
  taking the score on faith — together these turn it into a risk-mapping tool
  rather than a black-box tutorial classifier.

## Project layout

```
scripts/
  dataset.py      # PyTorch Dataset/DataLoader
  model.py        # ResNet18 transfer-learning setup
  train.py        # Fine-tuning loop with early stopping
  evaluate.py      # Accuracy / F1 / ROC-AUC / confusion matrix
  calibrate.py     # Temperature scaling + reliability diagrams
  gradcam.py       # Grad-CAM sample overlays
  download_data.py # Kaggle API dataset pull
  verify_data.py   # Class balance + corrupted-image check
notebooks/
  train_colab.ipynb  # Self-contained Day-1 pipeline for a Colab T4 GPU
app/
  app.py            # Gradio interface (also the Hugging Face Space entrypoint)
  requirements.txt  # Inference-only dependencies for the Space
models/             # best_model.pt (gitignored — produced by training)
outputs/            # metrics, calibration plots, Grad-CAM samples (gitignored)
```

## Running the pipeline

1. **Data:** place a Kaggle API token at `~/.kaggle/kaggle.json`, then
   `python scripts/download_data.py` and `python scripts/verify_data.py`.
2. **Train (Colab recommended — free GPU):** open
   `notebooks/train_colab.ipynb` in Colab, run all cells, download
   `best_model.pt` + metrics back into `models/` and `outputs/`.
   Alternatively `python scripts/train.py` locally (CPU-only will be slow).
3. **Evaluate:** `python scripts/evaluate.py`
4. **Calibrate:** `python scripts/calibrate.py`
5. **Grad-CAM samples:** `python scripts/gradcam.py`
6. **Run the app locally:** copy `models/best_model.pt` into `app/`, then
   `python app/app.py`.

## Results

**Baseline (ResNet18, 15 epochs, no early stopping triggered):**

| Metric | Value |
|---|---|
| Accuracy | 0.9875 |
| F1 | 0.9886 |
| ROC-AUC | 0.9992 |

Confusion matrix (test set, rows=true/cols=pred, order=[nowildfire, wildfire]):

```
[[2798,   22],
 [  57, 3423]]
```

Training and validation loss both decreased steadily every epoch (train 0.213 → 0.021,
val 0.110 → 0.040) with no sign of overfitting.

**Calibration (temperature scaling, fit on validation set):**

| | Brier score |
|---|---|
| Before | 0.0094 |
| After (T=1.153) | 0.0093 |

The improvement is modest because the uncalibrated model is already close to
well-calibrated — at 98.75% test accuracy, most predictions sit near 0 or 1,
so there's little probability mass left to correct. Reliability diagrams
(`outputs/calibration/reliability_before.png` / `reliability_after.png`) track
close to the diagonal; the middle bins (0.3-0.6 predicted probability) are
noisier simply because few test examples land there.

## Limitations

- **Geographic scope:** trained exclusively on Canadian satellite imagery.
  It will **not** generalize to other biomes, vegetation types, or imaging
  conditions without retraining on region-specific data.
- **Single-image classification, not forecasting:** this predicts risk from
  one static image. It is not a temporal or multi-band risk forecasting
  system and does not incorporate weather, vegetation moisture, or other
  fire-risk covariates.
- **Not a substitute for official fire risk assessments.**

## Deployment

Deployed to Hugging Face Spaces (Gradio SDK, free tier). Push the contents of
`app/` (plus `best_model.pt`) as the Space root. Public URL: _TBD_.
