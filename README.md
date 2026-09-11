# Wildfire Risk Mapping

## What is a wildfire

A wildfire is an uncontrolled fire burning in forests, grasslands, brush, or
other vegetation. It needs three things to start and spread: fuel (dry
vegetation), oxygen, and a heat source such as lightning, a spark, or human
activity. Once burning, wind and terrain can push a fire across huge
distances in a matter of hours, turning a small ignition into a landscape
scale disaster before anyone has time to react.

## Why wildfires are lethal

Wildfires kill and destroy in ways that go far beyond the flame front itself.

- **Direct harm.** People and animals caught near a fast moving fire face
  burns, smoke inhalation, and no time to escape. Wildfires have wiped out
  entire towns within hours.
- **Smoke and air quality.** Wildfire smoke carries fine particulate matter
  that travels hundreds of miles, harming respiratory and cardiovascular
  health for people who never see the fire itself.
- **Property and infrastructure.** Homes, power lines, water systems, and
  roads are destroyed, often leaving communities without basic services for
  months or years.
- **Ecosystem damage.** Severe fires can sterilize soil, destroy habitats,
  and take decades to recover from, especially as fires become more
  intense.
- **Economic cost.** Direct damages, firefighting costs, and lost economic
  activity from major wildfire seasons now run into the tens of billions of
  dollars per year in North America alone.
- **Getting worse.** Longer dry seasons and higher temperatures linked to
  climate change are making wildfires more frequent, more intense, and
  harder to predict.

The common thread is time. A fire that is caught and understood early is
manageable. A fire that grows unnoticed is catastrophic. That is the gap
this project is aimed at.

## Why this project

Most wildfire classifiers stop at a single number: fire or no fire. That is
not enough to act on. This project is built around two things a real risk
tool needs that a plain classifier does not give you.

- **A probability you can trust.** The model is calibrated with temperature
  scaling, so a predicted risk of 80 percent actually behaves like 80
  percent in practice, not an arbitrary confidence score that happens to be
  high. That distinction matters the moment someone has to decide whether a
  prediction is worth acting on.
- **A reason, not just a verdict.** Every prediction comes with a Grad-CAM
  heatmap showing which part of the image the model actually looked at.
  Instead of asking someone to take a black box's word for it, the tool
  shows its work, so a reviewer can sanity check whether the model is
  looking at smoke and burn scars or latching onto something irrelevant.

Put together, calibration plus interpretability is what turns a tutorial
level image classifier into something closer to an actual risk mapping
tool: a probability worth trusting, and a visual explanation worth checking.

## Try it live

**[huggingface.co/spaces/SHAH-MEER/wildfire-risk-mapping](https://huggingface.co/spaces/SHAH-MEER/wildfire-risk-mapping)**

Upload any satellite image tile, or click one of the built in example images
if you do not have a wildfire satellite photo lying around. You will get a
calibrated risk probability and a Grad-CAM overlay side by side.

## Project details

### Dataset

[Wildfire Prediction Dataset (Satellite Images)](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset),
sourced from Kaggle. Canadian satellite imagery, about 42,850 images at
350x350 pixels, labeled Wildfire or No Wildfire, pre-split into train,
valid, and test sets. See the Kaggle page for dataset licensing.

### Model and approach

- **Backbone:** ResNet18 (`timm`), pretrained on ImageNet. Early layers are
  frozen; only the last residual block (`layer4`) and the classifier head
  are fine-tuned.
- **Calibration:** raw softmax outputs are checked for calibration
  (reliability diagram, Brier score) and corrected with single parameter
  temperature scaling fit on the validation set.
- **Interpretability:** Grad-CAM overlays (`pytorch-grad-cam`) on the last
  convolutional block, generated alongside every prediction.

### Results

Baseline (ResNet18, 15 epochs, no early stopping triggered):

| Metric | Value |
|---|---|
| Accuracy | 0.9875 |
| F1 | 0.9886 |
| ROC-AUC | 0.9992 |

Confusion matrix (test set, rows are true labels, columns are predictions,
order is [nowildfire, wildfire]):

```
[[2798,   22],
 [  57, 3423]]
```

Training and validation loss both decreased steadily every epoch (train
0.213 to 0.021, val 0.110 to 0.040) with no sign of overfitting.

Calibration (temperature scaling, fit on the validation set):

| | Brier score |
|---|---|
| Before | 0.0094 |
| After (T=1.153) | 0.0093 |

The improvement is modest because the uncalibrated model is already close
to well-calibrated. At 98.75 percent test accuracy, most predictions sit
near 0 or 1, so there is little probability mass left to correct.
Reliability diagrams (`outputs/calibration/reliability_before.png` and
`reliability_after.png`) track close to the diagonal. The middle bins (0.3
to 0.6 predicted probability) are noisier simply because few test examples
land there.

### Limitations

- **Geographic scope.** Trained exclusively on Canadian satellite imagery.
  It will **not** generalize to other biomes, vegetation types, or imaging
  conditions without retraining on region specific data.
- **Single image classification, not forecasting.** This predicts risk from
  one static image. It is not a temporal or multi band risk forecasting
  system, and it does not incorporate weather, vegetation moisture, or
  other fire risk covariates.
- **Not a substitute for official fire risk assessments.**

### Deployment

Deployed to Hugging Face Spaces on **ZeroGPU** (dynamic A100 allocation,
free tier). Plain always on `cpu-basic` Gradio Spaces now require a PRO
subscription, but ZeroGPU is available without one. This requires the
`spaces` package, and the `predict()` function in `app/app.py` is decorated
with `@spaces.GPU`, since the GPU is only attached to the process for the
duration of that call. The decorator is a no-op outside a ZeroGPU Space, so
`app/app.py` runs identically when you run it locally.

Note: ZeroGPU pins supported `torch` versions (see `app/requirements.txt`).
Check Hugging Face's current supported list before bumping the torch
version used there.

To redeploy: push the contents of `app/` plus `models/best_model.pt` (as
`best_model.pt`) to the Space root, for example via `huggingface_hub`'s
`upload_folder` and `upload_file`.

Verified parity: predictions on held out test images match the local run to
within GPU/CPU floating point rounding (for example 0.99999666 local vs
0.99999666 deployed on a wildfire sample, 0.99942 vs 0.99941 on a
nowildfire sample).

### Project layout

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
  examples/          # Sample images shown in the app so you don't need your own
models/             # best_model.pt (gitignored, produced by training)
outputs/            # metrics, calibration plots, Grad-CAM samples (gitignored)
```

### Running the pipeline

1. **Data:** authenticate with Kaggle, then run
   `python scripts/download_data.py` and `python scripts/verify_data.py`.
   Either place a token at `~/.kaggle/kaggle.json` (Kaggle, Account, Create
   New Token) or, on newer Kaggle CLI versions, save an access token to
   `~/.kaggle/access_token` (Kaggle, Settings, API). Both are supported.
2. **Train.** Colab is recommended for the free GPU: open
   `notebooks/train_colab.ipynb`, run all cells, then download
   `best_model.pt` and the metrics files back into `models/` and
   `outputs/`. Alternatively run `python scripts/train.py` locally, though
   CPU only training will be slow.
3. **Evaluate:** `python scripts/evaluate.py`
4. **Calibrate:** `python scripts/calibrate.py`
5. **Grad-CAM samples:** `python scripts/gradcam.py`
6. **Run the app locally:** copy `models/best_model.pt` into `app/`, then
   run `python app/app.py`.
