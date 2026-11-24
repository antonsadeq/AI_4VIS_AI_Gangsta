# Inference and evaluation scripts

Standalone tools for running the trained DeepLabV3+ (MobileNetV2) model on the
beans dataset and scoring the results. They contain no training code and do not
depend on the training repository. Everything resolves paths relative to the
repository root (the parent of `scripts/`), so you can call them from anywhere.

| Script | Needs | Purpose |
|---|---|---|
| `predict.py` | torch + smp, **or** onnxruntime only | Run the model on a split, a list of ids or arbitrary images. Writes masks, optional probabilities and panels, per-image table, confusion matrix, metrics. |
| `evaluate_predictions.py` | numpy + opencv | Score any folder of prediction masks against the ground truth, optionally per variety or per crop presence. |
| `export_onnx.py` | torch + smp + onnx + onnxruntime | Re-create `model/model.onnx` from a checkpoint and verify it. |
| `inspect_model.py` | torch + smp | Module tree with parameter counts and activation shapes (hook points for Phase II). |
| `beans_common.py` | numpy + opencv (+ torch/smp for the model) | Shared library: preprocessing, model building, metrics, colour maps. Import it in your own notebooks. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # pick your CUDA; or CPU wheels
pip install -r scripts/requirements.txt
```

If you only want to run the model and not touch PyTorch, `pip install numpy opencv-python onnxruntime`
is enough for `predict.py --backend onnx` and `evaluate_predictions.py`.

Tested with Python 3.10, torch 2.7.1, segmentation-models-pytorch 0.5.0, onnxruntime 1.23.

## Quick start

```bash
# 1. Reproduce the test-set score (about 1 min on a GPU, a few minutes on CPU)
python scripts/predict.py --split test
#    -> predictions/test/{masks/, per_image.csv, confusion_matrix.csv, metrics.json}

# 2. Same, with softmax probabilities and visual panels
python scripts/predict.py --split test --save-probs --save-panels

# 3. Without PyTorch
python scripts/predict.py --split test --backend onnx

# 4. A hand-picked subset, or your own images (no ground truth, no metrics)
python scripts/predict.py --ids rose2_haricot_0006 rose3_haricot_0100 --out predictions/picked
python scripts/predict.py --images "my_images/*.png" --out predictions/custom

# 5. Re-score saved masks, e.g. per variety or for crop-free images
python scripts/evaluate_predictions.py --pred predictions/test --group variety
python scripts/evaluate_predictions.py --pred predictions/test --group has_crop

# 6. The final-epoch weights instead of the best-validation weights
python scripts/predict.py --split test --checkpoint model/last.pt --out predictions/last_epoch119

# 7. Where to hook in for feature maps / Grad-CAM
python scripts/inspect_model.py --depth 3
```

## Output format

`predictions/<split>/`

- `masks/<id>.png` – uint8 class ids at the original 800x1200 resolution. 0 soil, 1 crop, 2 weed.
  Load with `cv2.imread(path, cv2.IMREAD_GRAYSCALE)`.
- `probs/<id>.npy` – float16 array of shape (3, 800, 1200): softmax probabilities per class,
  same grid as the mask. `np.load(path).astype(np.float32)`.
- `panels/<id>.jpg` – RGB | ground truth | prediction | error map, half resolution.
  Error map colours: orange = crop predicted as soil, magenta = weed predicted as soil,
  yellow = soil predicted as plant, cyan = crop/weed confusion. Overlays tint crop green and weed red.
- `per_image.csv` – one row per image: pixel counts (GT and predicted) per class, per-class IoU /
  precision / recall at original resolution (NaN when the class is absent from both GT and
  prediction), pixel accuracy, confusion counts (`crop_pred_as_weed`, `weed_pred_as_crop`,
  `soil_pred_as_plant`, `plant_pred_as_soil`), mean confidence, fraction of pixels with
  confidence below 0.6, mean entropy in bits, `has_crop`, `has_weed`, `variety`.
- `confusion_matrix.csv` – rows = ground truth, columns = prediction, summed over the split, original resolution.
- `confusion_matrix_network_resolution.csv` – the same on the 1280 grid the model was trained and scored on.
- `metrics.json` – dataset-level metrics in two blocks plus checkpoint hash, device and timing:
  - `network_resolution`: how the training pipeline scored the model (prediction and nearest-resized
    ground truth on the 1280x1280 grid, padding ignored). Reproduces `run_metadata/eval_final.json`.
  - `original_resolution`: scored on the saved 800x1200 masks. Slightly lower (about 0.003 mIoU) because
    the logits are interpolated back to the original grid before the argmax.

## How the model is run (so you can replicate it in your own code)

```python
import sys; sys.path.insert(0, "scripts")
import beans_common as bc, numpy as np, torch

model, info = bc.load_model("model/best.pt", device="cuda")      # smp.DeepLabV3Plus, strict load
img = bc.load_image_rgb(bc.image_path(bc.repo_root(), "rose2_haricot_0012"))
x, meta = bc.preprocess(img)                 # (3, 1280, 1280) float32, ImageNet-normalised
with torch.no_grad():
    logits = model(torch.from_numpy(x[None]).cuda())[0].cpu().numpy()   # (3, 1280, 1280)
probs = bc.softmax(bc.postprocess_logits(logits, meta))              # (3, 800, 1200)
pred = probs.argmax(0)
```

Preprocessing has been checked against the training transform on real images
(max absolute difference 2.4e-7 on the normalised tensor, identical resized masks).

### Hook points for Phase II

`python scripts/inspect_model.py --depth 3` prints every module. Useful ones for a
1280x1280 input:

| module name | output shape | meaning |
|---|---|---|
| `encoder.features.1` | 16 x 640 x 640 | stride-2 low-level features |
| `encoder.features.3` | 24 x 320 x 320 | stride-4 features (fed to the decoder's skip path) |
| `encoder.features.6` | 32 x 160 x 160 | stride-8 |
| `encoder.features.13` | 96 x 80 x 80 | stride-16 |
| `encoder.features.18` | 1280 x 80 x 80 | encoder output (stride 16) |
| `decoder.aspp` | 256 x 80 x 80 | ASPP multi-scale context |
| `decoder.block1` | 48 x 320 x 320 | projected skip features |
| `decoder.block2` | 256 x 320 x 320 | fused decoder features (last layer before the classifier, the usual Grad-CAM target) |
| `segmentation_head.0` | 3 x 320 x 320 | class logits before the final 4x bilinear upsampling |

```python
feats = {}
h = dict(model.named_modules())["decoder.block2"].register_forward_hook(lambda m, i, o: feats.setdefault("block2", o.detach()))
```

## Notes and caveats

- The model was trained with AMP (float16) but evaluated in float32; the scripts default to float32.
  `--fp16` is available for speed and changes metrics at the fourth decimal.
- Soil, crop and weed are scored only where the ground truth is defined; padded border pixels are ignored.
- Per-image IoU for a class that is absent from an image is undefined and left as NaN. Dataset-level
  numbers always come from the summed confusion matrix, never from averaging per-image rows.
- The ONNX model has a fixed spatial size of 1280x1280 (batch dimension dynamic). Its logits agree
  with PyTorch to 2.6e-4 and the argmax agrees on 99.9999 % of pixels on the verification image.
