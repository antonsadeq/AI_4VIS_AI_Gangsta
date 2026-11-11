# Model card — `deeplabv3plus_mobilenetv2_rgb1280`

Semantic segmentation of top-down green-bean field images into soil / crop /
weed. This is the single model analysed in the Vis4AI / AI4Vis project.

## Files

| File | What it is |
|---|---|
| `best.pt` | PyTorch checkpoint from epoch 89, selected by best validation mIoU. Dictionary with `state_dict` (373 tensors, 4 379 027 trainable parameters), `epoch`, `val_metrics`, `config` (full training configuration), `source` (hash of the original training file; the tensors were copied unchanged, only the run-folder name inside `config` was normalised). |
| `last.pt` | Checkpoint from epoch 119, the epoch at which early stopping ended the run. Same layout without `val_metrics`. |
| `model.onnx` | ONNX export of `best.pt` (opset 17). Input `image` float32 (N, 3, 1280, 1280), output `logits` float32 (N, 3, 1280, 1280). Max |torch − onnx| logit difference 2.6e-4 on a real image, argmax agreement 99.9999 %. |
| `model.json` | Export record: hashes, I/O spec, preprocessing recipe, verification numbers. |
| `layers.csv` | Every module (depth 3) with type, parameter count and output shape for a 1280 px input. |

Loading in PyTorch (needs `segmentation-models-pytorch==0.5.0`):

```python
import torch, segmentation_models_pytorch as smp
model = smp.DeepLabV3Plus(encoder_name="mobilenet_v2", encoder_weights=None, in_channels=3, classes=3)
model.load_state_dict(torch.load("model/best.pt", map_location="cpu", weights_only=False)["state_dict"])
model.eval()
```

or simply `beans_common.load_model("model/best.pt", device)` from `scripts/`.

## Architecture

DeepLabV3+ (`segmentation_models_pytorch` 0.5.0) with a MobileNetV2 encoder
(torchvision implementation, ImageNet-pretrained at the start of training),
output stride 16, ASPP with rates 12 / 24 / 36 plus image pooling, decoder
fusing ASPP output with the stride-4 encoder feature (24 → 48 channels),
separable 3x3 conv to 256 channels, 1x1 classifier to 3 channels at stride 4,
bilinear 4x upsampling to the input size.

| Part | Parameters | Output at 1280 px input |
|---|---|---|
| encoder (MobileNetV2 features 0–18) | 2 223 872 | 1280 x 80 x 80 (plus skip 24 x 320 x 320) |
| decoder.aspp | 2 072 064 | 256 x 80 x 80 |
| decoder.block1 / block2 | 1 248 / 81 072 | 48 x 320 x 320 / 256 x 320 x 320 |
| segmentation_head | 771 | 3 x 1280 x 1280 |
| total | 4 379 027 | |

## Training data and recipe

- Dataset: 300 images, two bean varieties, 800 x 1200 px, RGB built from the
  675 / 570 / 450 nm bands of a six-band Airphen camera with a per-image
  1–99 percentile stretch. Split 210 / 30 / 60 (train / val / test), seed 48,
  stratified on crop presence. See the main `README.md`, section 2.
- Input: longest side resized to 1280 px, centre-padded to 1280 x 1280,
  ImageNet normalisation. Padding is ignored in the loss and in the metrics.
- Loss: class-weighted cross-entropy (weights soil 0.137, crop 1.333, weed 1.0,
  median-frequency balancing on the train split) + 0.5 × soft Dice.
- Optimiser: AdamW, lr 3e-4, weight decay 1e-4, 3 warm-up epochs then cosine
  decay to 1e-6 over a 400-epoch horizon, batch size 2, mixed precision,
  gradient clipping at norm 1.0.
- Augmentation: horizontal and vertical flips, 90° rotations, brightness /
  contrast ±0.2, HSV jitter, Gaussian blur.
- Early stopping with patience 30 on val mIoU. The run stopped at epoch 119;
  the best epoch is 89. About 9 minutes on an RTX 5090.
- Training environment: Python 3.10, torch 2.9.0 + CUDA 12.8,
  segmentation-models-pytorch 0.5.0, albumentations 2.0.8, timm 1.0.26.

## Scores

Computed with `scripts/predict.py` from the checkpoints in this folder
(float32, GPU). "Network resolution" is how training scored the model
(1280 grid, padding ignored) and reproduces `run_metadata/eval_final*.json`;
"original resolution" is scored on the saved 800 x 1200 masks.

| Checkpoint | Split | Scoring | mIoU | IoU soil | IoU crop | IoU weed | pixel acc |
|---|---|---|---|---|---|---|---|
| best (ep 89) | test (60) | network | 0.8859 | 0.9645 | 0.8559 | 0.8373 | 0.9672 |
| best (ep 89) | test (60) | original | 0.8833 | 0.9634 | 0.8520 | 0.8343 | 0.9663 |
| best (ep 89) | val (30) | network | 0.8953 | 0.9678 | 0.8403 | 0.8779 | 0.9702 |
| best (ep 89) | val (30) | original | 0.8933 | 0.9669 | 0.8364 | 0.8765 | 0.9696 |
| best (ep 89) | train (210) | network | 0.9021 | 0.9700 | 0.8809 | 0.8552 | 0.9734 |
| best (ep 89) | train (210) | original | 0.8994 | 0.9691 | 0.8761 | 0.8530 | 0.9727 |
| last (ep 119) | test (60) | network | 0.8853 | 0.9639 | 0.8540 | 0.8381 | 0.9668 |
| last (ep 119) | test (60) | original | 0.8829 | 0.9629 | 0.8510 | 0.8348 | 0.9660 |

The train/test gap is small (0.016 mIoU), and the last-epoch weights score
within 0.001 of the selected checkpoint on test.

Test precision / recall at network resolution: soil 0.992 / 0.972,
crop 0.889 / 0.958, weed 0.882 / 0.943. Per variety on test (original
resolution): rose2 mIoU 0.871 (weed IoU 0.787), rose3 mIoU 0.883 (weed IoU
0.850).

Throughput measured during training evaluation on an RTX 5090 at 1280 px,
batch 1, float32: 154 FPS, 564 MB peak GPU memory. On a Quadro RTX 5000 the
scripts take about 0.25 s per image for the forward pass including the
transfer of the full-resolution logits back to the CPU, and about one minute
for the whole test split with all outputs; onnxruntime on CPU takes about
5 s per image.

## Intended use and caveats

- Research model for the course project. Trained on one field, two
  varieties, one camera, one preprocessing; no claim of generalisation to
  other fields or sensors.
- Validation was used for checkpoint selection, so val numbers are
  optimistic; test is the only untouched split.
- Single seed. Two further training sessions with the same seed landed
  within 0.001 val mIoU (see `run_metadata/metrics_all_sessions.csv`), so
  differences below that are noise.
- The model was trained with float16 autocast; inference here is float32.
  Running `predict.py --fp16` changes metrics in the fourth decimal.
- Known behavioural hypotheses (over-segmentation of vegetation, crop/weed
  confusions, variety effect) are listed in the main `README.md`, section 4.
