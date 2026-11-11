# Crop / weed segmentation on green beans — handover for the Vis4AI / AI4Vis project

This folder is the shared package for both pairs. It describes the task, the
dataset, the one model we analyse, how it was trained and evaluated, and every
file you get. The training code itself is not shared; the model ships as a
checkpoint plus an ONNX export, and `scripts/` contains standalone inference
and evaluation tools (see `scripts/README.md`) so you can run it, re-score it
and extend it without the training repository.

**The model under analysis:** `deeplabv3plus_mobilenetv2_rgb1280` — DeepLabV3+
with a MobileNetV2 encoder, RGB input at 1280 px, 4.4 M parameters, test mIoU
0.886. It is the lightweight baseline of a larger encoder study; the other
runs (ResNet-101, EfficientNet-B0, MobileNetV3, six-band inputs) are held
back and can be added in Phase II for "what did the model learn" comparisons.

Status of the package (10.11.2026):

| Part | Status |
|---|---|
| `README.md` (this file) | ready |
| `model/` (best and last checkpoint, ONNX export, model card, layer table) | ready |
| `scripts/` (inference, evaluation, ONNX export, model inspection; own README) | ready, tested on GPU (torch) and CPU (onnxruntime) |
| `predictions/` (masks for all 300 images, probabilities and panels for val + test, per-image tables, confusion matrices) | ready |
| `notebooks/midpoint_phase1.ipynb` (executed Phase I midpoint notebook with figures; HTML copy alongside) | ready, submitted 11.11.2026 |
| `run_metadata/` (config, per-epoch metrics, validation confusion matrix, val and test scores) | ready |
| `splits/` (train / val / test sample lists) | ready |
| `data/rgb/`, `data/class_weights.json`, `data/dataset_manifest.csv` | ready (rebuilt from the raw TIFFs with the project's preprocessing; splits and class weights verified identical to the training run) |
| `data/raw/` (TIFFs, gt.png, gt.xml per sample, 4.5 GB) | ready |
| `data/s41598-024-61981-9.pdf` (source publication) | ready |
| `SHA256SUMS.txt` | ready (verify a download with `sha256sum -c SHA256SUMS.txt`) |
| training trajectory (checkpoints every few epochs) | not yet, planned for Phase II (section 6) |

Where to start: `predictions/test/per_image.csv` and `predictions/test/panels/`
for the behaviour analysis; `scripts/predict.py` to produce your own
predictions (perturbed inputs, subsets, the last-epoch weights); the hook
table in `scripts/README.md` for anything inside the network.

---

## 1. The task in one paragraph

Pixel-wise semantic segmentation of top-down multispectral images of a green
bean (haricot) field into three classes: **soil (0), crop (1), weed (2)**. The
downstream use is spot spraying: the sprayer must hit weed pixels and must not
hit crop pixels, so crop/weed confusions and boundary errors are the
behaviours that matter, not overall pixel accuracy (soil alone is ~81 % of
pixels, so accuracy is dominated by it).

---

## 2. Dataset

### 2.1 Source material

300 annotated samples from two bean varieties, imaged top-down with a
six-band multispectral camera (Airphen). The raw folders match the public
"Multi-Spectral Leaf Segmentation For Crop/Weed Identification" dataset
(https://b2find.eudat.eu/dataset/3855ede8-6e9c-5c68-a695-9e19d5ec3f24).
The PDF shipped with the raw data, `data/s41598-024-61981-9.pdf`, is
Gao et al., "An accurate semantic segmentation model for bean seedlings and
weeds identification based on improved ERFnet", Scientific Reports 2024
(doi 10.1038/s41598-024-61981-9); it is the reference for the dataset and
for a competing architecture (EPANet), and its dataset section is the best
description of how the images were acquired.

| | |
|---|---|
| Varieties | `rose2_haricot` (176 samples), `rose3_haricot` (124 samples) |
| Sample ID | `<variety>_<NNNN>`, e.g. `rose2_haricot_0006` |
| Raw image | `images.tiff`, 800 x 1200 px, 6 channels, uint16 (roughly 12-bit range) |
| Bands (nm) | 450 (blue), 570 (green), 675 (red), 710 (red edge), 730 (red edge), 850 (NIR) |
| Raw mask | `gt.png`, 807 x 1195 px, colour coded: black = soil, blue = crop, green = weed (the raw file's colours; every overlay produced by this package shows crop in green and weed in red) |
| Polygons | `gt.xml`, CVAT-style polygon annotations with labels `Crop leaf` and `Weed leaf`, one `id` attribute per polygon |
| Extras | `false.png` (false-colour composite, 807 x 1195) and `index.png` (8-bit single-channel, 800 x 1200; an index image, semantics not documented) |

The annotation is at leaf level: in the 7 raw samples inspected there were
1 245 `Crop leaf` polygons versus 3 908 `Weed leaf` polygons, so weeds are many
small objects while crops are fewer larger ones. The XML is worth mining:
it gives you object counts, object sizes and shapes per image, which the
pixel masks do not.

Every sample contains soil and weed, but **27 of the 300 contain no crop at
all** (18 rose2, 9 rose3; IDs come in contiguous runs such as
`rose2_haricot_0100`–`0110`, `0202`–`0212`, `0320`–`0332`). Visual check: they
are bare, often darker soil with tractor tracks and only small weed
seedlings. By split: 19 train, 3 val, 5 test. On these images every crop
pixel the model predicts is a false positive, which makes them a ready-made
probe group. Per-image crop fraction ranges 0–20 % (median 8.7 %); weed
fraction 0.1–68 % (median 4.2 %), so weed density varies far more between
images than crop density does.

### 2.2 Preprocessing applied before training (RGB version)

1. Take bands 675 / 570 / 450 nm as R / G / B.
2. For each image and each band separately: clip to the 1st–99th percentile,
   stretch linearly to 0–255, store as 8-bit PNG (800 x 1200).
3. Resample the 807 x 1195 colour mask to the 800 x 1200 image grid with
   nearest-neighbour, then map colours to class IDs 0 / 1 / 2. Value 255 is
   reserved for "ignore" and only appears in padding added at model-input time.

Because the stretch is per image, global brightness information is removed
before the model ever sees the data. Keep that in mind when you look for
illumination effects: they can only survive as within-image contrast.

### 2.3 Splits

Fixed split, seed 48, stratified on crop presence (so the 27 crop-free
images are spread proportionally). Stem lists are in `splits/`, with the
generator's record in `splits/splits_meta.json`. The split was regenerated
from the raw data on a second machine and matched the training lists
exactly, as did the recomputed class weights.

| Split | Samples | Role |
|---|---|---|
| train | 210 | gradient updates |
| val | 30 | early stopping and checkpoint selection |
| test | 60 | evaluated once, after selection was frozen |

Because val drove early stopping and selection, val metrics are optimistic.
Treat **test** as the only unseen set, and **val** as "lightly seen".

### 2.4 Class balance and class weights

Pixel frequencies on the train split: soil 80.6 %, crop 8.3 %, weed 11.1 %.

The cross-entropy term was weighted with median-frequency balancing computed
on the train split: soil 0.137, crop 1.333, weed 1.0. Crop is the rarest
class and gets the largest weight; this is a candidate explanation for the
crop over-prediction described in section 4.

---

## 3. The model

### 3.1 Architecture

A `segmentation_models_pytorch` DeepLabV3+ with encoder `mobilenet_v2`
(ImageNet-pretrained), 3 input channels, 3 output classes. 4 379 027
trainable parameters.

Data flow: MobileNetV2 encoder → ASPP (atrous spatial pyramid pooling at
output stride 16, with an image-level pooling branch) → decoder that fuses
the ASPP output with a stride-4 encoder feature → 3-channel logits →
bilinear upsampling to the input size.

Useful hook points for Phase II: the five encoder stages (strides 2, 4, 8,
16, 16 for MobileNetV2 in smp), the ASPP output, the decoder output
(256 channels at stride 4), and the final logits. `model/layers.csv` lists
every module with its parameter count and output shape for a 1280 px input,
and `scripts/README.md` has a table of the recommended hook names.

### 3.2 Input contract (implemented in `scripts/beans_common.py`)

1. Load the 8-bit RGB PNG (800 x 1200), channel order RGB.
2. Resize so the longer side is 1280 px, bilinear → 853 x 1280.
3. Pad to 1280 x 1280 with zeros, centred: 213 rows on top, 214 at the
   bottom, no left/right padding. The ground-truth mask gets the same
   geometry with nearest-neighbour resizing and padding value 255 (ignored).
4. Scale to [0, 1], then normalise per channel with ImageNet mean
   (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225).
5. Tensor layout NCHW, float32. Output: logits NCHW with 3 channels, same
   spatial size as the input. Class = argmax over channel; softmax gives
   per-class probabilities.
6. To go back to the image grid, crop the padding away and resize the
   logits bilinearly to 800 x 1200, then softmax / argmax. The scripts do
   this for every saved mask and probability file.

The scripts' preprocessing was checked against the training transform on
real images: the normalised tensors differ by at most 2.4e-7 and the
resized masks are identical, and the test-split score it produces matches
`run_metadata/eval_final.json` to within 2e-5 on every metric.

### 3.3 Training recipe

| | |
|---|---|
| Loss | class-weighted cross-entropy + 0.5 × soft Dice |
| Optimiser | AdamW, lr 3e-4, weight decay 1e-4, gradient clipping at norm 1.0 |
| Schedule | 3 warm-up epochs, then cosine to 1e-6 over a 400-epoch horizon |
| Batch | 2, mixed precision |
| Early stopping | patience 30 epochs on val mIoU; the kept checkpoint is the best-val-mIoU epoch |
| Augmentation | horizontal + vertical flip, 90° rotations, brightness/contrast ±0.2, HSV jitter, Gaussian blur |
| Seed | 48 |

The run stopped after 119 epochs; the best checkpoint is epoch 89. Training
took about 9 minutes on an RTX 5090.

Only two checkpoints exist and both are in `model/`: `best.pt` (epoch 89,
the model under analysis) and `last.pt` (epoch 119, where early stopping
ended the run). Training-trajectory analyses with more intermediate
checkpoints need the planned re-run (section 6).

### 3.4 A free reproducibility datapoint

The same configuration was trained three times with the same seed (the log
was appended; see `run_metadata/metrics_all_sessions.csv`, column `session`):

| Session | Epochs | Best val mIoU (epoch) | Note |
|---|---|---|---|
| 1 | 119 | 0.8944 (89) | complete |
| 2 | 22 | 0.8834 (20) | aborted |
| 3 | 119 | 0.8954 (89) | **the kept model**; `metrics.csv` holds this session |

Sessions 1 and 3 land within 0.001 mIoU with the same best epoch, which says
the recipe is stable but also that per-epoch values wobble by a few
thousandths for no reason; differences smaller than that are noise.

### 3.5 Evaluation protocol

All metrics derive from a single pixel confusion matrix accumulated over a
split (padding pixels excluded). mIoU is the plain mean over the three
classes; FWmIoU weights by class frequency; per-class precision, recall and
F1 are also reported. Throughput (FPS at batch 1), parameter count and peak
GPU memory were measured on an RTX 5090 at 1280 px.

---

## 4. Results and what they already hint at

### 4.1 Scores

Scored as in training (1280 grid, padding ignored; `predictions/<split>/metrics.json`,
block `network_resolution`). The `original_resolution` block, scored on the
saved 800 x 1200 masks, is about 0.003 mIoU lower everywhere.

| Checkpoint | Split | mIoU | IoU soil | IoU crop | IoU weed | pixel acc |
|---|---|---|---|---|---|---|
| best (epoch 89) | train (210) | 0.902 | 0.970 | 0.881 | 0.855 | 0.973 |
| best (epoch 89) | val (30) | 0.895 | 0.968 | 0.840 | 0.878 | 0.970 |
| best (epoch 89) | **test (60)** | **0.886** | 0.964 | 0.856 | 0.837 | 0.967 |
| last (epoch 119) | test (60) | 0.885 | 0.964 | 0.854 | 0.838 | 0.967 |

Test precision / recall: crop 0.889 / 0.958, weed 0.882 / 0.943,
soil 0.992 / 0.972. Throughput 154 FPS at 1280 px, 564 MB peak GPU memory
(RTX 5090). The train/test gap is only 0.016 mIoU, and the last-epoch weights
are within 0.001 of the selected checkpoint, so neither overfitting nor
checkpoint choice is a big factor here.

Test confusion matrix (pixels on the 1280 grid; rows = ground truth, columns =
prediction), `predictions/test/confusion_matrix_network_resolution.csv`:

| | soil | crop | weed |
|---|---|---|---|
| **soil** | 49 941 478 | 521 855 | 909 334 |
| **crop** | 99 692 | 5 547 940 | 143 395 |
| **weed** | 307 574 | 169 348 | 7 869 784 |

Validation confusion matrix at the best epoch, `run_metadata/confusion_matrix_val.csv`:

| | soil | crop | weed |
|---|---|---|---|
| **soil** | 24 482 618 | 244 977 | 400 238 |
| **crop** | 40 814 | 2 345 172 | 77 480 |
| **weed** | 128 535 | 82 305 | 4 953 061 |

Per variety on test (`scripts/evaluate_predictions.py --pred predictions/test --group variety`):
rose2 (33 images) mIoU 0.871 with weed IoU 0.787; rose3 (27 images) mIoU 0.883
with weed IoU 0.850. Soil goes the other way (0.977 vs 0.942).

The per-image picture is much less flattering than the aggregate: image-level
mIoU (over classes present in the image) has median 0.82 and ranges 0.63 to
0.92 on test. Dataset-level scores are pixel-weighted, so large plants in
"easy" images dominate them.

### 4.2 Starting hypotheses

These are visible in the aggregates; the point of Phase I is to find out
where, when and why they happen, so treat them as questions, not findings.

- **Vegetation is over-segmented.** Both crop and weed have recall well above
  precision, and the two largest error cells are soil predicted as weed and
  soil predicted as crop. Boundary dilation, shadows and small vegetation
  fragments are candidate causes.
- **Crop/weed confusion is small in pixels but large in consequence.** About
  160 k val pixels go between crop and weed, versus ~815 k between vegetation
  and soil. Whether those crop/weed errors are scattered pixels or whole
  mislabelled plants is unknown and is what matters for spraying.
- **Weakest class differs between val and test.** On val the worst class is
  crop (0.840); on test it is weed (0.837). With 30 / 60 images, per-image
  variation is large, so look at the per-image distribution, not the mean.
- **Crop precision is the slow learner.** Crop recall is above 0.95 from
  epoch 5 onward, while crop precision climbs from 0.74 (epoch 5) to 0.88
  (epoch 89). The model learns "where vegetation is" fast and "which
  vegetation is crop" slowly. Early checkpoints should therefore over-call
  crop much more than the final one.
- **Val loss and val mIoU disagree late.** Val loss is lowest around epoch
  80 (0.18) and drifts up to 0.19 by epoch 119 while train loss keeps falling
  (0.18 → 0.14) and mIoU stays flat; calibration of the final checkpoint is
  worth checking.
- **Variety as a hidden group.** Nothing was ever stratified or reported by
  rose2 vs rose3. First numbers (section 4.1) show weed IoU 0.79 on rose2
  against 0.85 on rose3, and soil the other way round. The `variety` column
  in `per_image.csv` lets you take this apart.
- **Crop-free images do not fool the model.** On the 5 crop-free test
  images the model predicts exactly zero crop pixels. The expected
  false-positive probe is a non-finding; what these images show instead is
  the weed problem below.
- **Sparse weed is where the model fails.** The six worst test images
  (image mIoU 0.63–0.76) all have little weed (1 k–33 k weed pixels out of
  960 k), and their weed IoU is 0.27–0.50 while crop IoU stays above 0.81.
  Small or isolated weed seedlings, not crop/weed confusion, drive the worst
  cases; the error-map panels in `predictions/test/panels/` show where. On
  the crop-free images (e.g. `rose3_haricot_0100`) many "false plant"
  pixels sit on tiny seedlings that the annotation does not mark at all, so
  part of the measured weed error is probably annotation incompleteness
  rather than model failure. Separating the two (with the polygon XML and
  the raw bands) is a worthwhile Phase I task.
- **Confidence is high almost everywhere.** Mean max-probability is 0.98 on
  test and only 1.2 % of pixels fall below 0.6 confidence. Whether the
  low-confidence pixels coincide with the errors (useful) or not
  (over-confident mistakes) is a direct Phase I question; the probabilities
  in `predictions/test/probs/` are there for it.

---

## 5. Files in this package

```
handover/
├── README.md
├── splits/                     train.txt / val.txt / test.txt (one sample ID per line)
├── run_metadata/
│   ├── config.json             full training configuration
│   ├── metrics.csv             one row per epoch of the kept training session (119 rows)
│   ├── metrics_all_sessions.csv  same, all three sessions, with a `session` column
│   ├── confusion_matrix_val.csv  val confusion matrix at the best epoch
│   ├── val_metrics_best.json   val metrics at the best epoch (from training)
│   ├── eval_final_val.json     val metrics from the final evaluator (+ params, FPS, memory)
│   └── eval_final.json         test metrics from the final evaluator (+ params, FPS, memory)
│
├── data/
│   ├── rgb/images/<id>.png     800x1200 8-bit, the exact model inputs (300)
│   ├── rgb/masks/<id>.png      class IDs 0/1/2 (300)
│   ├── raw/<variety>/<NNNN>/   images.tiff (6-band), gt.png, gt.xml, false.png, index.png (rose3 has no index.png)
│   ├── raw/MANIFEST.TXT        file list of the raw export
│   ├── class_weights.json      median-frequency weights + the train-split pixel counts behind them
│   ├── dataset_manifest.csv    per-sample size, class presence, pixel counts, preprocessing parameters
│   └── s41598-024-61981-9.pdf  source publication
├── SHA256SUMS.txt              checksums of every file in this folder
│
├── model/
│   ├── best.pt                 PyTorch checkpoint, epoch 89 (the model under analysis):
│   │                           {"state_dict", "epoch", "val_metrics", "config", "source"}
│   ├── last.pt                 PyTorch checkpoint, epoch 119 (end of the run)
│   ├── model.onnx              ONNX export of best.pt, input "image" (N,3,1280,1280) → "logits"
│   ├── model.json              export record: hashes, opset, I/O names, preprocessing, verification
│   ├── layers.csv              every module: type, parameter count, output shape at 1280 px
│   └── model_card.md           model card (architecture, data, recipe, scores, intended use, caveats)
│
├── scripts/                    standalone tools, see scripts/README.md
│   ├── predict.py              run the model (torch or onnxruntime) on splits / ids / own images
│   ├── evaluate_predictions.py score any masks/ folder, optional grouping by variety or crop presence
│   ├── export_onnx.py          rebuild model.onnx from a checkpoint and verify it
│   ├── inspect_model.py        module tree with shapes (hook points)
│   ├── beans_common.py         shared library: preprocessing, model, metrics, colours
│   ├── requirements.txt
│   └── README.md
│
├── notebooks/
│   ├── midpoint_phase1.ipynb   executed Phase I midpoint submission (runs on this folder, no GPU needed)
│   ├── midpoint_phase1.html    static copy of the executed notebook
│   └── figures/                the nine figures as PNG
│
└── predictions/                produced with scripts/predict.py and model/best.pt (float32, GPU)
    ├── test/                   60 images
    │   ├── masks/<id>.png      argmax class map, uint8, 800x1200
    │   ├── probs/<id>.npy      softmax probabilities, float16, (3, 800, 1200)
    │   ├── panels/<id>.jpg     RGB | ground truth | prediction | error map
    │   ├── per_image.csv       per-image pixel counts, IoU / precision / recall, confusion
    │   │                       counts, confidence and entropy statistics, variety, has_crop
    │   ├── confusion_matrix.csv                       original resolution
    │   ├── confusion_matrix_network_resolution.csv    1280 grid (as scored in training)
    │   └── metrics.json        dataset-level metrics (both resolutions), checkpoint hash, timing
    ├── val/                    30 images, same content as test/
    ├── train/                  210 images, masks + per_image.csv + metrics (no probs / panels)
    └── last_epoch119/test/     test split predicted with model/last.pt, masks + metrics
```

`metrics.csv` columns: epoch, train_loss, val_loss, val_pixel_acc, val_mIoU,
val_FWmIoU, val_mF1, per-class val IoU / precision / recall, backbone and head
learning rate, train and eval time, best_epoch_so_far.

---

## 6. Known limitations and what is still being produced

- **Single seed, single model.** Everything is seed 48. Comparison models
  (other encoders, six-band input) exist and can be added in Phase II.
- **Small evaluation sets.** 60 test images (≈ 58 M pixels, tens of
  thousands of leaf polygons). Report patterns at pixel, object or patch
  level, where the sample sizes are large; be careful with image-level claims.
- **Val is not clean.** It selected the checkpoint.
- **No training trajectory yet.** A re-run with a checkpoint every 5 epochs
  and predictions on a fixed probe set is planned for Phase II (it is a
  9-minute run); it will be added under `model/trajectory/`.
- **Raw `index.png` semantics are undocumented.** Treat it as an auxiliary
  channel, not as ground truth.
- **The training code is not shared.** The inference and evaluation scripts
  in `scripts/` are, and `beans_common.py` exposes the model, preprocessing
  and metrics as plain functions for your own code. Anything else
  model-related (a different export, more checkpoints, a retrained variant)
  can be requested and will be produced on the training workstation.
- **Reproducibility of the data side has been checked.** Re-running the
  preprocessing on `data/raw/` on a second machine produced masks whose
  train-split class weights and whose seed-48 splits are identical to the
  ones used in training, so `data/rgb/` can be regenerated from `data/raw/`
  if ever needed.
