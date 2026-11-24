# AI & Visualization Lab WS26 — Team *AI Gangsta*

**Project:** How does a crop / weed segmentation model behave, and what did it learn?
**Model under analysis:** `deeplabv3plus_mobilenetv2_rgb1280` — DeepLabV3+ with a MobileNetV2 encoder
(4.4 M parameters), trained by us to segment top-down images of a green-bean field into
**soil / crop / weed**. Test mIoU 0.886.
**Team situation:** one active member (Anton Sadeq). The other three registered members dropped
the course or never contributed; see `Group_situation` and `workload.json`. There is therefore one
pair, the AI-aided one; the human-only pair does not exist.

## Deliverables in this repository

| Deliverable | Where |
|---|---|
| Phase I analysis (executed notebook, 13 figures) | `notebooks/AI_Gangsta_mid_submission.ipynb` (HTML copy: `notebooks/midpoint_phase1.html`; PNGs: `notebooks/figures/`) |
| AI interaction record, Phase I | `AI_INTERACTION_RECORD.md` |
| Reflection on GenAI usage | `GENAI_REFLECTION.md` |
| Team and workload files | `team-info.json`, `workload.json`, `Group_situation` |
| Model (checkpoints, ONNX export, model card) | `model/` |
| Code to run and score the model, and to reproduce every table and figure | `scripts/`, `tools/` |

## Phase I in one paragraph

Judged from inputs and outputs alone, on the 60 untouched test images: the pixel-weighted score
(mIoU 0.886) hides a per-image median of 0.82 and a minimum of 0.63. The model finds vegetation
**by colour** (a grey input contains no vegetation for it) and over-segments it by a thin halo around
every plant, which is two thirds of all wrong pixels but harmless for spraying. It separates crop
from weed **by fine leaf texture**: when that texture is absent — plants below ~200 px, blurred,
noisy or low-resolution input — it defaults to *weed*, and confidently so (crop → weed is its most
confident error). Small weeds are missed when they are faint, not merely when they are small.
Brightness, contrast and mild noise do not matter. Crop-free images do not produce false crop, and
the "unannotated seedlings" explanation of the weed score, suggested at the midpoint, is contradicted
by the full-split numbers. Details, figures and the expectation-by-expectation verdict are in the
notebook (section 4).

## Repository layout

```
AI_4VIS_AI_Gangsta/
├── README.md, AI_INTERACTION_RECORD.md, GENAI_REFLECTION.md, Group_situation, team-info.json, workload.json
├── notebooks/
│   ├── AI_Gangsta_mid_submission.ipynb   executed Phase I notebook (runs on this repository, no GPU needed)
│   ├── midpoint_phase1.html              static copy
│   └── figures/                          fig01 … fig13 as PNG
├── model/
│   ├── best.pt                           PyTorch checkpoint, epoch 89 (the model under analysis)
│   ├── last.pt                           checkpoint at epoch 119 (end of the run)
│   ├── model.onnx                        ONNX export of best.pt (input "image" N×3×1280×1280 → "logits")
│   ├── model.json, layers.csv            export record; every module with parameters and output shape
│   └── model_card.md                     architecture, data, recipe, scores, caveats
├── scripts/                              standalone tools (own README): predict.py, evaluate_predictions.py,
│                                         export_onnx.py, inspect_model.py, beans_common.py, requirements.txt
├── tools/
│   ├── build_notebook.py                 the notebook's source (generates the .ipynb)
│   ├── run_perturbations.py              the perturbation probe of notebook section 3.6
│   └── make_checksums.py
├── run_metadata/                         training config, per-epoch metrics, val/test scores from training
├── splits/                               train.txt / val.txt / test.txt (210 / 30 / 60 sample ids)
├── predictions/                          produced with scripts/predict.py and tools/run_perturbations.py
│   ├── test/, val/, train/               per_image.csv, confusion matrices, metrics.json, eval/ (grouped scores),
│   │                                     panels/ (test + val; RGB | GT | prediction | error map)
│   ├── last_epoch119/test/               the same for the last checkpoint
│   └── perturbations/<condition>/        per-image and summary scores for 17 input perturbations
└── data/                                 NOT in git (640 MB): rgb/images, rgb/masks, dataset_manifest.csv, class_weights.json
```

Not tracked in git because of size: `data/` and the per-pixel outputs `predictions/**/masks/` and
`predictions/**/probs/`. The notebook needs them; see "Reproducing" below.

## The task, data and model in brief

- **Task.** Pixel-wise segmentation into soil (0), crop (1), weed (2) for spot spraying: the sprayer
  must hit weed and must not hit crop, so crop/weed confusions and missed small weeds are the errors
  that matter, not pixel accuracy (soil is ~81 % of pixels).
- **Data.** 300 annotated images of two bean varieties (`rose2_haricot` 176, `rose3_haricot` 124),
  800 × 1200 px, from a six-band multispectral camera; the public "Multi-Spectral Leaf Segmentation
  for Crop/Weed Identification" dataset (Airphen camera; Gao et al., *Scientific Reports* 2024,
  doi 10.1038/s41598-024-61981-9 describes it). The model input is an RGB rendering of the
  675 / 570 / 450 nm bands with a per-image 1–99 percentile stretch, so absolute brightness is removed.
  Leaf-level polygon annotations were rasterised to class masks. 27 images contain no crop at all.
- **Splits.** Fixed, seed 48, stratified on crop presence: 210 train / 30 val / 60 test. Val drove
  early stopping and checkpoint selection, so test is the only untouched split; the behaviour
  analysis uses test only.
- **Model and training.** `segmentation_models_pytorch` DeepLabV3+, MobileNetV2 encoder
  (ImageNet-initialised), input resized to 1280 px and centre-padded; class-weighted cross-entropy
  (soil 0.137, crop 1.333, weed 1.0) + 0.5 × Dice; AdamW 3e-4, cosine schedule, batch 2, AMP,
  flips / rotations / brightness-contrast / HSV / blur augmentation; early stopping (patience 30)
  ended the run at epoch 119, best epoch 89. Full configuration in `run_metadata/config.json`,
  everything else in `model/model_card.md`.
- **Scores** (test, 60 images, scored as in training): mIoU 0.886, IoU soil 0.964 / crop 0.856 /
  weed 0.837; train 0.902, val 0.895; last epoch 0.885. The scripts reproduce the archived test
  evaluation within 2e-5.

The training code is not part of this repository; the model ships as checkpoints plus an ONNX
export, and `scripts/beans_common.py` re-implements the training-time preprocessing (verified
against it to 2.4e-7), so everything in the analysis can be re-run without it.

## Reproducing

```bash
pip install -r scripts/requirements.txt                   # numpy, opencv, pandas, torch, segmentation-models-pytorch, onnxruntime
# 1. data/  — obtain the 300 preprocessed images and masks (not in git; contact the author, or rebuild
#    from the public dataset with the recipe in model/model_card.md) and place them under data/rgb/
# 2. predictions (about 5 minutes on a GPU, longer on CPU; --backend onnx works without PyTorch)
python scripts/predict.py --split test val --save-probs --save-panels
python scripts/predict.py --split train
python scripts/predict.py --split test --checkpoint model/last.pt --out predictions/last_epoch119
python scripts/evaluate_predictions.py --pred predictions/test --group variety
python scripts/evaluate_predictions.py --pred predictions/test --group has_crop
python tools/run_perturbations.py                         # 17 conditions × 60 images, about 10 minutes on a GPU
# 3. the notebook
python tools/build_notebook.py notebooks/AI_Gangsta_mid_submission.ipynb
cd notebooks && python -m jupyter nbconvert --to notebook --execute --inplace AI_Gangsta_mid_submission.ipynb
```

`scripts/README.md` documents every script, the output formats and the hook points inside the
network that Phase II will use.

## Status

| | |
|---|---|
| Phase I (behaviour, black box) | done; revised after the midpoint (object-level analysis, false-plant anatomy, confidence by error type, perturbation probe, honest verdict table) |
| AI interaction record and GenAI reflection, Phase I | done |
| Phase II (what the model learned) | planned, see notebook section 6 |
| Progress video (≤ 5 min) | link submitted on Moodle |
