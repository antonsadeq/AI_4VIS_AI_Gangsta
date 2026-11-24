"""Shared helpers for running the DeepLabV3+ (MobileNetV2) beans model.

This module is self-contained: it only needs numpy, opencv and (for the
PyTorch backend) torch + segmentation_models_pytorch. It re-implements the
deterministic evaluation preprocessing of the training pipeline with plain
OpenCV so that no augmentation library is required.

Preprocessing contract (must match training exactly):
    1. image PNG (800x1200, RGB uint8) -> LongestMaxSize(1280), bilinear
       -> 853x1280
    2. pad to 1280x1280 (centre), image fill 0, mask fill 255 (ignore)
    3. x = (x/255 - ImageNet mean) / ImageNet std, channel-first float32
    4. model -> logits (3, 1280, 1280)
Postprocessing:
    crop the valid 853x1280 window, bilinear-resize logits to 800x1200,
    softmax -> probabilities, argmax -> class mask at original resolution.

Class ids: 0 = soil, 1 = crop (bean), 2 = weed, 255 = ignore (padding).
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

CLASS_NAMES: List[str] = ["soil", "crop", "weed"]
NUM_CLASSES = 3
IGNORE_INDEX = 255
NETWORK_SIZE = 1280
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# RGB colours used for overlays: crop = green, weed = red, soil untinted.
# (The raw gt.png uses blue for crop and green for weed; the class ids are
# what matters, the display colours are only a convention of this package.)
PALETTE_RGB: Dict[int, Tuple[int, int, int]] = {1: (40, 210, 60), 2: (235, 40, 40)}
ERROR_COLOURS_RGB = {
    "correct": (0, 0, 0),
    "fn_crop": (255, 170, 0),    # crop pixel predicted as something else
    "fn_weed": (255, 0, 200),    # weed pixel predicted as something else
    "fp_plant": (255, 255, 0),   # soil pixel predicted as crop/weed
    "swap": (0, 230, 255),       # crop<->weed confusion
}


# ───────────────────────── paths & splits ─────────────────────────────

def repo_root() -> Path:
    """Repository root: the folder that contains data/, splits/, model/ (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def read_split(root: Path, split: str) -> List[str]:
    p = Path(root) / "splits" / f"{split}.txt"
    if not p.is_file():
        raise FileNotFoundError(p)
    return [s.strip() for s in p.read_text().splitlines() if s.strip()]


def read_id_list(path: Path) -> List[str]:
    return [s.strip() for s in Path(path).read_text().splitlines() if s.strip()]


def image_path(root: Path, sample_id: str) -> Path:
    return Path(root) / "data" / "rgb" / "images" / f"{sample_id}.png"


def mask_path(root: Path, sample_id: str) -> Path:
    return Path(root) / "data" / "rgb" / "masks" / f"{sample_id}.png"


def variety_of(sample_id: str) -> str:
    """'rose2_haricot_0006' -> 'rose2_haricot'."""
    return sample_id.rsplit("_", 1)[0]


# ───────────────────────── image I/O ──────────────────────────────────

def load_image_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return m


def save_png(path: Path, array: np.ndarray, rgb: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), array):
        raise IOError(f"could not write {path}")


# ───────────────────────── preprocessing ──────────────────────────────

def _resize_geometry(h: int, w: int, size: int) -> Tuple[int, int, int, int]:
    """Return (new_h, new_w, pad_top, pad_left) for LongestMaxSize + centre pad."""
    scale = size / max(h, w)
    new_h, new_w = round(h * scale), round(w * scale)
    pad_top = int((size - new_h) / 2.0)
    pad_left = int((size - new_w) / 2.0)
    return new_h, new_w, pad_top, pad_left


def preprocess(image_rgb: np.ndarray, size: int = NETWORK_SIZE) -> Tuple[np.ndarray, Dict]:
    """uint8 RGB (H, W, 3) -> float32 (3, size, size) normalised tensor + geometry."""
    h, w = image_rgb.shape[:2]
    new_h, new_w, top, left = _resize_geometry(h, w, size)
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized
    x = canvas.astype(np.float32)
    x = (x - IMAGENET_MEAN * 255.0) * np.float32(1.0) / (IMAGENET_STD * 255.0)
    meta = {"orig_h": h, "orig_w": w, "new_h": new_h, "new_w": new_w,
            "pad_top": top, "pad_left": left, "size": size}
    return np.ascontiguousarray(x.transpose(2, 0, 1)), meta


def preprocess_mask(mask: np.ndarray, size: int = NETWORK_SIZE) -> np.ndarray:
    """GT mask (H, W) -> (size, size) with nearest resize and 255 padding.

    This is the mask the training pipeline scored against, so metrics computed
    on it reproduce the published numbers exactly.
    """
    h, w = mask.shape[:2]
    new_h, new_w, top, left = _resize_geometry(h, w, size)
    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((size, size), IGNORE_INDEX, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def postprocess_logits(logits: np.ndarray, meta: Dict) -> np.ndarray:
    """(C, size, size) logits -> (C, orig_h, orig_w) logits.

    Crops the valid (unpadded) window and bilinearly resizes back to the
    original image grid.
    """
    t, l = meta["pad_top"], meta["pad_left"]
    crop = logits[:, t:t + meta["new_h"], l:l + meta["new_w"]]
    out = np.empty((logits.shape[0], meta["orig_h"], meta["orig_w"]), dtype=np.float32)
    for c in range(logits.shape[0]):
        out[c] = cv2.resize(crop[c].astype(np.float32), (meta["orig_w"], meta["orig_h"]),
                            interpolation=cv2.INTER_LINEAR)
    return out


def softmax(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z, dtype=np.float32)
    return e / e.sum(axis=axis, keepdims=True)


# ───────────────────────── model (PyTorch backend) ────────────────────

def build_model():
    """Architecture exactly as trained: smp DeepLabV3Plus, MobileNetV2 encoder,
    3 input channels, 3 classes, no pretrained download (weights come from
    the checkpoint)."""
    import segmentation_models_pytorch as smp
    return smp.DeepLabV3Plus(encoder_name="mobilenet_v2", encoder_weights=None,
                             in_channels=3, classes=NUM_CLASSES)


def load_model(checkpoint: Path, device: str = "cpu"):
    """Return (model in eval mode on device, checkpoint info dict)."""
    import torch
    blob = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    model = build_model()
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    info = {k: v for k, v in blob.items() if k != "state_dict"} if isinstance(blob, dict) else {}
    info["checkpoint"] = str(checkpoint)
    info["checkpoint_sha256"] = sha256_of(checkpoint)
    info["n_params"] = int(sum(p.numel() for p in model.parameters()))
    return model, info


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ───────────────────────── metrics ────────────────────────────────────

def confusion_matrix(pred: np.ndarray, gt: np.ndarray, num_classes: int = NUM_CLASSES,
                     ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    """cm[t, p] = number of pixels with GT class t predicted as p."""
    p = pred.reshape(-1).astype(np.int64)
    t = gt.reshape(-1).astype(np.int64)
    valid = (t != ignore_index) & (t >= 0) & (t < num_classes) & (p >= 0) & (p < num_classes)
    idx = t[valid] * num_classes + p[valid]
    return np.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def metrics_from_cm(cm: np.ndarray, class_names: Sequence[str] = CLASS_NAMES) -> Dict[str, float]:
    """Same formulas as the training pipeline: IoU/precision/recall/F1 per
    class, pixel accuracy, mIoU and mF1 over classes present in GT, and
    frequency-weighted mIoU."""
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    gt = cm.sum(axis=1)
    pr = cm.sum(axis=0)
    fn, fp = gt - tp, pr - tp
    total = cm.sum()
    eps = 1e-12
    iou = tp / np.maximum(tp + fp + fn, eps)
    prec = tp / np.maximum(tp + fp, eps)
    rec = tp / np.maximum(tp + fn, eps)
    f1 = 2 * prec * rec / np.maximum(prec + rec, eps)
    present = gt > 0
    out: Dict[str, float] = {
        "pixel_acc": float(tp.sum() / max(total, 1.0)),
        "mIoU": float(iou[present].mean()) if present.any() else 0.0,
        "FWmIoU": float(((gt / max(total, 1.0)) * iou).sum()) if total > 0 else 0.0,
        "mF1": float(f1[present].mean()) if present.any() else 0.0,
        "n_pixels": int(total),
    }
    for i, n in enumerate(class_names):
        out[f"IoU_{n}"] = float(iou[i])
        out[f"prec_{n}"] = float(prec[i])
        out[f"rec_{n}"] = float(rec[i])
        out[f"F1_{n}"] = float(f1[i])
        out[f"gt_pixels_{n}"] = int(gt[i])
        out[f"pred_pixels_{n}"] = int(pr[i])
    return out


def save_confusion_csv(cm: np.ndarray, path: Path, class_names: Sequence[str] = CLASS_NAMES) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gt\\pred"] + list(class_names))
        for i, n in enumerate(class_names):
            w.writerow([n] + [int(v) for v in cm[i]])


def write_csv(path: Path, rows: List[Dict], columns: Optional[List[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    columns = columns or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path: Path, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


# ───────────────────────── visualisation ──────────────────────────────

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Class-id mask -> RGB image (soil black, crop green, weed red, ignore grey)."""
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    for cid, col in PALETTE_RGB.items():
        rgb[mask == cid] = col
    rgb[mask == IGNORE_INDEX] = (128, 128, 128)
    return rgb


def overlay(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    tint = image_rgb.copy()
    for cid, col in PALETTE_RGB.items():
        tint[mask == cid] = col
    return cv2.addWeighted(tint, alpha, image_rgb, 1 - alpha, 0)


def error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """RGB map of disagreement types between prediction and ground truth."""
    rgb = np.zeros(gt.shape + (3,), dtype=np.uint8)
    rgb[(gt == 1) & (pred == 0)] = ERROR_COLOURS_RGB["fn_crop"]
    rgb[(gt == 2) & (pred == 0)] = ERROR_COLOURS_RGB["fn_weed"]
    rgb[(gt == 0) & (pred != 0)] = ERROR_COLOURS_RGB["fp_plant"]
    rgb[((gt == 1) & (pred == 2)) | ((gt == 2) & (pred == 1))] = ERROR_COLOURS_RGB["swap"]
    return rgb


def _label(panel: np.ndarray, text: str, subtitle: Optional[str] = None) -> np.ndarray:
    out = panel.copy()
    cv2.putText(out, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(out, subtitle, (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, subtitle, (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_panel(image_rgb: np.ndarray, pred: np.ndarray, gt: Optional[np.ndarray] = None,
               scale: float = 0.5) -> np.ndarray:
    """Side-by-side RGB | GT overlay | prediction overlay | error map (2x2 grid)."""
    panels = [_label(image_rgb, "RGB"), _label(overlay(image_rgb, pred), "prediction")]
    if gt is not None:
        panels.insert(1, _label(overlay(image_rgb, gt), "ground truth"))
        panels.append(_label(error_map(pred, gt), "errors",
                             "orange = missed crop   magenta = missed weed   yellow = false plant   cyan = crop/weed swap"))
    if len(panels) == 4:
        top = np.concatenate(panels[:2], axis=1)
        bottom = np.concatenate(panels[2:], axis=1)
        grid = np.concatenate([top, bottom], axis=0)
    else:
        grid = np.concatenate(panels, axis=1)
    if scale != 1.0:
        grid = cv2.resize(grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return grid
