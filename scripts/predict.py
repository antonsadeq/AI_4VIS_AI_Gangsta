#!/usr/bin/env python3
"""Run the DeepLabV3+ (MobileNetV2) beans model and save predictions + metrics.

Typical use (from the repository root; any working directory works):

    python scripts/predict.py --split test                    # masks + metrics
    python scripts/predict.py --split test --save-probs --save-panels
    python scripts/predict.py --split val test --out predictions
    python scripts/predict.py --ids rose2_haricot_0006 rose3_haricot_0100
    python scripts/predict.py --images my_photos/*.png --out predictions/custom
    python scripts/predict.py --split test --backend onnx     # no torch/smp needed
    python scripts/predict.py --split test --checkpoint model/last.pt --out predictions/test_last

Outputs (in --out, default predictions/<split>):
    masks/<id>.png            uint8 class ids at the original 800x1200 resolution
                              (0 soil, 1 crop, 2 weed)
    probs/<id>.npy            float16 (3, 800, 1200) softmax probabilities  [--save-probs]
    panels/<id>.jpg           RGB | GT | prediction | error map             [--save-panels]
    per_image.csv             one row per image: pixel counts, per-class IoU / precision /
                              recall at original resolution, confidence statistics
    confusion_matrix.csv      summed over the run, original resolution
    metrics.json              dataset-level metrics in two flavours:
                                "network_resolution"  — exactly how the model was scored
                                                        during training (1280 grid, padding ignored);
                                                        reproduces run_metadata/eval_final.json
                                "original_resolution" — on the saved 800x1200 masks
                              plus checkpoint / environment information.

Metrics are only computed when ground-truth masks exist (dataset ids, not --images).
"""
from __future__ import annotations

import argparse
import glob
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beans_common as bc  # noqa: E402


# ───────────────────────── backends ───────────────────────────────────

class TorchBackend:
    def __init__(self, checkpoint: Path, device: Optional[str], fp16: bool):
        import torch
        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.fp16 = fp16 and device.startswith("cuda")
        self.model, self.info = bc.load_model(checkpoint, device)
        self.info["backend"] = "torch"
        self.info["device"] = device
        self.info["autocast_fp16"] = self.fp16
        self.info["torch"] = torch.__version__
        if device.startswith("cuda"):
            self.info["gpu"] = torch.cuda.get_device_name(0)

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = torch.from_numpy(batch).to(self.device, non_blocking=True)
        with torch.no_grad():
            if self.fp16:
                with torch.autocast("cuda", dtype=torch.float16):
                    y = self.model(x)
                y = y.float()
            else:
                y = self.model(x)
        return y.cpu().numpy()


class OnnxBackend:
    def __init__(self, onnx_path: Path, device: Optional[str]):
        import onnxruntime as ort
        providers = ["CPUExecutionProvider"]
        if device is None or device.startswith("cuda"):
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
        self.sess = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.info = {"backend": "onnx", "onnx_model": str(onnx_path),
                     "onnx_sha256": bc.sha256_of(onnx_path),
                     "providers": self.sess.get_providers(),
                     "onnxruntime": ort.__version__}

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.input_name: batch.astype(np.float32)})[0]


# ───────────────────────── per-image bookkeeping ──────────────────────

def per_image_row(sample_id: str, split: str, pred: np.ndarray, probs: np.ndarray,
                  gt: Optional[np.ndarray]) -> Dict:
    conf = probs.max(axis=0)
    row: Dict = {
        "id": sample_id, "variety": bc.variety_of(sample_id), "split": split,
        "height": pred.shape[0], "width": pred.shape[1],
        "pred_pixels_soil": int((pred == 0).sum()),
        "pred_pixels_crop": int((pred == 1).sum()),
        "pred_pixels_weed": int((pred == 2).sum()),
        "mean_confidence": float(conf.mean()),
        "frac_confidence_below_0.6": float((conf < 0.6).mean()),
        "mean_entropy_bits": float(-(probs * np.log2(np.clip(probs, 1e-7, 1))).sum(axis=0).mean()),
    }
    if gt is not None:
        cm = bc.confusion_matrix(pred, gt)
        m = bc.metrics_from_cm(cm)
        row.update({
            "gt_pixels_soil": m["gt_pixels_soil"], "gt_pixels_crop": m["gt_pixels_crop"],
            "gt_pixels_weed": m["gt_pixels_weed"],
            "has_crop": bool(m["gt_pixels_crop"] > 0), "has_weed": bool(m["gt_pixels_weed"] > 0),
            "pixel_acc": m["pixel_acc"], "mIoU_present_classes": m["mIoU"],
        })
        for n in bc.CLASS_NAMES:
            row[f"IoU_{n}"] = m[f"IoU_{n}"] if m[f"gt_pixels_{n}"] + m[f"pred_pixels_{n}"] > 0 else float("nan")
            row[f"prec_{n}"] = m[f"prec_{n}"] if m[f"pred_pixels_{n}"] > 0 else float("nan")
            row[f"rec_{n}"] = m[f"rec_{n}"] if m[f"gt_pixels_{n}"] > 0 else float("nan")
        row["crop_pred_as_weed"] = int(cm[1, 2])
        row["weed_pred_as_crop"] = int(cm[2, 1])
        row["soil_pred_as_plant"] = int(cm[0, 1] + cm[0, 2])
        row["plant_pred_as_soil"] = int(cm[1, 0] + cm[2, 0])
    return row


# ───────────────────────── main loop ──────────────────────────────────

def run(backend, items: List[Dict], out: Path, split_name: str, batch_size: int,
        save_probs: bool, save_panels: bool, panel_scale: float) -> Dict:
    out.mkdir(parents=True, exist_ok=True)
    cm_orig = np.zeros((bc.NUM_CLASSES, bc.NUM_CLASSES), dtype=np.int64)
    cm_net = np.zeros_like(cm_orig)
    rows: List[Dict] = []
    has_gt_any = False
    t_start = time.perf_counter()
    infer_time = 0.0

    for b0 in range(0, len(items), batch_size):
        chunk = items[b0:b0 + batch_size]
        images = [bc.load_image_rgb(it["image"]) for it in chunk]
        pre = [bc.preprocess(im) for im in images]
        batch = np.stack([p[0] for p in pre])
        t0 = time.perf_counter()
        logits = backend(batch)                                   # (B, 3, 1280, 1280)
        infer_time += time.perf_counter() - t0

        for it, im, (x, meta), lg in zip(chunk, images, pre, logits):
            gt = bc.load_mask(it["mask"]) if it.get("mask") else None
            if gt is not None:
                has_gt_any = True
                # canonical scoring on the network grid (how training evaluated)
                cm_net += bc.confusion_matrix(lg.argmax(axis=0), bc.preprocess_mask(gt))
            lg_orig = bc.postprocess_logits(lg, meta)
            probs = bc.softmax(lg_orig, axis=0)
            pred = probs.argmax(axis=0).astype(np.uint8)
            bc.save_png(out / "masks" / f"{it['id']}.png", pred)
            if save_probs:
                (out / "probs").mkdir(exist_ok=True)
                np.save(out / "probs" / f"{it['id']}.npy", probs.astype(np.float16))
            if save_panels:
                panel = bc.make_panel(im, pred, gt, scale=panel_scale)
                bc.save_png(out / "panels" / f"{it['id']}.jpg", panel, rgb=True)
            if gt is not None:
                cm_orig += bc.confusion_matrix(pred, gt)
            rows.append(per_image_row(it["id"], split_name, pred, probs, gt))
        done = min(b0 + batch_size, len(items))
        print(f"\r  {done}/{len(items)} images", end="", flush=True)
    print()

    bc.write_csv(out / "per_image.csv", rows)
    summary: Dict = {
        "split": split_name, "n_images": len(items),
        "wall_time_s": time.perf_counter() - t_start,
        "pure_inference_time_s": infer_time,
        "ms_per_image_inference": 1000.0 * infer_time / max(len(items), 1),
        "class_names": bc.CLASS_NAMES, "ignore_index": bc.IGNORE_INDEX,
        "network_input_size": bc.NETWORK_SIZE,
        **backend.info,
        "python": platform.python_version(), "numpy": np.__version__,
        "outputs": {"masks": "masks/<id>.png", "per_image": "per_image.csv",
                    "probs": "probs/<id>.npy" if save_probs else None,
                    "panels": "panels/<id>.jpg" if save_panels else None},
    }
    if has_gt_any:
        summary["network_resolution"] = bc.metrics_from_cm(cm_net)
        summary["original_resolution"] = bc.metrics_from_cm(cm_orig)
        summary["confusion_matrix_network_resolution"] = cm_net.tolist()
        summary["confusion_matrix_original_resolution"] = cm_orig.tolist()
        bc.save_confusion_csv(cm_orig, out / "confusion_matrix.csv")
        bc.save_confusion_csv(cm_net, out / "confusion_matrix_network_resolution.csv")
        m = summary["network_resolution"]
        print(f"  [{split_name}] network-resolution  mIoU={m['mIoU']:.4f}  "
              f"soil={m['IoU_soil']:.4f} crop={m['IoU_crop']:.4f} weed={m['IoU_weed']:.4f}  "
              f"pixel_acc={m['pixel_acc']:.4f}")
        m = summary["original_resolution"]
        print(f"  [{split_name}] original-resolution mIoU={m['mIoU']:.4f}  "
              f"soil={m['IoU_soil']:.4f} crop={m['IoU_crop']:.4f} weed={m['IoU_weed']:.4f}")
    bc.write_json(out / "metrics.json", summary)
    print(f"  wrote {out}")
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("what to predict (choose one)")
    src.add_argument("--split", nargs="+", choices=["train", "val", "test"],
                     help="dataset split(s) from splits/*.txt")
    src.add_argument("--ids", nargs="+", help="explicit sample ids from the dataset")
    src.add_argument("--id-file", type=Path, help="text file with one sample id per line")
    src.add_argument("--images", nargs="+", help="arbitrary RGB image files / globs (no ground truth)")
    ap.add_argument("--root", type=Path, default=bc.repo_root(),
                    help="repository root containing data/, splits/, model/ (default: parent of scripts/)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="default: <root>/model/best.pt")
    ap.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    ap.add_argument("--onnx", type=Path, default=None, help="default: <root>/model/model.onnx")
    ap.add_argument("--device", default=None, help="cuda | cpu | cuda:0 (default: auto)")
    ap.add_argument("--fp16", action="store_true", help="torch backend: run under float16 autocast")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder (default: <root>/predictions/<split>; required for --ids/--images "
                         "unless --split is given)")
    ap.add_argument("--save-probs", action="store_true", help="also save float16 softmax probabilities")
    ap.add_argument("--save-panels", action="store_true", help="also save RGB|GT|pred|error panels")
    ap.add_argument("--panel-scale", type=float, default=0.5, help="downscale factor for panels")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not any([args.split, args.ids, args.id_file, args.images]):
        print("[fatal] give --split, --ids, --id-file or --images")
        return 2

    if args.backend == "torch":
        backend = TorchBackend(args.checkpoint or root / "model" / "best.pt", args.device, args.fp16)
    else:
        backend = OnnxBackend(args.onnx or root / "model" / "model.onnx", args.device)
    for key in ("checkpoint", "onnx_model"):          # record paths relative to the repository root
        if key in backend.info:
            p = Path(backend.info[key]).resolve()
            backend.info[key] = p.relative_to(root).as_posix() if p.is_relative_to(root) else p.as_posix()
    print(f"[predict] backend={backend.info['backend']}  "
          f"{backend.info.get('device') or backend.info.get('providers')}")
    if "epoch" in backend.info:
        print(f"[predict] checkpoint epoch={backend.info['epoch']}  sha256={backend.info['checkpoint_sha256'][:12]}…")

    jobs: List = []   # (split_name, items, out_dir)
    def dataset_items(ids: List[str]) -> List[Dict]:
        return [{"id": i, "image": bc.image_path(root, i), "mask": bc.mask_path(root, i)} for i in ids]

    if args.split:
        for s in args.split:
            jobs.append((s, dataset_items(bc.read_split(root, s)),
                         (args.out / s) if args.out else root / "predictions" / s))
    ids: List[str] = list(args.ids or [])
    if args.id_file:
        ids += bc.read_id_list(args.id_file)
    if ids:
        jobs.append(("custom_ids", dataset_items(ids), args.out or root / "predictions" / "custom_ids"))
    if args.images:
        files = sorted({Path(f) for pat in args.images for f in glob.glob(pat)} | {Path(p) for p in args.images if Path(p).is_file()})
        if not files:
            print("[fatal] no image files matched --images")
            return 2
        items = [{"id": f.stem, "image": f, "mask": None} for f in files]
        jobs.append(("custom_images", items, args.out or root / "predictions" / "custom_images"))

    for split_name, items, out in jobs:
        print(f"[predict] {split_name}: {len(items)} images -> {out}")
        run(backend, items, out, split_name, args.batch_size, args.save_probs,
            args.save_panels, args.panel_scale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
