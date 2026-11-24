#!/usr/bin/env python3
"""Score saved prediction masks against the ground truth. No GPU / torch needed.

    python scripts/evaluate_predictions.py --pred predictions/test
    python scripts/evaluate_predictions.py --pred predictions/test --ids-file my_subset.txt
    python scripts/evaluate_predictions.py --pred predictions/test --group variety
    python scripts/evaluate_predictions.py --pred predictions/test --group has_crop

Reads <pred>/masks/<id>.png (uint8 class ids, 800x1200) and data/rgb/masks/<id>.png,
accumulates a confusion matrix, and writes <pred>/eval/<name>.json + .csv.
Grouping lets you compare subsets (per variety, crop-free images, ...).

Use this to re-score any prediction set you produce yourself (e.g. after
perturbing inputs, or from another model) as long as it follows the same
masks/<id>.png layout.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beans_common as bc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True, help="prediction folder containing masks/")
    ap.add_argument("--root", type=Path, default=bc.repo_root())
    ap.add_argument("--ids-file", type=Path, default=None, help="restrict to these ids (one per line)")
    ap.add_argument("--group", choices=["none", "variety", "has_crop"], default="none",
                    help="additionally report metrics per group")
    ap.add_argument("--name", default=None, help="output name (default: derived from options)")
    args = ap.parse_args()

    masks_dir = args.pred / "masks"
    if not masks_dir.is_dir():
        print(f"[fatal] {masks_dir} not found")
        return 1
    ids = sorted(p.stem for p in masks_dir.glob("*.png"))
    if args.ids_file:
        keep = set(bc.read_id_list(args.ids_file))
        ids = [i for i in ids if i in keep]
    if not ids:
        print("[fatal] no prediction masks selected")
        return 1

    total = np.zeros((bc.NUM_CLASSES, bc.NUM_CLASSES), dtype=np.int64)
    groups: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros_like(total))
    rows: List[Dict] = []
    for i in ids:
        pred = bc.load_mask(masks_dir / f"{i}.png")
        gt = bc.load_mask(bc.mask_path(args.root, i))
        if pred.shape != gt.shape:
            raise ValueError(f"{i}: prediction {pred.shape} vs ground truth {gt.shape}")
        cm = bc.confusion_matrix(pred, gt)
        total += cm
        if args.group == "variety":
            groups[bc.variety_of(i)] += cm
        elif args.group == "has_crop":
            groups["has_crop" if (gt == 1).any() else "no_crop"] += cm
        m = bc.metrics_from_cm(cm)
        rows.append({"id": i, "variety": bc.variety_of(i), "pixel_acc": m["pixel_acc"],
                     "mIoU_present_classes": m["mIoU"],
                     **{f"IoU_{n}": m[f"IoU_{n}"] for n in bc.CLASS_NAMES},
                     **{f"gt_pixels_{n}": m[f"gt_pixels_{n}"] for n in bc.CLASS_NAMES},
                     **{f"pred_pixels_{n}": m[f"pred_pixels_{n}"] for n in bc.CLASS_NAMES}})

    name = args.name or ("eval" + (f"_{args.ids_file.stem}" if args.ids_file else "") +
                         (f"_by_{args.group}" if args.group != "none" else ""))
    out_dir = args.pred / "eval"
    result = {"n_images": len(ids), "overall": bc.metrics_from_cm(total),
              "confusion_matrix": total.tolist(), "class_names": bc.CLASS_NAMES}
    if groups:
        result["groups"] = {g: {"metrics": bc.metrics_from_cm(cm), "confusion_matrix": cm.tolist()}
                            for g, cm in sorted(groups.items())}
    bc.write_json(out_dir / f"{name}.json", result)
    bc.write_csv(out_dir / f"{name}_per_image.csv", rows)
    bc.save_confusion_csv(total, out_dir / f"{name}_confusion_matrix.csv")

    o = result["overall"]
    print(f"{len(ids)} images  mIoU={o['mIoU']:.4f}  soil={o['IoU_soil']:.4f}  "
          f"crop={o['IoU_crop']:.4f}  weed={o['IoU_weed']:.4f}  pixel_acc={o['pixel_acc']:.4f}")
    for g, cm in sorted(groups.items()):
        m = bc.metrics_from_cm(cm)
        print(f"  {g:>14}: mIoU={m['mIoU']:.4f}  soil={m['IoU_soil']:.4f}  "
              f"crop={m['IoU_crop']:.4f}  weed={m['IoU_weed']:.4f}")
    print(f"wrote {out_dir / name}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
