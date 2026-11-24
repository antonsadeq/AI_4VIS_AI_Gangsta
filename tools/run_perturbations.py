"""Phase I perturbation probe: run the model on controlled modifications of the test images.

Each condition modifies the 800x1200 RGB input (before the normal preprocessing)
and scores the prediction against the unmodified ground truth. Results go to
predictions/perturbations/<condition>/{per_image.csv, metrics.json} and a
summary table predictions/perturbations/summary.csv.

Usage (from the repository root, GPU recommended, about 5 minutes):
    python tools/run_perturbations.py [--split test] [--device cuda]
"""
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import beans_common as bc          # noqa: E402
import predict as pr               # noqa: E402

RNG_SEED = 48


def gaussian_blur(img, sigma):
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigma)


def brightness(img, factor):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def contrast(img, factor):
    return np.clip((img.astype(np.float32) - 128.0) * factor + 128.0, 0, 255).astype(np.uint8)


def noise(img, sigma, seed):
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def grayscale(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.repeat(g[:, :, None], 3, axis=2)


def downscale(img, factor):
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


# name -> (family, level, function(img, seed))
CONDITIONS = {
    "identity":      ("control",    0.0, lambda im, s: im),
    "hflip":         ("control",    1.0, lambda im, s: im[:, ::-1].copy()),
    "blur_s1":       ("blur",       1.0, lambda im, s: gaussian_blur(im, 1.0)),
    "blur_s2":       ("blur",       2.0, lambda im, s: gaussian_blur(im, 2.0)),
    "blur_s4":       ("blur",       4.0, lambda im, s: gaussian_blur(im, 4.0)),
    "bright_0.6":    ("brightness", 0.6, lambda im, s: brightness(im, 0.6)),
    "bright_0.8":    ("brightness", 0.8, lambda im, s: brightness(im, 0.8)),
    "bright_1.2":    ("brightness", 1.2, lambda im, s: brightness(im, 1.2)),
    "bright_1.4":    ("brightness", 1.4, lambda im, s: brightness(im, 1.4)),
    "contrast_0.6":  ("contrast",   0.6, lambda im, s: contrast(im, 0.6)),
    "contrast_1.4":  ("contrast",   1.4, lambda im, s: contrast(im, 1.4)),
    "noise_10":      ("noise",     10.0, lambda im, s: noise(im, 10, s)),
    "noise_20":      ("noise",     20.0, lambda im, s: noise(im, 20, s)),
    "noise_40":      ("noise",     40.0, lambda im, s: noise(im, 40, s)),
    "grayscale":     ("colour",     1.0, lambda im, s: grayscale(im)),
    "downscale_0.5": ("resolution", 0.5, lambda im, s: downscale(im, 0.5)),
    "downscale_0.25":("resolution", 0.25, lambda im, s: downscale(im, 0.25)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default=None)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "model" / "best.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "predictions" / "perturbations")
    ap.add_argument("--showcase-ids", nargs="*", default=["rose2_haricot_0380", "rose2_haricot_0324", "rose3_haricot_0100"],
                    help="ids whose predicted masks are saved for every condition")
    ap.add_argument("--only", nargs="*", help="subset of condition names")
    args = ap.parse_args()

    backend = pr.TorchBackend(args.checkpoint, args.device, fp16=False)
    ids = bc.read_split(ROOT, args.split)
    conds = {k: v for k, v in CONDITIONS.items() if not args.only or k in args.only}
    print(f"[perturb] {len(ids)} {args.split} images x {len(conds)} conditions on {backend.info.get('device')}")

    images = {i: bc.load_image_rgb(bc.image_path(ROOT, i)) for i in ids}
    masks = {i: bc.load_mask(bc.mask_path(ROOT, i)) for i in ids}
    summary = []
    for ci, (name, (family, level, fn)) in enumerate(conds.items()):
        out = args.out / name
        out.mkdir(parents=True, exist_ok=True)
        cm_orig = np.zeros((3, 3), np.int64); cm_net = np.zeros((3, 3), np.int64)
        rows = []; t0 = time.perf_counter()
        for k, sid in enumerate(ids):
            im = fn(images[sid], RNG_SEED + k)
            x, meta = bc.preprocess(im)
            lg = backend(x[None])[0]
            gt = masks[sid]
            if name == "hflip":                      # un-flip the logits before scoring
                lg = lg[:, :, ::-1].copy()
                # padding is top/bottom only, so the flipped canvas maps onto the same geometry
            cm_net += bc.confusion_matrix(lg.argmax(0), bc.preprocess_mask(gt))
            probs = bc.softmax(bc.postprocess_logits(lg, meta), axis=0)
            pred = probs.argmax(0).astype(np.uint8)
            cm_orig += bc.confusion_matrix(pred, gt)
            rows.append(pr.per_image_row(sid, name, pred, probs, gt))
            if sid in args.showcase_ids:
                bc.save_png(out / "showcase_masks" / f"{sid}.png", pred)
                if ci == 0 or name != "identity":
                    bc.save_png(out / "showcase_inputs" / f"{sid}.jpg", im, rgb=True)
            print(f"\r  {name:15s} {k + 1}/{len(ids)}", end="", flush=True)
        print(f"  ({time.perf_counter() - t0:.0f}s)")
        bc.write_csv(out / "per_image.csv", rows)
        m_net = bc.metrics_from_cm(cm_net); m_orig = bc.metrics_from_cm(cm_orig)
        bc.write_json(out / "metrics.json", {
            "condition": name, "family": family, "level": level, "split": args.split, "n_images": len(ids),
            "network_resolution": m_net, "original_resolution": m_orig,
            "confusion_matrix_network_resolution": cm_net.tolist(),
            "confusion_matrix_original_resolution": cm_orig.tolist(),
            **{k: backend.info[k] for k in ("checkpoint", "checkpoint_sha256", "epoch", "device") if k in backend.info}})
        summary.append({"condition": name, "family": family, "level": level,
                        "mIoU": m_net["mIoU"], "IoU_soil": m_net["IoU_soil"], "IoU_crop": m_net["IoU_crop"],
                        "IoU_weed": m_net["IoU_weed"], "pixel_acc": m_net["pixel_acc"],
                        "prec_crop": m_net["prec_crop"], "rec_crop": m_net["rec_crop"],
                        "prec_weed": m_net["prec_weed"], "rec_weed": m_net["rec_weed"],
                        "soil_as_crop": int(cm_net[0, 1]), "soil_as_weed": int(cm_net[0, 2]),
                        "crop_as_soil": int(cm_net[1, 0]), "weed_as_soil": int(cm_net[2, 0]),
                        "crop_as_weed": int(cm_net[1, 2]), "weed_as_crop": int(cm_net[2, 1]),
                        "pred_pixels_crop": int(cm_net[:, 1].sum()), "pred_pixels_weed": int(cm_net[:, 2].sum()),
                        "mean_confidence": float(np.mean([r["mean_confidence"] for r in rows]))})
        print(f"    mIoU {m_net['mIoU']:.4f}  crop {m_net['IoU_crop']:.4f}  weed {m_net['IoU_weed']:.4f}  "
              f"crop->weed {cm_net[1, 2]:,}  weed->crop {cm_net[2, 1]:,}")
    bc.write_csv(args.out / "summary.csv", summary)
    print("wrote", args.out / "summary.csv")


if __name__ == "__main__":
    main()
