#!/usr/bin/env python3
"""Export the checkpoint to ONNX so the model runs with onnxruntime alone.

    python scripts/export_onnx.py                       # -> model/model.onnx
    python scripts/export_onnx.py --checkpoint model/last.pt --out model/model_last.onnx

Input  "image":  float32 (N, 3, 1280, 1280), preprocessed as in beans_common.preprocess
Output "logits": float32 (N, 3, 1280, 1280)
The export is checked against the PyTorch model on a real test image.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beans_common as bc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=bc.repo_root())
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    root = args.root.resolve()
    ckpt = args.checkpoint or root / "model" / "best.pt"
    out = args.out or root / "model" / "model.onnx"

    import torch
    model, info = bc.load_model(ckpt, "cpu")
    dummy = torch.zeros(1, 3, bc.NETWORK_SIZE, bc.NETWORK_SIZE)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, dummy, str(out), opset_version=args.opset,
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      do_constant_folding=True, dynamo=False)
    print(f"[export] wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")

    # verification on a real image
    import onnxruntime as ort
    test_id = bc.read_split(root, "test")[0]
    x, _ = bc.preprocess(bc.load_image_rgb(bc.image_path(root, test_id)))
    with torch.no_grad():
        ref = model(torch.from_numpy(x[None])).numpy()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"image": x[None]})[0]
    diff = float(np.abs(ref - got).max())
    agree = float((ref.argmax(1) == got.argmax(1)).mean())
    print(f"[verify] {test_id}: max |torch - onnx| = {diff:.2e}, argmax agreement = {agree:.6f}")

    bc.write_json(out.with_suffix(".json"), {
        "source_checkpoint": str(ckpt.name), "source_sha256": info["checkpoint_sha256"],
        "epoch": info.get("epoch"), "opset": args.opset, "onnx_sha256": bc.sha256_of(out),
        "input": {"name": "image", "shape": ["batch", 3, bc.NETWORK_SIZE, bc.NETWORK_SIZE], "dtype": "float32"},
        "output": {"name": "logits", "shape": ["batch", 3, bc.NETWORK_SIZE, bc.NETWORK_SIZE], "dtype": "float32"},
        "preprocessing": {
            "resize": f"LongestMaxSize({bc.NETWORK_SIZE}), bilinear; 800x1200 -> 853x1280",
            "pad": f"centre pad to {bc.NETWORK_SIZE}x{bc.NETWORK_SIZE}, image fill 0, mask fill 255",
            "normalize": "x/255 then (x - mean)/std with ImageNet mean/std",
            "mean": bc.IMAGENET_MEAN.tolist(), "std": bc.IMAGENET_STD.tolist(), "channel_order": "RGB",
        },
        "verification": {"image": test_id, "max_abs_diff": diff, "argmax_agreement": agree},
    })
    return 0 if diff < 1e-2 else 1


if __name__ == "__main__":
    sys.exit(main())
