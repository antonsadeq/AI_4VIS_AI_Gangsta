#!/usr/bin/env python3
"""Print the model's module tree with parameter counts and activation shapes.

    python scripts/inspect_model.py                 # table of modules
    python scripts/inspect_model.py --depth 3       # finer granularity
    python scripts/inspect_model.py --csv model/layers.csv

Useful for choosing hook points (feature maps, Grad-CAM targets) in Phase II.
Shapes are for one 1280x1280 input. Module names are the ones you pass to
`dict(model.named_modules())[name].register_forward_hook(...)`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beans_common as bc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=bc.repo_root())
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--depth", type=int, default=2, help="max dotted depth of module names to list")
    ap.add_argument("--csv", type=Path, default=None, help="also write the table as CSV")
    args = ap.parse_args()
    root = args.root.resolve()

    import torch
    model, info = bc.load_model(args.checkpoint or root / "model" / "best.pt", "cpu")
    shapes: Dict[str, str] = {}
    hooks = []
    for name, mod in model.named_modules():
        if name == "":
            continue
        def _hook(m, inp, out, _name=name):
            if isinstance(out, (list, tuple)):
                shapes[_name] = " | ".join(str(tuple(o.shape)) for o in out if hasattr(o, "shape"))
            elif hasattr(out, "shape"):
                shapes[_name] = str(tuple(out.shape))
        hooks.append(mod.register_forward_hook(_hook))
    with torch.no_grad():
        model(torch.zeros(1, 3, bc.NETWORK_SIZE, bc.NETWORK_SIZE))
    for h in hooks:
        h.remove()

    rows: List[Dict] = []
    for name, mod in model.named_modules():
        if name == "" or name.count(".") >= args.depth:
            continue
        n_params = sum(p.numel() for p in mod.parameters())
        rows.append({"module": name, "type": type(mod).__name__, "params": n_params,
                     "output_shape": shapes.get(name, "")})
    width = max(len(r["module"]) for r in rows)
    print(f"checkpoint epoch {info.get('epoch')}  total params {info['n_params']:,}")
    print(f"{'module':<{width}}  {'type':<24} {'params':>10}  output shape (N,C,H,W)")
    for r in rows:
        print(f"{r['module']:<{width}}  {r['type']:<24} {r['params']:>10,}  {r['output_shape']}")
    if args.csv:
        bc.write_csv(args.csv, rows)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
