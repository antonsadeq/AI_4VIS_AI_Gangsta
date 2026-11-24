"""Generate the Phase I notebook (unexecuted) from this script's cells.

Usage (from the repository root):
    python tools/build_notebook.py notebooks/AI_Gangsta_mid_submission.ipynb
    cd notebooks && python -m jupyter nbconvert --to notebook --execute --inplace AI_Gangsta_mid_submission.ipynb

Inputs the notebook needs: data/ (images + masks + manifest), splits/, predictions/{train,val,test}
(from scripts/predict.py), predictions/last_epoch119, predictions/test/eval (evaluate_predictions.py)
and predictions/perturbations (tools/run_perturbations.py).
"""
import sys
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r'''
# How does a crop / weed segmentation model behave? — Phase I (midpoint submission, revised)

**Team:** *AI Gangsta* · **Author:** Anton Sadeq (single active member; see `Group_situation` in the repository root)

**Model under analysis:** DeepLabV3+ with a MobileNetV2 encoder (a convolutional segmentation network, 4.4 M parameters), trained by us to segment top-down field images of green beans into **soil / crop / weed**.

**Phase I question:** how does the model behave, judged only from its inputs and outputs? The model is treated as a black box throughout; nothing in this notebook looks at weights or activations.

This notebook runs entirely on the repository: the dataset, the fixed splits, the trained checkpoint and predictions that were pre-computed with `scripts/predict.py` and `tools/run_perturbations.py`. Re-running it needs numpy, pandas, OpenCV and matplotlib only (no GPU). The companion documents `AI_INTERACTION_RECORD.md` and `GENAI_REFLECTION.md` describe how generative AI was used to produce it.

**Contents**

1. Problem context
2. Questions and expectations
3. Investigation and findings (Q1–Q6)
4. Findings against expectations
5. Reproducibility
6. What is not done yet, and the plan
''')

code(r'''
import json, sys, platform
from pathlib import Path
import numpy as np, pandas as pd, cv2
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import Patch

# locate the repository root (folder that holds splits/, data/, predictions/, scripts/)
ROOT = Path.cwd().resolve()
while not (ROOT / "splits").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
assert (ROOT / "splits").is_dir(), "run this notebook from inside the repository"
sys.path.insert(0, str(ROOT / "scripts"))
import importlib, beans_common as bc
bc = importlib.reload(bc)          # always use the current scripts/beans_common.py, even in a long-running kernel
PRED = ROOT / "predictions"
FIG = Path.cwd() / "figures"; FIG.mkdir(exist_ok=True)

# plotting defaults: one colour per entity, fixed everywhere in this notebook
C = {"soil": "#2a78d6", "crop": "#eb6834", "weed": "#1baf7a",          # classes in charts
     "rose2_haricot": "#2a78d6", "rose3_haricot": "#eb6834",            # varieties
     "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df", "surface": "#fcfcfb"}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
OVL = {"crop": np.array(bc.PALETTE_RGB[1]) / 255, "weed": np.array(bc.PALETTE_RGB[2]) / 255}   # overlay colours (crop green, weed red)
mpl.rcParams.update({"figure.dpi": 110, "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C["ink2"],
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "text.color": C["ink"], "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
    "legend.frameon": False})
pd.set_option("display.max_colwidth", None); pd.set_option("display.width", 160)
def savefig(name): plt.savefig(FIG / f"{name}.png", bbox_inches="tight"); plt.show()
def overlay_legend(fig, x=0.99, y=1.04):
    fig.legend(handles=[Patch(color=OVL["crop"], label="crop"), Patch(color=OVL["weed"], label="weed")], loc="upper right", ncol=2, bbox_to_anchor=(x, y))
print("repository root:", ROOT)
''')

md(r'''
## 1. Problem context

**Task.** Pixel-wise segmentation of top-down images of a green-bean field into soil (0), crop (1) and weed (2). The downstream use is spot spraying: the sprayer must hit weeds and must not hit crop. The behaviours that matter are therefore **crop/weed confusions** and **missed small weeds**, not pixel accuracy.

**Data.** 300 annotated images of two bean varieties (`rose2_haricot`, `rose3_haricot`), 800 × 1200 px, from a six-band multispectral camera; the model sees an RGB rendering of three bands with a per-image contrast stretch, so absolute brightness is removed before the model ever sees an image. Leaf-level polygon annotations were rasterised to masks. Fixed split 210 / 30 / 60 (train / val / test), stratified on crop presence. Val drove early stopping, so **test is the only untouched set** and all behaviour analysis below uses it; train is used only as a source of hypotheses.

**Model.** DeepLabV3+ (MobileNetV2 encoder, ImageNet-initialised), input resized to 1280 px, trained with class-weighted cross-entropy + Dice for 119 epochs; the analysed checkpoint is the best-validation epoch (89). One observation from training that motivates a question below: crop *recall* on validation is above 0.95 from epoch 5 on, while crop *precision* climbs slowly over the whole run — the model learns *where vegetation is* fast and *which vegetation is crop* slowly. Details: `model/model_card.md`, `run_metadata/`.
''')

code(r'''
sid = "rose2_haricot_0380"
img = bc.load_image_rgb(bc.image_path(ROOT, sid)); gt = bc.load_mask(bc.mask_path(ROOT, sid)); pr = bc.load_mask(PRED / "test" / "masks" / f"{sid}.png")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.1))
for ax, im, t in zip(axes, [img, bc.overlay(img, gt, 0.8), bc.overlay(img, pr, 0.8)], ["input (RGB rendering, 800 × 1200)", "annotation", "model output (argmax)"]):
    ax.imshow(im); ax.axis("off"); ax.set_title(t)
overlay_legend(fig, y=1.03)
fig.suptitle(f"Figure 1 · Input and output for one typical test image ({sid}): two crop rows, scattered weeds", x=0.01, y=1.02, ha="left", fontsize=11)
plt.tight_layout(); savefig("fig01_input_output")
''')

code(r'''
man = pd.read_csv(ROOT / "data" / "dataset_manifest.csv")
counts = man["class_pixel_counts"].apply(json.loads)
for k, n in zip(["0", "1", "2"], ["soil", "crop", "weed"]):
    man[f"px_{n}"] = counts.apply(lambda d: d[k])
man["px_total"] = man[["px_soil", "px_crop", "px_weed"]].sum(axis=1)
for n in ["soil", "crop", "weed"]:
    man[f"frac_{n}"] = man[f"px_{n}"] / man["px_total"]
man["variety"] = man["sample_id"].map(bc.variety_of)
split_of = {i: s for s in ["train", "val", "test"] for i in bc.read_split(ROOT, s)}
man["split"] = man["sample_id"].map(split_of)
man["has_crop"] = man["px_crop"] > 0
print("images per variety and split")
display(pd.crosstab(man["variety"], man["split"], margins=True).loc[:, ["train", "val", "test", "All"]])
share = man.groupby("split")[["px_soil", "px_crop", "px_weed"]].sum()
share = (share.div(share.sum(axis=1), axis=0) * 100).round(1).loc[["train", "val", "test"]]
share.columns = ["soil %", "crop %", "weed %"]
share["crop-free images"] = man.groupby("split")["has_crop"].apply(lambda s: int((~s).sum()))
print("pixel share per split, and images without any crop")
display(share)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
bins = np.linspace(0, 0.7, 29)
for ax, cls in zip(axes, ["crop", "weed"]):
    for v in ["rose2_haricot", "rose3_haricot"]:
        ax.hist(man.loc[man.variety == v, f"frac_{cls}"], bins=bins, histtype="step", linewidth=2, color=C[v], label=v.split("_")[0])
    ax.set_title(f"{cls} fraction per image (all 300 images)"); ax.set_xlabel("fraction of image pixels")
axes[0].set_ylabel("images"); axes[0].legend(title="variety", loc="upper right")
fig.suptitle("Figure 2 · Crop cover is narrow (0–20 % per image); weed cover ranges from almost nothing to two thirds of an image", x=0.01, ha="left", fontsize=11)
savefig("fig02_class_fraction_hist")
''')

md(r'''
**Special notes that shape the investigation.** Soil is about 81 % of all pixels, so any pixel-weighted score is dominated by it. Crop appears as a few large plants per image, while "weed" spans two regimes — dense patches and isolated seedlings — so weed density is the main source of variation between images. 27 images contain no crop at all (19 / 3 / 5 in train / val / test); on them every predicted crop pixel is a false positive, a ready-made probe group. Weed annotations are mostly small objects (10 722 weed objects against 1 360 crop objects on the 60 test images, section 3.3), which is why the analysis works at three granularities: image, object and pixel.

## 2. Questions and expectations

| | Question | Expectation before looking | Motivated by |
|---|---|---|---|
| **Q1** | Is the aggregate test score (mIoU 0.886) representative of single images? | No: images with little vegetation will score much lower; the aggregate is pixel-weighted. | Class imbalance and the wide spread of weed cover (Figure 2). |
| **Q2** | Where does the model fail — which images, which objects? | (a) Images with sparse weed fail; (b) crop-free images will show false crop, because crop had the largest loss weight (1.33); (c) small objects are missed; (d) part of the "false plant" error is seedlings that the annotation does not mark. | Pixel shares; the class weights; a first look at error panels during the midpoint. |
| **Q3** | Are crop ↔ weed confusions boundary noise or whole plants? | Whole small plants, since crop precision was the slow learner during training. | Training curves (section 1); spraying cost of a swapped plant. |
| **Q4** | Does the model's confidence flag its errors? | Mostly yes, except for crop/weed swaps, which will be confident. | Validation loss rose late in training while mIoU stayed flat, a sign of over-confidence. |
| **Q5** | Which input properties drive the decisions? (controlled perturbations) | Brightness and contrast: no effect (per-image stretch, augmentation). Blur: hurts small weeds. Removing colour: hurts crop vs weed. | The preprocessing; the augmentation recipe; the fact that leaves of both classes are green. |
| **Q6** | Does the bean variety matter? | A density effect in disguise: rose2 images simply have more sparse-weed cases. | Nothing was stratified by variety; the two varieties differ in weed cover (Figure 2). |

## 3. Investigation and findings

All numbers come from the pre-computed test predictions (`predictions/test/`: argmax masks, softmax probabilities, `per_image.csv`). Each sub-section states the finding in the figure title, then explains it.

### 3.1 Baseline: what the aggregate numbers say
''')

code(r'''
runs = [("best (ep 89)", "train", PRED / "train"), ("best (ep 89)", "val", PRED / "val"),
        ("best (ep 89)", "test", PRED / "test"), ("last (ep 119)", "test", PRED / "last_epoch119" / "test")]
rows = []
for ck, split, d in runs:
    m = json.load(open(d / "metrics.json")); s = m["network_resolution"]
    rows.append({"checkpoint": ck, "split": f"{split} ({m['n_images']})", "mIoU": s["mIoU"], "IoU soil": s["IoU_soil"], "IoU crop": s["IoU_crop"],
                 "IoU weed": s["IoU_weed"], "prec crop": s["prec_crop"], "rec crop": s["rec_crop"], "prec weed": s["prec_weed"], "rec weed": s["rec_weed"]})
print("scores on the 1280 grid, exactly as the model was scored in training (predictions/<split>/metrics.json)")
display(pd.DataFrame(rows).set_index(["checkpoint", "split"]).round(3))
mt = json.load(open(PRED / "test" / "metrics.json"))
cm = np.array(mt["confusion_matrix_network_resolution"], dtype=float)
rown = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(5.6, 4.4))
im = ax.imshow(rown, cmap=mpl.colors.LinearSegmentedColormap.from_list("seq", SEQ), vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{rown[i, j]*100:.1f} %\n{int(cm[i, j]):,}", ha="center", va="center", color="white" if rown[i, j] > 0.5 else C["ink"], fontsize=9)
ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(bc.CLASS_NAMES); ax.set_yticklabels(bc.CLASS_NAMES)
ax.set_xlabel("predicted"); ax.set_ylabel("ground truth"); ax.grid(False)
ax.set_title("Figure 3 · Test confusion matrix: vegetation is over-segmented,\ncrop/weed swaps are a small share of pixels")
plt.colorbar(im, ax=ax, fraction=0.046, label="share of ground-truth row")
savefig("fig03_confusion_test")
err = {"soil → crop": cm[0, 1], "soil → weed": cm[0, 2], "crop → soil": cm[1, 0], "weed → soil": cm[2, 0], "crop → weed": cm[1, 2], "weed → crop": cm[2, 1]}
tot = sum(err.values())
print("share of all wrong test pixels by error type:")
for k, v in sorted(err.items(), key=lambda kv: -kv[1]): print(f"  {k:12s} {v/tot*100:5.1f} %")
''')

md(r'''
Test mIoU is 0.886 (soil 0.964, crop 0.856, weed 0.837); the train/test gap is 0.016 and the last-epoch weights are within 0.001, so neither over-fitting nor checkpoint choice explains anything below. Both vegetation classes have recall above precision, and **two thirds of all wrong pixels are soil predicted as vegetation**. Crop ↔ weed swaps are only 15 % of wrong pixels — but they are the expensive error for spraying, so Q3 looks at them separately.

### 3.2 Q1 — Is the aggregate score representative of single images?
''')

code(r'''
per = pd.concat([pd.read_csv(PRED / s / "per_image.csv") for s in ["train", "val", "test"]], ignore_index=True)
per["split"] = pd.Categorical(per["split"], ["train", "val", "test"])
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(8.5, 4))
for k, split in enumerate(["train", "val", "test"]):
    sub = per[per.split == split]
    for v in ["rose2_haricot", "rose3_haricot"]:
        ss = sub[sub.variety == v]; off = -0.12 if v.startswith("rose2") else 0.12
        ax.scatter(k + off + rng.uniform(-0.08, 0.08, len(ss)), ss.mIoU_present_classes, s=22, color=C[v], alpha=0.75, edgecolor=C["surface"], linewidth=0.6,
                   label=v.split("_")[0] if k == 0 else None)
    agg = json.load(open(PRED / split / "metrics.json"))["original_resolution"]["mIoU"]
    ax.plot([k - 0.3, k + 0.3], [agg, agg], color=C["ink"], lw=2); ax.text(k + 0.32, agg, f"dataset-level {agg:.3f}", va="center", fontsize=9, color=C["ink"])
    ax.text(k, 0.57, f"median image {sub.mIoU_present_classes.median():.2f}\nmin {sub.mIoU_present_classes.min():.2f}", ha="center", fontsize=9, color=C["ink2"])
ax.set_xticks(range(3)); ax.set_xticklabels(["train (210)", "val (30)", "test (60)"]); ax.set_ylim(0.55, 1.0)
ax.set_ylabel("image mIoU (classes present in the image)"); ax.legend(title="variety", loc="upper right", ncol=2)
ax.set_title("Figure 4 · The per-image score sits well below the pixel-weighted dataset score, in every split")
savefig("fig04_per_image_miou")
''')

md(r'''
**Finding Q1 — expectation confirmed.** The dataset-level mIoU is pixel-weighted, so images with large plants dominate it. Per image, the median is about 0.82 and the worst test images fall to 0.63. The spread is the same in train, val and test, so it is a property of the images, not of generalisation. Everything that follows therefore works per image, per object or per pixel, never with the single score.

### 3.3 Q2 — Where does the model fail: which images, which objects?
''')

code(r'''
test = per[per.split == "test"].copy()
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
for ax, cls in zip(axes, ["crop", "weed"]):
    sub = test[test[f"gt_pixels_{cls}"] > 0]
    for v in ["rose2_haricot", "rose3_haricot"]:
        ss = sub[sub.variety == v]
        ax.scatter(ss[f"gt_pixels_{cls}"], ss[f"IoU_{cls}"], s=26, color=C[v], alpha=0.8, edgecolor=C["surface"], linewidth=0.6, label=v.split("_")[0])
    ax.set_xscale("log"); ax.set_xlabel(f"{cls} pixels in the ground truth (log)"); ax.set_title(f"{cls} IoU per test image"); ax.set_ylim(0, 1)
axes[0].set_ylabel("IoU"); axes[1].legend(title="variety", loc="lower right")
fig.suptitle("Figure 5 · Per-class IoU falls with the class's pixel count in the image; sparse weed is the common case, sparse crop the rare one", x=0.01, ha="left", fontsize=11)
savefig("fig05_iou_vs_pixels")
for cls in ["crop", "weed"]:
    sub = test[test[f"gt_pixels_{cls}"] > 0]
    r = np.corrcoef(np.log10(sub[f"gt_pixels_{cls}"]), sub[f"IoU_{cls}"])[0, 1]
    print(f"correlation of {cls} IoU with log {cls} pixel count (test, {len(sub)} images): r = {r:.2f}")
worst = test.nsmallest(6, "mIoU_present_classes")[["id", "mIoU_present_classes", "IoU_crop", "IoU_weed", "gt_pixels_crop", "gt_pixels_weed", "crop_pred_as_weed", "weed_pred_as_crop", "soil_pred_as_plant"]]
print("six worst test images"); display(worst.round(3).set_index("id"))

groups = {}
for name in ["by_variety", "by_has_crop"]:
    g = json.load(open(PRED / "test" / "eval" / f"eval_{name}.json"))["groups"]
    for k, v in g.items():
        m = v["metrics"]; cmg = np.array(v["confusion_matrix"])
        groups[k] = {"images": None, "mIoU": m["mIoU"], "IoU soil": m["IoU_soil"], "IoU crop": m["IoU_crop"], "IoU weed": m["IoU_weed"],
                     "prec weed": m["prec_weed"], "rec weed": m["rec_weed"], "crop px predicted": int(cmg[:, 1].sum())}
groups["rose2_haricot"]["images"] = int((test.variety == "rose2_haricot").sum()); groups["rose3_haricot"]["images"] = int((test.variety == "rose3_haricot").sum())
groups["has_crop"]["images"] = int(test.has_crop.sum()); groups["no_crop"]["images"] = int((~test.has_crop).sum())
print("test metrics by group (scripts/evaluate_predictions.py --group variety | has_crop)")
display(pd.DataFrame(groups).T.astype({"images": int, "crop px predicted": int}).round(3))

# Q6: variety after controlling for weed density — residual of weed IoU from a log-linear fit on pixel count
sub = test[test.gt_pixels_weed > 0]
a, b = np.polyfit(np.log10(sub.gt_pixels_weed), sub.IoU_weed, 1)
res = sub.IoU_weed - (a * np.log10(sub.gt_pixels_weed) + b)
print("Q6  weed IoU residual after removing the pixel-count trend, mean by variety:",
      {v.split("_")[0]: round(float(res[sub.variety == v].mean()), 3) for v in ["rose2_haricot", "rose3_haricot"]},
      "| median weed pixels per image:", {v.split("_")[0]: int(sub.loc[sub.variety == v, "gt_pixels_weed"].median()) for v in ["rose2_haricot", "rose3_haricot"]})
''')

md(r'''
**At image level.** IoU of both vegetation classes tracks the class's pixel count in the image (r ≈ 0.8–0.9 on a log scale): the model is not worse on some *kind* of image, it is worse wherever a class is *sparse*. Crop is sparse in a handful of images, weed in many, which is why the six worst test images all have little weed (weed IoU 0.27–0.50) while their crop IoU stays above 0.8. **(2b) is contradicted:** on the five crop-free test images the model predicts *zero* crop pixels, so the large crop loss weight did not produce false crop. **(Q6)** rose2 has lower weed IoU than rose3 (0.79 vs 0.85), but its test images also have about a third of the weed pixels per image; after removing the pixel-count trend the gap *reverses* (residual +0.04 for rose2, −0.04 for rose3), so the raw variety gap is a weed-density effect, not a variety effect — expectation confirmed, with the caveat that 33 vs 27 images cannot settle a residual of a few hundredths.

Image-level IoU cannot say *what* is lost in a sparse image: a few large plants partly, or many small ones entirely. The next cell scores every annotated plant as an object.
''')

code(r'''
# Object level: every connected component of the ground truth is one "object"; it counts as detected if at least half of its pixels
# are predicted as its own class, swapped if at least half are predicted as vegetation but not as its class, missed otherwise.
# "False plant" blobs: connected components of predicted vegetation that do not touch any annotated vegetation.
# Greenness: excess-green index 2G-R-B, rescaled per image so that the median annotated soil = 0 and the median annotated vegetation = 1.
edges = [0, 50, 200, 1000, 5000, 20000, np.inf]; blabels = ["< 50", "50–200", "200–1k", "1k–5k", "5k–20k", "> 20k"]
obj_rows, fp_rows = [], []
fp_attached_px = 0
for sid in test.id:
    gt = bc.load_mask(bc.mask_path(ROOT, sid)); pr = bc.load_mask(PRED / "test" / "masks" / f"{sid}.png")
    img = bc.load_image_rgb(bc.image_path(ROOT, sid)).astype(np.float32)
    exg = 2 * img[..., 1] - img[..., 0] - img[..., 2]
    s_med, v_med = np.median(exg[gt == 0]), np.median(exg[gt > 0]); green = (exg - s_med) / (v_med - s_med)
    for c, name in [(1, "crop"), (2, "weed")]:
        n, lab, stats, _ = cv2.connectedComponentsWithStats((gt == c).astype(np.uint8), connectivity=8)
        for k in range(1, n):
            m = lab == k; as_c = (pr[m] == c).mean(); as_plant = (pr[m] > 0).mean()
            obj_rows.append({"id": sid, "cls": name, "area": int(stats[k, cv2.CC_STAT_AREA]), "green": float(green[m].mean()),
                             "status": "detected" if as_c >= 0.5 else ("swapped" if as_plant >= 0.5 else "missed")})
    n, lab, stats, _ = cv2.connectedComponentsWithStats((pr > 0).astype(np.uint8), connectivity=8)
    for k in range(1, n):
        m = lab == k
        if (gt[m] > 0).any():
            fp_attached_px += int(((gt == 0) & m).sum())
        else:
            fp_rows.append({"id": sid, "area": int(stats[k, cv2.CC_STAT_AREA]), "green": float(green[m].mean()), "pred": int(np.bincount(pr[m], minlength=3)[1:].argmax() + 1)})
obj = pd.DataFrame(obj_rows); fp = pd.DataFrame(fp_rows)
obj["bin"] = pd.cut(obj.area, edges, labels=blabels, right=False); fp["bin"] = pd.cut(fp.area, edges, labels=blabels, right=False)
print("annotated objects on the 60 test images:", obj.groupby("cls").size().to_dict(),
      "| share of objects below 200 px:", obj.groupby("cls").area.apply(lambda a: round(float((a < 200).mean()), 2)).to_dict(),
      "| share of class pixels in those objects:", (obj[obj.area < 200].groupby("cls").area.sum() / obj.groupby("cls").area.sum()).round(3).to_dict())

fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), sharey=True)
scol = {"detected": C["soil"], "swapped": C["crop"], "missed": "#9a9892"}
for ax, cls in zip(axes, ["crop", "weed"]):
    d = obj[obj.cls == cls]; share = d.groupby("bin", observed=False).status.value_counts(normalize=True).unstack().reindex(blabels).fillna(0)
    cnt = d.groupby("bin", observed=False).size().reindex(blabels).fillna(0).astype(int)
    y = np.arange(len(blabels))[::-1]; left = np.zeros(len(blabels))
    for st in ["detected", "swapped", "missed"]:
        vals = share.get(st, pd.Series(0, index=blabels)).values
        ax.barh(y, vals * 100, left=left, color=scol[st], height=0.66, label=st); left += vals * 100 + 0.4
    for yi, lab_, n_ in zip(y, blabels, cnt.values): ax.text(101.5, yi, f"n = {n_:,}", va="center", fontsize=8.5, color=C["ink2"])
    ax.set_yticks(y); ax.set_yticklabels(blabels); ax.set_xlim(0, 118); ax.set_xlabel("share of annotated objects (%)"); ax.set_title(f"{cls} objects by size (pixels)")
axes[0].set_ylabel("object size")
fig.legend(handles=[Patch(color=scol[s], label=s) for s in ["detected", "swapped", "missed"]], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("Figure 6 · Small annotated plants are the failure: small weeds are missed, small crops are called weed", x=0.01, ha="left", fontsize=11)
plt.tight_layout(); savefig("fig06_object_detection_by_size")
print(obj.groupby(["cls", "bin"], observed=False).status.value_counts(normalize=True).unstack().reindex(columns=["detected", "swapped", "missed"]).fillna(0).round(2))
''')

md(r'''
**At object level (Figure 6).** Objects above about 1 000 px (a leaf of a few millimetres) are found in more than 97 % of cases for both classes. Below 200 px the picture splits by class: **small weeds are missed** (a third of weed objects under 50 px and an eighth of those between 50 and 200 px are not predicted as vegetation at all, and almost none are swapped), while **small crops are called weed** (half of the crop objects between 50 and 200 px and a fifth of those between 200 and 1 000 px are predicted as weed; only the tiniest crops, under 50 px, are also missed, a third of them). The model has learned a size prior: a small green object is a weed. Half of all annotated weed objects are below 50 px, but they hold only 2 % of the weed pixels — which is why the pixel-level weed IoU looks acceptable while the object-level miss rate for seedlings is around 30 %. For spot spraying the object count is the relevant unit, not the pixel count.

The next cell tests expectation (2d): are the "false plant" pixels — soil predicted as vegetation, the largest error family in Figure 3 — unannotated seedlings?
''')

code(r'''
fp_isolated_px = int(fp.area.sum()); fp_total_px = fp_attached_px + fp_isolated_px
small = obj[(obj.cls == "weed") & (obj.area < 200)]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [1, 1.5]})
ax = axes[0]
ax.barh([1, 0], [fp_attached_px / 1e6, fp_isolated_px / 1e6], color=[C["soil"], C["crop"]], height=0.6)
ax.set_yticks([1, 0]); ax.set_yticklabels(["attached to an annotated plant\n(boundary over-segmentation)", "isolated blobs on soil\n(candidate unannotated plants)"])
for yi, v in zip([1, 0], [fp_attached_px, fp_isolated_px]): ax.text(v / 1e6 + 0.02, yi, f"{v/1e6:.2f} M px · {v/fp_total_px*100:.0f} %", va="center", fontsize=9)
ax.set_xlim(0, fp_total_px / 1e6 * 1.45); ax.set_xlabel("soil px predicted as vegetation (millions)"); ax.set_title("where the false-plant pixels are")
ax = axes[1]
bins = np.linspace(-0.5, 1.5, 41)
for d, lab_, col in [(small[small.status == "detected"].green, "annotated small weeds (< 200 px), detected", C["soil"]),
                     (small[small.status == "missed"].green, "annotated small weeds (< 200 px), missed", "#9a9892"),
                     (fp.green, "isolated false-plant blobs", C["crop"])]:
    ax.hist(d, bins=bins, histtype="step", lw=2, density=True, color=col, label=f"{lab_} (n = {len(d):,})")
ax.axvline(0, color=C["ink2"], lw=1, ls=":"); ax.axvline(1, color=C["ink2"], lw=1, ls=":")
ax.text(-0.02, ax.get_ylim()[1] * 0.55, "typical\nsoil", ha="right", va="center", fontsize=8, color=C["ink2"]); ax.text(1.02, ax.get_ylim()[1] * 0.55, "typical\nplant", ha="left", va="center", fontsize=8, color=C["ink2"])
ax.set_xlabel("greenness of the object (excess-green index, rescaled per image)"); ax.set_ylabel("density"); ax.set_title("how green are they?")
ax.legend(loc="upper right", fontsize=8)
fig.suptitle("Figure 7 · The false-plant error is boundary dilation, not hallucinated plants; the isolated blobs are as faint as the missed seedlings", x=0.01, ha="left", fontsize=11)
plt.tight_layout(); savefig("fig07_false_plants")
print(f"isolated false-plant blobs: {len(fp):,} (median {int(fp.area.median())} px; {(fp.area < 50).mean()*100:.0f} % below 50 px; predicted as weed in {(fp.pred == 2).mean()*100:.0f} %)")
print("median greenness — detected small weeds: %.2f, missed small weeds: %.2f, isolated false-plant blobs: %.2f" % (
      small[small.status == "detected"].green.median(), small[small.status == "missed"].green.median(), fp.green.median()))
print("share with greenness above 0.5 — detected: %.0f %%, missed: %.0f %%, false-plant blobs: %.0f %%" % (
      (small[small.status == "detected"].green > 0.5).mean() * 100, (small[small.status == "missed"].green > 0.5).mean() * 100, (fp.green > 0.5).mean() * 100))
top = fp.groupby("id").agg(blobs=("area", "size"), px=("area", "sum"), green_above_0_5=("green", lambda g: (g > 0.5).mean())).sort_values("px", ascending=False).head(5)
print("images with the most isolated false-plant pixels:"); display(top.round(2))
''')

md(r'''
**Finding (2d) — expectation contradicted in magnitude.** The midpoint version of this notebook suggested, from one image, that much of the weed error could be seedlings the annotation does not mark. Over the whole test split that is not what the pixels say: **96 % of the false-plant pixels are attached to an annotated plant** — the thin band of over-segmentation around every leaf — and the isolated blobs that *could* be unannotated plants are 4 % of the false-plant pixels (0.1 % of all pixels). The greenness check (right panel) does not rescue the hypothesis either: the isolated blobs are as faint as the small weeds the model *misses*, and clearly fainter than the small weeds it detects; only on one crop-free image (`rose3_haricot_0100`, shown in Figure 13) is a sizeable share of the blobs as green as a plant. So annotation incompleteness exists but is a per-image curiosity, not an explanation of the weed score.

Two further points follow from the greenness panel. The model's misses are not random: **missed small weeds are the faint ones** (median greenness 0.2 versus 0.4 for detected ones) — a colour-signal threshold rather than a size threshold alone. And the "false plant" family, being boundary dilation, is cheap for spraying: a wider spray margin around a real plant.

### 3.4 Q3 — Are crop ↔ weed confusions boundary noise or whole plants?
''')

code(r'''
kernel = np.ones((3, 3), np.uint8)
types = [("soil → crop", 0, 1), ("soil → weed", 0, 2), ("crop → soil", 1, 0), ("weed → soil", 2, 0), ("crop → weed", 1, 2), ("weed → crop", 2, 1)]
acc = {t[0]: {"near": 0, "far": 0} for t in types}
BAND = 3  # pixels
for sid in test.id:
    gt = bc.load_mask(bc.mask_path(ROOT, sid)); pr = bc.load_mask(PRED / "test" / "masks" / f"{sid}.png")
    edge = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, kernel) > 0
    near = cv2.dilate(edge.astype(np.uint8), np.ones((2 * BAND + 1, 2 * BAND + 1), np.uint8)) > 0
    for name, g, p in types:
        e = (gt == g) & (pr == p); acc[name]["near"] += int((e & near).sum()); acc[name]["far"] += int((e & ~near).sum())
bd = pd.DataFrame(acc).T; bd["total"] = bd.near + bd.far; bd["near share"] = bd.near / bd.total
fig, ax = plt.subplots(figsize=(8.5, 3.6))
y = np.arange(len(bd))[::-1]
ax.barh(y, bd.near / 1e6, color=C["soil"], height=0.62, label=f"within {BAND} px of a ground-truth boundary")
ax.barh(y, bd.far / 1e6, left=bd.near / 1e6 + 0.004, color=C["crop"], height=0.62, label="further away (object-level error)")
for yi, (name, row) in zip(y, bd.iterrows()):
    ax.text(row.total / 1e6 + 0.02, yi, f"{row.total/1e6:.2f} M px · {row['near share']*100:.0f} % at boundary", va="center", fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(bd.index); ax.set_xlabel("wrong pixels on the test split (millions)"); ax.set_xlim(0, bd.total.max() / 1e6 * 1.6)
ax.legend(loc="lower right"); ax.set_title("Figure 8 · Soil/vegetation errors are boundary pixels; crop ↔ weed swaps have the largest interior share")
savefig("fig08_error_types_boundary")
print(f"all errors: {bd.total.sum()/1e6:.2f} M px, {bd.near.sum()/bd.total.sum()*100:.0f} % within {BAND} px of a boundary")
''')

md(r'''
**Finding Q3 — expectation confirmed.** 93 % of all wrong test pixels lie within 3 px of an annotation boundary. The soil/vegetation families are almost entirely that band (90–100 % at the boundary). The crop ↔ weed swaps are the exception: a fifth to a quarter of their pixels are interior pixels, i.e. parts of plants or whole plants labelled as the other class. Figure 6 already showed which plants: the small crops. Together: **the expensive error is "small crop called weed", and it is an object-level error**, not a boundary artefact.

### 3.5 Q4 — Does the model's confidence flag its errors?
''')

code(r'''
edges_c = np.array([1/3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0 + 1e-6])
n_pix = np.zeros(len(edges_c) - 1); n_err = np.zeros(len(edges_c) - 1); conf_sum = np.zeros(len(edges_c) - 1)
type_hist = {t[0]: np.zeros(len(edges_c) - 1) for t in types}
for sid in test.id:
    probs = np.load(PRED / "test" / "probs" / f"{sid}.npy").astype(np.float32)
    gt = bc.load_mask(bc.mask_path(ROOT, sid)); pr = probs.argmax(axis=0); conf = probs.max(axis=0); wrong = pr != gt
    b = np.clip(np.searchsorted(edges_c, conf, side="right") - 1, 0, len(edges_c) - 2)
    n_pix += np.bincount(b.ravel(), minlength=len(edges_c) - 1)
    n_err += np.bincount(b.ravel(), weights=wrong.ravel(), minlength=len(edges_c) - 1)
    conf_sum += np.bincount(b.ravel(), weights=conf.ravel(), minlength=len(edges_c) - 1)
    for name, g, p in types:
        type_hist[name] += np.histogram(conf[(gt == g) & (pr == p)], bins=edges_c)[0]
err_rate = n_err / np.maximum(n_pix, 1); mean_conf = conf_sum / np.maximum(n_pix, 1); share = n_pix / n_pix.sum()
ece = float((share * np.abs((1 - err_rate) - mean_conf)).sum())
labels = [f"{a:.2f}–{b:.2f}" for a, b in zip(edges_c[:-1], edges_c[1:])]; labels[0] = "0.33–0.50"; labels[-1] = "0.99–1.00"
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axes[0]; x = np.arange(len(labels))
ax.bar(x, err_rate * 100, color=C["soil"], width=0.7)
ax.plot(x, (1 - mean_conf) * 100, color=C["crop"], lw=2, marker="o", ms=6, label="expected error if calibrated (1 − confidence)")
for xi, (e_, s_) in enumerate(zip(err_rate, share)): ax.text(xi, e_ * 100 + 1.5, f"{s_*100:.1f} % of px", ha="center", fontsize=8, color=C["ink2"])
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8); ax.set_ylabel("observed error rate (%)")
ax.set_title("error rate per confidence bin (test)"); ax.legend(loc="upper right", fontsize=8)
ax = axes[1]
cum_err = np.cumsum(n_err) / n_err.sum(); cum_pix = np.cumsum(n_pix) / n_pix.sum()
ax.plot(cum_pix * 100, cum_err * 100, color=C["soil"], lw=2, marker="o", ms=6)
for xi, yi, lab_ in zip(cum_pix * 100, cum_err * 100, edges_c[1:]):
    ax.annotate(f"conf < {lab_:.2f}", (xi, yi), textcoords="offset points", xytext=(6, -10), fontsize=8, color=C["ink2"])
ax.plot([0, 100], [0, 100], color=C["grid"], lw=1); ax.set_xlabel("least confident pixels flagged (%)"); ax.set_ylabel("errors caught (%)")
ax.set_title("how many errors a confidence threshold would catch"); ax.set_xscale("log"); ax.set_xlim(0.3, 100)
fig.suptitle(f"Figure 9 · Calibration on the test split (expected calibration error {ece:.3f}): the low-confidence tail holds most errors", x=0.01, ha="left", fontsize=11)
plt.tight_layout(); savefig("fig09_calibration")
print(f"pixels with confidence < 0.9: {share[:-3].sum()*100:.1f} % of all pixels, containing {cum_err[-4]*100:.0f} % of all errors")
print(f"pixels with confidence >= 0.99: {share[-1]*100:.1f} % of all pixels, error rate {err_rate[-1]*100:.2f} %")

th = pd.DataFrame(type_hist, index=labels).T; th_share = th.div(th.sum(axis=1), axis=0)
fig, ax = plt.subplots(figsize=(8.5, 3.4))
y = np.arange(len(th))[::-1]
hi = th_share.iloc[:, -2] * 100; vhi = th_share.iloc[:, -1] * 100
ax.barh(y, hi, color=C["soil"], height=0.62, label="confidence 0.95–0.99")
ax.barh(y, vhi, left=hi + 0.3, color=C["ink"], height=0.62, label="confidence ≥ 0.99")
for yi, (name, row) in zip(y, th.iterrows()):
    ax.text(hi[name] + vhi[name] + 1.2, yi, f"{(hi[name]+vhi[name]):.0f} % of {int(row.sum())/1e3:,.0f} k wrong px", va="center", fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(th.index); ax.set_xlabel("share of this error type's pixels that are highly confident (%)"); ax.set_xlim(0, 48)
ax.legend(loc="upper right"); ax.set_title("Figure 10 · The crop ↔ weed swaps are the most confident errors")
savefig("fig10_confidence_by_error_type")
''')

md(r'''
**Finding Q4 — expectation confirmed, including the exception.** The model is well calibrated overall (expected calibration error 0.013): 86 % of pixels have confidence above 0.99 with an error rate of 0.2 %, and the 6 % of pixels below 0.9 contain 68 % of all wrong pixels. A sprayer that treats uncertain pixels as "do not decide" would remove most errors at little cost. But the errors that remain confident are the expensive ones: **a crop pixel predicted as weed is above 0.95 confidence in about 30 % of cases**, the highest share of any error type, and about three times the share of the crop → soil family. Confidence flags boundary errors, not the swapped small crops.

### 3.6 Q5 — Which input properties drive the decisions? (controlled perturbations)

`tools/run_perturbations.py` re-runs the model on the 60 test images after one controlled change of the input each time, and scores against the unchanged annotation. Families: a horizontal flip as a control (the training augmentation contained flips), Gaussian blur, brightness, contrast, additive noise, removal of colour, and a down-/up-scaling that removes fine detail without blurring edges as strongly.
''')

code(r'''
ps = pd.read_csv(PRED / "perturbations" / "summary.csv")
order = ["identity", "hflip", "blur_s1", "blur_s2", "blur_s4", "downscale_0.5", "downscale_0.25", "grayscale",
         "bright_0.6", "bright_0.8", "bright_1.2", "bright_1.4", "contrast_0.6", "contrast_1.4", "noise_10", "noise_20", "noise_40"]
nice = {"identity": "unchanged", "hflip": "h-flip", "blur_s1": "blur σ 1", "blur_s2": "blur σ 2", "blur_s4": "blur σ 4",
        "downscale_0.5": "½ resolution", "downscale_0.25": "¼ resolution", "grayscale": "no colour",
        "bright_0.6": "bright ×0.6", "bright_0.8": "bright ×0.8", "bright_1.2": "bright ×1.2", "bright_1.4": "bright ×1.4",
        "contrast_0.6": "contrast ×0.6", "contrast_1.4": "contrast ×1.4", "noise_10": "noise σ 10", "noise_20": "noise σ 20", "noise_40": "noise σ 40"}
ps = ps.set_index("condition")
ps = ps.loc[[c for c in order if c in ps.index]]
base = ps.loc["identity"]
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
x = np.arange(len(ps))
ax = axes[0]
for cls, mk in [("soil", "s"), ("crop", "o"), ("weed", "^")]:
    ax.plot(x, ps[f"IoU_{cls}"], color=C[cls], marker=mk, ms=7, lw=1.2, label=f"IoU {cls}")
    ax.axhline(base[f"IoU_{cls}"], color=C[cls], lw=0.8, ls=":")
ax.set_ylim(0, 1); ax.set_ylabel("IoU on the test split"); ax.legend(loc="lower left", ncol=3); ax.set_title("per-class IoU under each perturbation (dotted: unchanged input)")
ax = axes[1]
w = 0.38
ax.bar(x - w / 2, ps["crop_as_weed"] / 1e6, width=w, color=C["crop"], label="crop pixels predicted as weed")
ax.bar(x + w / 2, ps["weed_as_crop"] / 1e6, width=w, color=C["weed"], label="weed pixels predicted as crop")
ax.set_yscale("log"); ax.set_ylim(0.03, 8); ax.set_ylabel("swapped pixels (millions, log)"); ax.legend(loc="upper left"); ax.set_title("crop ↔ weed swaps")
ax.set_xticks(x); ax.set_xticklabels([nice[c] for c in ps.index], rotation=35, ha="right")
for xi in [1.5, 4.5, 6.5, 7.5, 11.5, 13.5]:
    for a in axes: a.axvline(xi, color=C["grid"], lw=1)
fig.suptitle("Figure 11 · Brightness, contrast and mild noise barely matter; removing fine detail turns crop into weed; removing colour removes all vegetation", x=0.01, ha="left", fontsize=11)
plt.tight_layout(); savefig("fig11_perturbations")
show = ps[["family", "mIoU", "IoU_soil", "IoU_crop", "IoU_weed", "prec_crop", "rec_crop", "crop_as_weed", "weed_as_crop", "pred_pixels_crop", "mean_confidence"]].copy()
show.index = [nice[c] for c in show.index]; display(show.round(3))
''')

code(r'''
sid = "rose2_haricot_0380"
conds = ["identity", "blur_s2", "blur_s4", "downscale_0.25", "grayscale"]
fig, axes = plt.subplots(2, len(conds), figsize=(4.2 * len(conds), 6.8))
for j, c in enumerate(conds):
    inp = cv2.cvtColor(cv2.imread(str(PRED / "perturbations" / c / "showcase_inputs" / f"{sid}.jpg")), cv2.COLOR_BGR2RGB)
    pm = bc.load_mask(PRED / "perturbations" / c / "showcase_masks" / f"{sid}.png")
    row = pd.read_csv(PRED / "perturbations" / c / "per_image.csv").set_index("id").loc[sid]
    axes[0, j].imshow(inp); axes[0, j].set_title(f"{nice[c]}"); axes[1, j].imshow(bc.overlay(inp, pm, 0.8))
    axes[1, j].set_title(f"crop IoU {row.IoU_crop:.2f} · weed IoU {row.IoU_weed:.2f}", fontsize=9)
    for a in axes[:, j]: a.axis("off")
overlay_legend(fig, y=1.01)
fig.suptitle(f"Figure 12 · One image under the perturbations that matter ({sid}); top: model input, bottom: prediction", x=0.01, y=1.0, ha="left", fontsize=11)
plt.tight_layout(h_pad=2.5); savefig("fig12_perturbation_showcase")
''')

md(r'''
**Finding Q5 — two expectations confirmed, two surprises.** *Brightness and contrast* changes of ±40 % and additive noise of σ 10 move the test mIoU by less than 0.01 (the largest single change is crop IoU −0.016 under contrast ×1.4) — confirmed, as the per-image stretch and the augmentation predicted. *Blur* was expected to hurt small weeds; it does, but the dominant effect is different and large: at σ 4 **crop IoU collapses from 0.86 to 0.21 while weed IoU stays at 0.51, because most crop pixels are now predicted as weed** (4.4 M crop → weed pixels against 0.14 M unperturbed; Figure 11 bottom, Figure 12). Strong noise (σ 40) and quarter resolution do the same thing more mildly. So the crop-versus-weed decision is carried by **fine leaf texture and edge detail**, and when that evidence is removed the model's default is *weed* — the same default it applies to small plants (Figure 6). *Removing colour* was expected to blur the crop/weed line; instead **the grey image contains no vegetation for the model at all** (crop and weed IoU 0.00, everything predicted as soil): colour is the plant-versus-soil cue, texture the crop-versus-weed cue, and the two are separable by these simple probes. Finally the horizontal *flip*, a training augmentation, still changes the test score by about 0.006 mIoU: the model is not exactly flip-invariant, and which small plants get swapped changes with the flip. None of these behaviours is visible in a score table, and they are the natural entry points for Phase II (which representations carry the texture cue, and where it is lost under blur).

### 3.7 The failure cases, looked at directly
''')

code(r'''
cases = [("rose2_haricot_0324", "worst test image: ~1 000 annotated weed pixels in tiny seedlings, weed IoU 0.27"),
         ("rose3_haricot_0100", "crop-free image: zero crop predicted; the few green seedlings the annotation skips show as yellow"),
         ("rose2_haricot_0380", "typical image: cyan patches are small crop plants in the upper row called weed")]
fig, axes = plt.subplots(len(cases), 1, figsize=(12, 8.4 * len(cases)))
for ax, (sid, what) in zip(axes, cases):
    ax.imshow(cv2.cvtColor(cv2.imread(str(PRED / "test" / "panels" / f"{sid}.jpg")), cv2.COLOR_BGR2RGB)); ax.axis("off"); ax.set_title(f"{sid} — {what}")
fig.suptitle("Figure 13 · RGB | ground truth (crop green, weed red) | prediction | error map (orange missed crop, magenta missed weed, yellow false plant, cyan crop/weed swap)",
             x=0.01, y=0.995, ha="left", fontsize=10)
plt.tight_layout(rect=(0, 0, 1, 0.985)); savefig("fig13_failure_panels")
''')

md(r'''
## 4. Findings against expectations

| | Expectation | Result | Evidence |
|---|---|---|---|
| Q1 | The aggregate score hides weak images | **confirmed** — median image mIoU 0.82, minimum 0.63; same spread in all splits | Figure 4 |
| Q2a | Sparse-weed images fail | **confirmed** — IoU tracks the class's pixel count, r ≈ 0.9 for weed | Figure 5 |
| Q2b | Crop-free images show false crop (large crop loss weight) | **contradicted** — zero crop pixels predicted on all five | group table, Figure 13 |
| Q2c | Small objects are missed | **confirmed and sharpened** — small *weeds* are missed, small *crops* are called weed; objects > 1 000 px are found > 97 % | Figure 6 |
| Q2d | Part of the false-plant error is unannotated seedlings | **contradicted in magnitude** — 96 % of false-plant pixels are boundary dilation; isolated blobs are 4 % and as faint as missed seedlings | Figure 7 |
| Q3 | Crop ↔ weed swaps are whole plants, not boundaries | **confirmed** — 93 % of all errors are boundary pixels, swaps have the largest interior share | Figure 8 |
| Q4 | Confidence flags errors, except the swaps | **confirmed** — ECE 0.013, 68 % of errors below 0.9 confidence; crop → weed is the most confident error | Figures 9, 10 |
| Q5 | Brightness/contrast irrelevant; blur hurts small weeds; colour separates crop from weed | **partly** — brightness/contrast/mild noise irrelevant; blur turns *crop* into weed (texture is the crop/weed cue); without colour nothing is vegetation (colour is the plant/soil cue) | Figures 11, 12 |
| Q6 | The variety effect is a weed-density effect | **confirmed** (with 60 images) — after removing the density trend the gap reverses; rose2 is not harder, it is sparser | section 3.3 |

**The picture of the model that emerges.** It finds vegetation by colour, reliably and over-generously (a thin halo around every plant; no colour, no vegetation). It separates crop from weed by fine leaf texture: when that texture is absent — small plants, blurred, noisy or low-resolution input — it defaults to *weed*, confidently. Its misses are the faintest, smallest seedlings. None of this is visible in the score table, and all of it matters for the sprayer: small crops get sprayed, faint seedlings get missed.

## 5. Reproducibility

Everything in this notebook derives from files in the repository; the heavy inputs (`data/`, per-pixel masks and probabilities) are not tracked in git and can be regenerated:

```bash
pip install -r scripts/requirements.txt
python scripts/predict.py --split test val --save-probs --save-panels     # torch backend, GPU or CPU
python scripts/predict.py --split train
python scripts/predict.py --split test --checkpoint model/last.pt --out predictions/last_epoch119
python scripts/evaluate_predictions.py --pred predictions/test --group variety
python scripts/evaluate_predictions.py --pred predictions/test --group has_crop
python tools/run_perturbations.py                                          # about 10 minutes on a GPU
python tools/build_notebook.py notebooks/AI_Gangsta_mid_submission.ipynb   # regenerate this notebook from its source
```

The preprocessing in `scripts/beans_common.py` was checked against the training transform (max difference 2.4e-7 on the normalised tensor, identical masks); the test score it produces matches the archived `run_metadata/eval_final.json` within 2e-5; the ONNX export agrees with PyTorch on 99.9999 % of pixels.
''')

code(r'''
print("predictions produced with:", {k: mt[k] for k in ["checkpoint", "checkpoint_sha256", "epoch", "backend", "device", "torch"] if k in mt})
print("notebook executed with: python", platform.python_version(), "| numpy", np.__version__, "| pandas", pd.__version__, "| opencv", cv2.__version__, "| matplotlib", mpl.__version__)
print("figures written to", FIG)
''')

md(r'''
## 6. What is not done yet, and the plan

**Not done in Phase I (stated rather than faked).** A matched variety comparison with more images than the 60 test images allow (the residual in section 3.3 is suggestive, not conclusive). The object-level analysis uses connected components of the rasterised masks; the original polygon annotations would give cleaner object counts for touching leaves. The perturbation probe uses global changes; local ones (occluding one plant, pasting a small crop into a weed patch) would test the size prior directly.

**Phase II (opening the box), continuing from the findings above.**
- *Where is the texture cue?* Feature maps and Grad-CAM-style attributions (hook points in `scripts/README.md`) on small crop plants that are called weed, and on the same plants under blur σ 4, to see at which stage the crop evidence disappears.
- *Is "small = weed" a learned prior?* Compare the representations of small crops, small weeds and large crops in the decoder features; nearest neighbours of swapped crops in that space.
- *When was it learned?* The best (epoch 89) and last (epoch 119) checkpoints are in `model/`; crop precision was the slow learner, so the swaps should decrease along training.
''')

nb["cells"] = cells
out = sys.argv[1]
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
