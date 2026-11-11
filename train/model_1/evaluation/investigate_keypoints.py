"""
investigate_keypoints.py  —  Task 2: keypoint/obs investigation before retraining

The dataset is stored as per-source COCO files under
  data/annotations/v2/
Splits are derived virtually from the source_video name (and frame range
for the chronological 20250509 video).

This script reconstructs the ORIGINAL train/val/test assignment used during
the v9_20260513_013233 training run, then runs all five investigative checks.

Outputs:
  investigation_plots/  (PNG figures)
  investigation_report.md

Run from train/train_v9_v1/ :
    python investigate_keypoints.py
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[2]  # evaluation -> model_1 -> train -> repo root
sys.path.insert(0, str(_HERE.parent / "structure"))

from config import FEAT_H, FEAT_W

DATA_ROOT = REPO_ROOT / "data" / "annotations" / "v2"
PLOT_DIR = REPO_ROOT / "investigation_plots"
REPORT_PATH = REPO_ROOT / "investigation_report.md"

SOURCE_FILES = {
    "20250508": "20250508_annotations.coco.json",
    "20250509": "20250509_annotations.coco.json",
    "20250510": "20250510_annotations.coco.json",
    "CAM1 - V1": "CAM1 - V1_annotations.coco.json",
    "CAM1 - V2": "CAM1 - V2_annotations.coco.json",
    "CAM1 - V3": "CAM1 - V3_annotations.coco.json",
    "CAM2 - V1": "CAM2 - V1_annotations.coco.json",
    "CAM2 - V2": "CAM2 - V2_annotations.coco.json",
    "CAM2 - V3": "CAM2 - V3_annotations.coco.json",
}
RANGES_20250509 = [
    ("20250509_frames_0-213", range(0, 214)),
    ("20250509_frames_230-261", range(230, 262)),
    ("20250509_frames_275-304", range(275, 305)),
]

# Original split (matches v9_20260513_013233 training)
ASSIGNMENT = {
    "train": {
        "20250509_frames_0-213",
        "20250508",
        "CAM1 - V1",
        "CAM1 - V2",
        "CAM1 - V3",
    },
    "val": {
        "20250509_frames_230-261",
        "CAM2 - V1",
    },
    "test": {
        "20250509_frames_275-304",
        "20250510",
        "CAM2 - V2",
        "CAM2 - V3",
    },
}
SRC_TO_SPLIT = {src: spl for spl, srcs in ASSIGNMENT.items() for src in srcs}

KPT_LABELS = ["tip_x", "tip_y", "left_x", "left_y", "right_x", "right_y"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────


def tag_20250509(file_name: str):
    stem = Path(file_name).stem
    try:
        frame = int(stem.split("_")[-1])
    except (ValueError, IndexError):
        return None
    for tag, rng in RANGES_20250509:
        if frame in rng:
            return tag
    return None  # excluded gap


def load_spray_instances():
    """Return list of dicts for every nozzle_spray annotation across all sources."""
    instances = []
    for source_key, fname in SOURCE_FILES.items():
        path = DATA_ROOT / fname
        if not path.exists():
            print(f"  WARN: missing {path.name}")
            continue
        with open(path) as f:
            d = json.load(f)
        imgs = {i["id"]: i for i in d["images"]}
        for ann in d["annotations"]:
            if ann.get("_meta", {}).get("original_class") != "nozzle_spray":
                continue
            img = imgs.get(ann["image_id"], {})

            if source_key == "20250509":
                src = tag_20250509(img.get("file_name", ""))
                if src is None:
                    continue  # gap-range frame, not in any split
            else:
                src = source_key
            split = SRC_TO_SPLIT.get(src)
            if split is None:
                continue

            W = float(img.get("width", 1))
            H = float(img.get("height", 1))

            b = [float(v) for v in ann.get("bbox", [0, 0, 1, 1])]
            cx = (b[0] + b[2] / 2) / W
            cy = (b[1] + b[3] / 2) / H
            bw, bh = b[2] / W, b[3] / H

            raw = ann.get("keypoints", [])
            kpts = []
            for k in range(3):
                kx = float(raw[k * 3]) / W if len(raw) >= k * 3 + 2 else 0.0
                ky = float(raw[k * 3 + 1]) / H if len(raw) >= k * 3 + 2 else 0.0
                kpts.extend([kx, ky])
            if sum(kpts) == 0.0:
                continue

            instances.append(
                {
                    "split": split,
                    "source_video": src,
                    "img_id": ann["image_id"],
                    "ann_id": ann["id"],
                    "bbox_norm": [cx, cy, bw, bh],
                    "kpts_norm": kpts,
                }
            )
    return instances


def spray_angle(kpts):
    tip = np.array(kpts[0:2])
    left = np.array(kpts[2:4])
    right = np.array(kpts[4:6])
    v1, v2 = left - tip, right - tip
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


# ─────────────────────────────────────────────────────────────────────────────
# 2a: keypoint coordinate distribution
# ─────────────────────────────────────────────────────────────────────────────


def check_2a(instances, plot_dir):
    results = {}
    for subset in ("train", "val", "test", "all"):
        sub = (
            instances
            if subset == "all"
            else [x for x in instances if x["split"] == subset]
        )
        if not sub:
            continue
        kpts = np.array([x["kpts_norm"] for x in sub])
        stats = {}
        for i, lbl in enumerate(KPT_LABELS):
            col = kpts[:, i]
            stats[lbl] = {
                "min": float(col.min()),
                "max": float(col.max()),
                "mean": float(col.mean()),
                "std": float(col.std()),
            }
        results[subset] = {"n": len(sub), "stats": stats}

    # scatter plot
    colours = {"train": "#2196F3", "val": "#FF9800", "test": "#F44336"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax_i, (kx, ky, title) in enumerate(
        [(0, 1, "Tip (x,y)"), (2, 3, "Left-edge (x,y)"), (4, 5, "Right-edge (x,y)")]
    ):
        ax = axes[ax_i]
        for spl, col in colours.items():
            sub = [x for x in instances if x["split"] == spl]
            if not sub:
                continue
            arr = np.array([x["kpts_norm"] for x in sub])
            ax.scatter(
                arr[:, kx],
                arr[:, ky],
                c=col,
                s=12,
                alpha=0.55,
                label=f"{spl} (n={len(sub)})",
            )
        ax.set_title(title)
        ax.set_xlabel("x (norm)")
        ax.set_ylabel("y (norm)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
    fig.suptitle("Keypoint positions — all spray instances by split", fontsize=11)
    plt.tight_layout()
    scatter_path = plot_dir / "2a_keypoint_scatter.png"
    plt.savefig(scatter_path, dpi=120)
    plt.close()

    # histograms train vs test
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for idx, lbl in enumerate(KPT_LABELS):
        ax = axes[idx // 3, idx % 3]
        for spl, col in [("train", "#2196F3"), ("test", "#F44336")]:
            sub = [x for x in instances if x["split"] == spl]
            if not sub:
                continue
            vals = [x["kpts_norm"][idx] for x in sub]
            ax.hist(
                vals,
                bins=25,
                alpha=0.6,
                color=col,
                label=f"{spl} (n={len(sub)})",
                density=True,
            )
        ax.set_title(lbl)
        ax.set_xlabel("norm coord")
        ax.legend(fontsize=8)
    fig.suptitle("Keypoint coordinate histograms — train vs test", fontsize=11)
    plt.tight_layout()
    hist_path = plot_dir / "2a_keypoint_histograms.png"
    plt.savefig(hist_path, dpi=120)
    plt.close()

    return results, scatter_path, hist_path


# ─────────────────────────────────────────────────────────────────────────────
# 2b: FCOS cell assignment heatmap
# ─────────────────────────────────────────────────────────────────────────────


def check_2b(instances, plot_dir):
    heatmaps = {}
    for subset in ("train", "test"):
        sub = [x for x in instances if x["split"] == subset]
        grid = np.zeros((FEAT_H, FEAT_W), dtype=int)
        for inst in sub:
            cx, cy, bw, bh = inst["bbox_norm"]
            x1, y1 = cx - bw / 2, cy - bh / 2
            x2, y2 = cx + bw / 2, cy + bh / 2
            j_min = max(0, math.floor(x1 * FEAT_W))
            j_max = min(FEAT_W - 1, math.ceil(x2 * FEAT_W - 1))
            i_min = max(0, math.floor(y1 * FEAT_H))
            i_max = min(FEAT_H - 1, math.ceil(y2 * FEAT_H - 1))
            for ii in range(i_min, i_max + 1):
                for jj in range(j_min, j_max + 1):
                    ccx = (jj + 0.5) / FEAT_W
                    ccy = (ii + 0.5) / FEAT_H
                    if x1 <= ccx <= x2 and y1 <= ccy <= y2:
                        grid[ii, jj] += 1
        heatmaps[subset] = grid

    train_cells = int((heatmaps["train"] > 0).sum())
    test_cells = int((heatmaps["test"] > 0).sum())
    overlap_cells = int(((heatmaps["train"] > 0) & (heatmaps["test"] > 0)).sum())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im = axes[0].imshow(heatmaps["train"], origin="upper", cmap="Blues", aspect="equal")
    axes[0].set_title(f"Train spray cell activations ({train_cells} cells)")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    im = axes[1].imshow(heatmaps["test"], origin="upper", cmap="Reds", aspect="equal")
    axes[1].set_title(f"Test spray cell activations ({test_cells} cells)")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    rgb = np.zeros((FEAT_H, FEAT_W, 3))
    t_mask = heatmaps["train"] > 0
    e_mask = heatmaps["test"] > 0
    rgb[t_mask & ~e_mask] = [0.13, 0.59, 0.95]
    rgb[~t_mask & e_mask] = [0.96, 0.26, 0.21]
    rgb[t_mask & e_mask] = [0.61, 0.15, 0.69]
    axes[2].imshow(rgb, origin="upper", aspect="equal")
    axes[2].set_title(
        f"Overlap: blue=train, red=test, purple=both ({overlap_cells} both)"
    )

    for ax in axes:
        ax.set_xlabel("grid col (0-39)")
        ax.set_ylabel("grid row (0-39)")
    fig.suptitle("FCOS cell activations — spray instances", fontsize=10)
    plt.tight_layout()
    p = plot_dir / "2b_cell_heatmaps.png"
    plt.savefig(p, dpi=120)
    plt.close()

    return {
        "train_active_cells": train_cells,
        "test_active_cells": test_cells,
        "overlap_cells": overlap_cells,
        "overlap_pct_of_test": (
            (overlap_cells / test_cells * 100) if test_cells else 0.0
        ),
    }, p


# ─────────────────────────────────────────────────────────────────────────────
# 2d: nearest-neighbour similarity
# ─────────────────────────────────────────────────────────────────────────────


def check_2d(instances):
    train_inst = [x for x in instances if x["split"] == "train"]
    test_inst = [x for x in instances if x["split"] == "test"]
    if not train_inst or not test_inst:
        return {"error": "missing split"}
    train_kpts = np.array([x["kpts_norm"] for x in train_inst])
    test_kpts = np.array([x["kpts_norm"] for x in test_inst])

    nn_dists = []
    details = []
    for i, tq in enumerate(test_kpts):
        dists = np.linalg.norm(train_kpts - tq, axis=1)
        nn_idx = int(np.argmin(dists))
        nn_d = float(dists[nn_idx])
        nn_dists.append(nn_d)
        details.append(
            {
                "test_ann": test_inst[i]["ann_id"],
                "test_src": test_inst[i]["source_video"],
                "nn_dist": round(nn_d, 4),
                "nn_src": train_inst[nn_idx]["source_video"],
            }
        )
    arr = np.array(nn_dists)
    return {
        "n_train": len(train_kpts),
        "n_test": len(test_kpts),
        "nn_dist_min": float(arr.min()),
        "nn_dist_mean": float(arr.mean()),
        "nn_dist_max": float(arr.max()),
        "nn_dist_std": float(arr.std()),
        "details": details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2e: GT angle distribution
# ─────────────────────────────────────────────────────────────────────────────


def check_2e(instances, plot_dir):
    by_split = defaultdict(list)
    for inst in instances:
        a = spray_angle(inst["kpts_norm"])
        if a is not None:
            by_split[inst["split"]].append(a)
    by_split["all"] = [
        a for spl in ("train", "val", "test") for a in by_split.get(spl, [])
    ]

    stats = {}
    for spl, angles in by_split.items():
        if not angles:
            continue
        a = np.array(angles)
        stats[spl] = {
            "n": len(a),
            "min": float(a.min()),
            "max": float(a.max()),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)),
        }

    fig, ax = plt.subplots(figsize=(9, 5))
    colours = {"train": "#2196F3", "val": "#FF9800", "test": "#F44336"}
    all_a = [a for sp in ("train", "val", "test") for a in by_split.get(sp, [])]
    bins = np.linspace(min(all_a) - 1, max(all_a) + 1, 35) if all_a else 30
    for spl in ("train", "val", "test"):
        if not by_split.get(spl):
            continue
        ax.hist(
            by_split[spl],
            bins=bins,
            alpha=0.6,
            color=colours[spl],
            label=f"{spl} (n={len(by_split[spl])})",
            density=True,
        )
    ax.set_xlabel("GT spray cone angle (degrees)")
    ax.set_ylabel("density")
    ax.set_title("Ground-truth spray angle distribution")
    ax.legend()
    plt.tight_layout()
    p = plot_dir / "2e_angle_distribution.png"
    plt.savefig(p, dpi=120)
    plt.close()
    return stats, p


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────


def verdict_for(
    name,
    val,
    *,
    concern_below=None,
    concern_above=None,
    review_below=None,
    review_above=None,
):
    """Generic verdict helper."""
    if concern_below is not None and val < concern_below:
        return "CONCERN"
    if concern_above is not None and val > concern_above:
        return "CONCERN"
    if review_below is not None and val < review_below:
        return "NEEDS REVIEW"
    if review_above is not None and val > review_above:
        return "NEEDS REVIEW"
    return "OK"


def write_report(r2a, r2b, r2d, r2e, plot_dir):
    L = []
    A = L.append

    A("# AFGNet Keypoint Investigation Report")
    A("")
    A("Pre-retraining diagnostic — is `MAE_primary = 1.74°` genuine learning")
    A("or position memorisation?")
    A("")
    A("---")
    A("")

    # ── 2a ───────────────────────────────────────────────────────────────────
    A("## CHECK 2a — Keypoint Position Distribution")
    A("")
    A(
        f"Plots: `{plot_dir.name}/2a_keypoint_scatter.png`,"
        f" `{plot_dir.name}/2a_keypoint_histograms.png`"
    )
    A("")
    for subset in ("train", "val", "test", "all"):
        if subset not in r2a:
            continue
        info = r2a[subset]
        A(f'### {subset.upper()}  (n={info["n"]})')
        A("")
        A("| coord | min | max | mean | std |")
        A("|---|---|---|---|---|")
        for lbl, s in info["stats"].items():
            A(
                "| %-10s | %.4f | %.4f | %.4f | %.4f |"
                % (lbl, s["min"], s["max"], s["mean"], s["std"])
            )
        A("")

    if "train" in r2a and "test" in r2a:
        t = r2a["train"]["stats"]
        e = r2a["test"]["stats"]
        tip_gap = math.hypot(
            t["tip_x"]["mean"] - e["tip_x"]["mean"],
            t["tip_y"]["mean"] - e["tip_y"]["mean"],
        )
        avg_std = np.mean([t[k]["std"] for k in KPT_LABELS])
        verdict = verdict_for("2a", avg_std, concern_below=0.05, review_below=0.10)
        A(
            f"**Verdict: {verdict}** — "
            f'train tip mean ({t["tip_x"]["mean"]:.3f}, {t["tip_y"]["mean"]:.3f}) vs '
            f'test tip mean ({e["tip_x"]["mean"]:.3f}, {e["tip_y"]["mean"]:.3f}); '
            f"centroid gap = {tip_gap:.4f}, mean per-coord std (train) = {avg_std:.4f}."
        )
        if verdict == "CONCERN":
            A("")
            A("Keypoints occupy a very narrow region of the image — consistent with a")
            A("fixed-position nozzle filmed across a single video. The model can hit")
            A("a low MAE by emitting near-constant coordinates regardless of input.")
    A("")
    A("---")
    A("")

    # ── 2b ───────────────────────────────────────────────────────────────────
    A("## CHECK 2b — FCOS Cell Assignment Overlap")
    A("")
    A(f"Plot: `{plot_dir.name}/2b_cell_heatmaps.png`")
    A("")
    A(f'- Train active cells: **{r2b["train_active_cells"]}**')
    A(f'- Test  active cells: **{r2b["test_active_cells"]}**')
    A(
        f'- Cells active in both: **{r2b["overlap_cells"]}** '
        f'({r2b["overlap_pct_of_test"]:.1f}% of test cells)'
    )
    A("")
    verdict = verdict_for(
        "2b", r2b["overlap_pct_of_test"], concern_above=70, review_above=40
    )
    A(
        f"**Verdict: {verdict}** — "
        f'{r2b["overlap_pct_of_test"]:.0f}% of test spray cells were also seen as '
        f"positive cells during training."
    )
    if verdict == "CONCERN":
        A("")
        A("The supervision falls on essentially the same grid locations in train and")
        A(
            "test. Spatial memorisation is plausible: the model can predict by cell index"
        )
        A("rather than by image content.")
    A("")
    A("---")
    A("")

    # ── 2c (code review) ─────────────────────────────────────────────────────
    A("## CHECK 2c — Keypoint Head Encoding & Loss")
    A("")
    A("### Decode (`model.py` AFGNetHead.forward)")
    A("```")
    A("KPT_REGRESSION_RANGE = 20  cells")
    A("cell_w = 1 / FEAT_W = 1 / 40 = 0.025")
    A("max_offset = KPT_REGRESSION_RANGE * cell_w = 0.5  (half image width)")
    A("")
    A("kpt_x = cell_x + (sigmoid(raw) * 2 - 1) * cell_w * KPT_REGRESSION_RANGE")
    A("      ∈  [cell_x - 0.5,  cell_x + 0.5]")
    A("```")
    A(
        "From any cell the model can place a keypoint **anywhere in [0,1]² minus a strip**."
    )
    A("The cell-aware anchor is effectively unconstrained.")
    A("")
    A("### Loss (`loss.py:128-134`)")
    A("```python")
    A("kpt_loss = F.smooth_l1_loss(")
    A(
        '    flat(preds["keypoints"], NUM_KEYPOINTS * 2)[spray_mask],   # decoded abs coords'
    )
    A(
        '    targets["keypoints"]...[pos_mask][spray_mask],              # GT abs coords [0,1]'
    )
    A(")")
    A("```")
    A("Both sides are absolute image coords in `[0,1]`. The loss is **mathematically**")
    A(
        "correct, but the regression target is not normalised by cell width or bbox size."
    )
    A("A 0.01 coordinate error contributes the same scalar regardless of where the")
    A(
        'keypoint actually lies in the image — so a model that learns "always output ≈ the'
    )
    A('mean spray-keypoint position" achieves near-trivial loss.')
    A("")
    A("**Verdict: CONCERN** — `KPT_REGRESSION_RANGE = 20` provides no meaningful local")
    A("anchor, and the L1 loss is over raw normalised coordinates. Combined with the")
    A(
        "narrow keypoint distribution in 2a, a constant predictor would already score well."
    )
    A("")
    A("---")
    A("")

    # ── 2d ───────────────────────────────────────────────────────────────────
    A("## CHECK 2d — Train↔Test Spray Similarity")
    A("")
    if "error" in r2d:
        A(f'Error: {r2d["error"]}')
    else:
        A(f'- n train sprays: **{r2d["n_train"]}**')
        A(f'- n test  sprays: **{r2d["n_test"]}**')
        A("")
        A("Per-test-instance NN distance in 6-D normalised keypoint space:")
        A("")
        A("| stat | value |")
        A("|---|---|")
        A(f'| min  | {r2d["nn_dist_min"]:.4f} |')
        A(f'| mean | {r2d["nn_dist_mean"]:.4f} |')
        A(f'| max  | {r2d["nn_dist_max"]:.4f} |')
        A(f'| std  | {r2d["nn_dist_std"]:.4f} |')
        A("")
        A("Closest 5 test→train pairs:")
        A("")
        A("| test_ann | test_src | nn_dist | train_src |")
        A("|---|---|---|---|")
        for row in sorted(r2d["details"], key=lambda x: x["nn_dist"])[:5]:
            A(
                f'| {row["test_ann"]} | {row["test_src"]} | {row["nn_dist"]:.4f} | {row["nn_src"]} |'
            )
        A("")
        verdict = verdict_for(
            "2d", r2d["nn_dist_mean"], concern_below=0.05, review_below=0.15
        )
        A(
            f"**Verdict: {verdict}** — mean NN distance "
            f'{r2d["nn_dist_mean"]:.4f} in 6-D normalised keypoint space.'
        )
        if verdict == "CONCERN":
            A("")
            A(
                "Every test spray has a near-identical neighbour in train. Memorisation is"
            )
            A("the most parsimonious explanation for the low MAE.")
    A("")
    A("---")
    A("")

    # ── 2e ───────────────────────────────────────────────────────────────────
    A("## CHECK 2e — Inter-Instance Angle Variance")
    A("")
    A(f"Plot: `{plot_dir.name}/2e_angle_distribution.png`")
    A("")
    A("| split | n | min° | max° | mean° | std° | p10° | p90° |")
    A("|---|---|---|---|---|---|---|---|")
    for spl in ("train", "val", "test", "all"):
        if spl not in r2e:
            continue
        s = r2e[spl]
        A(
            "| %-5s | %d | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |"
            % (spl, s["n"], s["min"], s["max"], s["mean"], s["std"], s["p10"], s["p90"])
        )
    A("")
    if "all" in r2e:
        std = r2e["all"]["std"]
        rng = r2e["all"]["max"] - r2e["all"]["min"]
        verdict = verdict_for("2e", std, concern_below=5.0, review_below=10.0)
        A(
            f"**Verdict: {verdict}** — overall angle std = {std:.1f}°, range = {rng:.1f}°."
        )
        if verdict == "CONCERN":
            A("")
            A(f"Reported MAE_primary = 1.74° is **below the natural σ of the data**")
            A(f"({std:.1f}°). A constant predictor that emits the mean angle would")
            A(f"achieve roughly std × 0.8 ≈ {std*0.8:.1f}° MAE — i.e. the metric does")
            A(f'not separate "learned the angle" from "guessed the mean".')
    A("")
    A("---")
    A("")

    A("## Summary table")
    A("")
    A("| check | topic | verdict |")
    A("|---|---|---|")
    A("| 2a | keypoint clustering          | see above |")
    A("| 2b | cell overlap train↔test      | see above |")
    A("| 2c | encoding/loss bias           | **CONCERN** |")
    A("| 2d | train↔test NN similarity     | see above |")
    A("| 2e | GT angle variance vs MAE     | see above |")
    A("")
    A("### Recommendation")
    A("")
    A("Before reporting MAE_primary as evidence for RQ2 in Chapter 6, confirm that:")
    A("")
    A("1. At least one of 2a/2b/2d shows that test positions are clearly distinct from")
    A("   train (otherwise the metric reflects memorisation).")
    A("2. The GT angle std exceeds the reported MAE by a meaningful factor (otherwise")
    A('   the metric is below the trivial "predict the mean" baseline).')
    A("")
    A("If all four geometric checks show high overlap and the angle std is small, the")
    A("correct action is to **resplit the data** so each camera/source appears in all")
    A("three splits, then retrain on the reverse split and re-evaluate.")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"Report written: {REPORT_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    if not DATA_ROOT.exists():
        sys.exit(f"ERROR: data root not found: {DATA_ROOT}")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading spray instances from per-source COCO files...")
    instances = load_spray_instances()
    by_split = {
        s: [x for x in instances if x["split"] == s] for s in ("train", "val", "test")
    }
    print(
        f'  train: {len(by_split["train"])}  '
        f'val: {len(by_split["val"])}  '
        f'test: {len(by_split["test"])}  '
        f"total: {len(instances)}"
    )
    if len(instances) == 0:
        sys.exit("No spray instances found — check DATA_ROOT and source files.")

    print("\nCheck 2a: keypoint distribution...")
    r2a, _, _ = check_2a(instances, PLOT_DIR)
    for s in ("train", "test"):
        if s in r2a:
            st = r2a[s]["stats"]
            print(
                f'  {s}: tip mean=({st["tip_x"]["mean"]:.3f}, {st["tip_y"]["mean"]:.3f})  '
                f'std=({st["tip_x"]["std"]:.3f}, {st["tip_y"]["std"]:.3f})'
            )

    print("\nCheck 2b: FCOS cell heatmap...")
    r2b, _ = check_2b(instances, PLOT_DIR)
    print(
        f'  train_cells={r2b["train_active_cells"]}  '
        f'test_cells={r2b["test_active_cells"]}  '
        f'overlap={r2b["overlap_cells"]}  '
        f'({r2b["overlap_pct_of_test"]:.1f}% of test)'
    )

    print("\nCheck 2d: NN similarity...")
    r2d = check_2d(instances)
    if "error" not in r2d:
        print(
            f'  NN dist: min={r2d["nn_dist_min"]:.4f}  '
            f'mean={r2d["nn_dist_mean"]:.4f}  max={r2d["nn_dist_max"]:.4f}'
        )

    print("\nCheck 2e: angle variance...")
    r2e, _ = check_2e(instances, PLOT_DIR)
    for s in ("train", "test", "all"):
        if s in r2e:
            st = r2e[s]
            print(
                f'  {s}: n={st["n"]}  mean={st["mean"]:.1f}°  '
                f'std={st["std"]:.1f}°  range=[{st["min"]:.1f}, {st["max"]:.1f}]'
            )

    print("\nWriting report...")
    write_report(r2a, r2b, r2d, r2e, PLOT_DIR)
    print(f"Plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
