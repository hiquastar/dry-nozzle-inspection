"""
confusion_matrix.py - All-in-one Confusion Matrix Tool

Generates 3 confusion matrices in a single run:
  1. CLASS (4x4): 4-class softmax classification
  2. OBSERVABILITY (2x2): Binary measurable/non-measurable (spraying cells only)
  3. DETECTION (4x4): Detection after NMS - PURE 4-class confusion
                       FP/FN tracked separately as numbers (NOT in matrix)

Outputs:
  - 1 COMBINED plot with all 3 matrices in one large image (combined_cm.png)
  - 3 separate plots, one per matrix (for detail)
  - 1 summary CSV + 3 individual CSVs
  - Print metrics ra terminal (TP/FP/FN/Precision/Recall/F1)

Usage:
    python confusion_matrix.py                       # latest run, test set
    python confusion_matrix.py --split val           # val set
    python confusion_matrix.py --run train/model_1/checkpoint/v9_xxx     # specific run
    python confusion_matrix.py --ckpt path/to/.pth   # specific checkpoint
    python confusion_matrix.py --iou_thresh 0.5 --score_thresh 0.35 --obs_thresh 0.5
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "structure"))

from dataset import AFGNetDataset, afgnet_collate
from matplotlib.gridspec import GridSpec
from metrics import box_iou, cxcywh_to_xyxy, decode_predictions
from model import AFGNet
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DEVICE,
    FACING_IDX,
    NMS_IOU_THRESH,
    NUM_CLASSES,
    NUM_WORKERS,
    OBJ_THRESH,
    REPO_ROOT,
    SPRAY_IDX,
    TEST_JSON,
    VAL_JSON,
)

# Detection CM = pure 4x4 (NO background row/col)
# FP/FN tracked separately as scalars

OBS_NAMES = ["Non-Measurable\n(facing)", "Measurable\n(spray)"]


# ============================================================================
# Run / checkpoint loading
# ============================================================================
def find_latest_run():
    runs_dir = REPO_ROOT / "train" / "model_1" / "checkpoint"
    if not runs_dir.exists():
        raise FileNotFoundError(f"No runs/ folder at {runs_dir}")
    # Only pick runs whose checkpoint matches single-scale architecture
    v9_runs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("v9_")]
    )
    for rd in reversed(v9_runs):
        pth = next(rd.glob("*.pth"), None)
        if pth is None:
            continue
        ckpt = torch.load(pth, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        if not any("smooth_p" in k for k in state.keys()):
            return rd
    raise FileNotFoundError("No single-scale v9 run found")


def find_checkpoint(run_dir):
    pth_files = list(Path(run_dir).glob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth in {run_dir}")
    if len(pth_files) > 1:
        for f in pth_files:
            if "v9" in f.name.lower():
                return f
    return pth_files[0]


def load_model(ckpt_path):
    import sys

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Auto-detect architecture from checkpoint keys
    is_multiscale = any("smooth_p" in k for k in state.keys())
    is_decoupled = any("obs_head" in k for k in state.keys())

    if is_multiscale:
        import importlib.util
        import sys as _sys

        if is_decoupled:
            train_dir = REPO_ROOT / "train" / "train_v9_v4"
            print(f"  Detected: multi-scale + decoupled head (train_v9_v4)")
        else:
            train_dir = REPO_ROOT / "train" / "train_v9_v2"
            print(f"  Detected: multi-scale (train_v9_v2)")

        def _load_mod(name, filepath):
            spec = importlib.util.spec_from_file_location(name, filepath)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        # Load config + model from the correct folder, inject into sys.modules
        cfg_mod = _load_mod("config", train_dir / "config.py")
        _sys.modules["config_ext"] = cfg_mod
        old_config = _sys.modules.get("config")
        _sys.modules["config"] = cfg_mod  # temporarily override
        model_mod = _load_mod("model_ext", train_dir / "model.py")
        _sys.modules["config"] = old_config  # restore
        ModelClass = model_mod.AFGNet
    else:
        print(f"  Detected: single-scale (train_v9_v1)")
        ModelClass = AFGNet

    model = ModelClass(pretrained=False).to(DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()

    if isinstance(ckpt, dict):
        if "epoch" in ckpt:
            print(f'  Epoch: {ckpt["epoch"]}')
        if "composite" in ckpt:
            print(f'  Composite: {ckpt["composite"]:.4f}')
    return model


# ============================================================================
# Data collection (1 inference pass, collect everything)
# ============================================================================
def get_gt_boxes_classes(targets, batch_idx):
    """Extract unique GT boxes + classes (deduplicated)."""
    pos_mask = targets["pos_mask"][batch_idx].bool().view(-1)
    if pos_mask.sum() == 0:
        return [], []

    bb = targets["bbox"][batch_idx].view(-1, 4)[pos_mask].cpu().numpy()
    cc = targets["class_ids"][batch_idx].view(-1)[pos_mask].cpu().numpy()

    seen = set()
    gt_boxes = []
    gt_classes = []
    for box, cls_id in zip(bb, cc):
        key = tuple(np.round(box, 4))
        if key not in seen:
            seen.add(key)
            gt_boxes.append(box)
            gt_classes.append(int(cls_id))
    return gt_boxes, gt_classes


@torch.no_grad()
def collect_all_data(model, loader, iou_thresh, score_thresh, obs_thresh):
    """Single inference pass — collect data for all 3 matrices.

    Detection CM: pure 4x4 (only TP-with-class-confusion).
    FP/FN tracked separately:
      - fp_per_class[c] = predictions of class c with no matching GT
      - fn_per_class[c] = GTs of class c not detected
    """

    # Class CM data (cell-level)
    cls_y_true, cls_y_pred = [], []

    # Observability CM data (cell-level, spraying cells only)
    obs_y_true, obs_y_pred = [], []
    obs_y_pred_raw = []

    # Detection CM (image-level, after NMS) - PURE 4x4
    det_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    # FP/FN tracked separately per class
    fp_per_class = np.zeros(NUM_CLASSES, dtype=np.int64)
    fn_per_class = np.zeros(NUM_CLASSES, dtype=np.int64)

    for imgs, tgts in tqdm(loader, desc="Inference", unit="batch"):
        imgs = imgs.to(DEVICE)
        out = model(imgs)

        B = imgs.shape[0]
        pos_mask = tgts["pos_mask"].to(DEVICE).bool().view(-1)

        # ─────── Cell-level: cls + obs ───────
        if pos_mask.sum() > 0:
            cls_logits = out["class_logits"]
            cls_flat = cls_logits.permute(0, 2, 3, 1).reshape(-1, NUM_CLASSES)
            cls_pred_pos = cls_flat[pos_mask].argmax(dim=1).cpu().numpy()
            cls_true_pos = tgts["class_ids"].to(DEVICE).view(-1)[pos_mask].cpu().numpy()

            cls_y_true.extend(cls_true_pos.tolist())
            cls_y_pred.extend(cls_pred_pos.tolist())

            # Observability — ONLY on spraying cells
            spraying_mask = (cls_true_pos == SPRAY_IDX) | (cls_true_pos == FACING_IDX)
            if spraying_mask.sum() > 0:
                obs_pred_all = (
                    out["observability"].permute(0, 2, 3, 1).reshape(-1, 1).squeeze(-1)
                )
                obs_pred_pos = obs_pred_all[pos_mask].cpu().numpy()
                obs_pred_filt = obs_pred_pos[spraying_mask]

                obs_true_filt = (cls_true_pos[spraying_mask] == SPRAY_IDX).astype(int)

                obs_y_true.extend(obs_true_filt.tolist())
                obs_y_pred_raw.extend(obs_pred_filt.tolist())
                obs_y_pred.extend((obs_pred_filt > obs_thresh).astype(int).tolist())

        # ─────── Image-level: detection CM (pure 4x4) ───────
        for b in range(B):
            preds = decode_predictions(
                out,
                batch_idx=b,
                obj_thresh=score_thresh,
                nms_iou=NMS_IOU_THRESH,
            )
            gt_boxes, gt_classes = get_gt_boxes_classes(tgts, b)

            preds_sorted = sorted(preds, key=lambda p: -p["score"])
            gt_matched = [False] * len(gt_boxes)

            for pred in preds_sorted:
                pred_box = pred["box"]
                pred_cls = pred["class_id"]

                best_iou = 0.0
                best_gt_idx = -1
                for gi, gb in enumerate(gt_boxes):
                    if gt_matched[gi]:
                        continue
                    iou = box_iou(pred_box, cxcywh_to_xyxy(gb))
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gi

                if best_iou >= iou_thresh and best_gt_idx >= 0:
                    # Matched → put into CM (class confusion)
                    gt_cls = gt_classes[best_gt_idx]
                    det_cm[gt_cls, pred_cls] += 1
                    gt_matched[best_gt_idx] = True
                else:
                    # Unmatched prediction → False Positive (track separately)
                    fp_per_class[pred_cls] += 1

            # Unmatched GTs → False Negative (track separately)
            for gi, matched in enumerate(gt_matched):
                if not matched:
                    fn_per_class[gt_classes[gi]] += 1

    return {
        "cls_y_true": np.array(cls_y_true),
        "cls_y_pred": np.array(cls_y_pred),
        "obs_y_true": np.array(obs_y_true),
        "obs_y_pred": np.array(obs_y_pred),
        "obs_y_pred_raw": np.array(obs_y_pred_raw),
        "det_cm": det_cm,
        "fp_per_class": fp_per_class,
        "fn_per_class": fn_per_class,
    }


# ============================================================================
# Confusion matrix computation
# ============================================================================
def compute_cm(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


# ============================================================================
# Plotting helpers
# ============================================================================
def draw_cm_on_axis(
    ax,
    cm,
    class_names,
    title,
    cmap="Blues",
    annotate_fontsize=11,
    normalize=False,
    tag_corners=False,
):
    """Draw 1 confusion matrix on given axis. Reusable for combined plot."""
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1)
        cm_display = cm.astype(np.float64) / row_sums
    else:
        cm_display = cm.astype(np.float64)

    im = ax.imshow(cm_display, cmap=cmap, aspect="auto", vmin=0)

    n = len(class_names)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, fontsize=10, rotation=20, ha="right")
    ax.set_yticklabels(class_names, fontsize=10)

    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("True (GT)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    thresh = cm_display.max() / 2.0 if cm_display.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            val = cm_display[i, j]
            color = "white" if val > thresh else "black"

            if cm.sum() > 0:
                row_sum = cm[i].sum()
                pct = 100.0 * cm[i, j] / row_sum if row_sum > 0 else 0
                text_str = f"{int(cm[i, j])}\n({pct:.1f}%)"
            else:
                text_str = "0"

            # TP/TN/FP/FN tags for 2x2
            if tag_corners and n == 2:
                tag = ["TN", "FP", "FN", "TP"][i * 2 + j]
                text_str = f"{tag}\n{text_str}"

            ax.text(
                j,
                i,
                text_str,
                ha="center",
                va="center",
                color=color,
                fontsize=annotate_fontsize,
                fontweight="bold",
            )

    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def plot_combined(
    cls_cm,
    obs_cm,
    det_cm,
    fp_per_class,
    fn_per_class,
    save_path,
    split_name,
    iou_thresh,
    score_thresh,
    obs_thresh,
):
    """Combined plot: 3 matrices side-by-side in one LARGE image.

    Detection CM is now pure 4x4. Total FP/FN are shown in the title.
    """

    # Figure 22x9 inch
    fig = plt.figure(figsize=(22, 9))

    # Detection is also 4x4, so the width ratios are more balanced
    gs = GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=[1.0, 0.7, 1.0],
        wspace=0.35,
        left=0.05,
        right=0.97,
        top=0.83,
        bottom=0.18,
    )

    # Class CM
    ax1 = fig.add_subplot(gs[0, 0])
    draw_cm_on_axis(
        ax1,
        cls_cm,
        CLASS_NAMES,
        title="1. CLASS CM (4×4)\n4-class on positive cells",
        cmap="Blues",
        annotate_fontsize=10,
    )

    # Observability CM
    ax2 = fig.add_subplot(gs[0, 1])
    draw_cm_on_axis(
        ax2,
        obs_cm,
        OBS_NAMES,
        title=f"2. OBSERVABILITY CM (2×2)\nthr={obs_thresh}, spraying cells only",
        cmap="Greens",
        annotate_fontsize=14,
        tag_corners=True,
    )

    # Detection CM (pure 4x4)
    total_fp = fp_per_class.sum()
    total_fn = fn_per_class.sum()
    ax3 = fig.add_subplot(gs[0, 2])
    draw_cm_on_axis(
        ax3,
        det_cm,
        CLASS_NAMES,
        title=(
            f"3. DETECTION CM (4×4)\n"
            f"IoU≥{iou_thresh}, score≥{score_thresh}  |  "
            f"Total FP={total_fp}, FN={total_fn}"
        ),
        cmap="Oranges",
        annotate_fontsize=10,
    )

    # Super title
    fig.suptitle(
        f"AFGNet V9 Confusion Matrices — {split_name.upper()} split",
        fontsize=18,
        fontweight="bold",
        y=0.97,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved combined plot: {save_path}")
    plt.close()


def plot_single_cm(
    cm,
    class_names,
    save_path,
    title,
    cmap="Blues",
    figsize=(12, 10),
    annotate_fontsize=13,
    tag_corners=False,
    normalize=False,
):
    """Plot a single matrix at large size."""
    fig, ax = plt.subplots(figsize=figsize)
    im = draw_cm_on_axis(
        ax,
        cm,
        class_names,
        title,
        cmap=cmap,
        annotate_fontsize=annotate_fontsize,
        normalize=normalize,
        tag_corners=tag_corners,
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


def plot_obs_score_distribution(y_true, y_pred_raw, save_path, threshold):
    """Histogram of the distribution of obs scores."""
    fig, ax = plt.subplots(figsize=(12, 7))

    facing_scores = y_pred_raw[y_true == 0]
    spray_scores = y_pred_raw[y_true == 1]

    bins = np.linspace(0, 1, 51)

    ax.hist(
        facing_scores,
        bins=bins,
        alpha=0.6,
        label=f"Non-Measurable / facing (n={len(facing_scores)})",
        color="salmon",
        edgecolor="darkred",
    )
    ax.hist(
        spray_scores,
        bins=bins,
        alpha=0.6,
        label=f"Measurable / spray (n={len(spray_scores)})",
        color="lightgreen",
        edgecolor="darkgreen",
    )

    ax.axvline(
        x=threshold,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold}",
    )

    ax.set_xlabel("Predicted Observability Score", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title(
        "Observability Score Distribution by True Class", fontsize=16, fontweight="bold"
    )
    ax.legend(fontsize=12, loc="upper center")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


# ============================================================================
# Save CSVs
# ============================================================================
def save_cm_csv(cm, class_names, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("GT\\Pred," + ",".join(class_names) + ",Total\n")
        for i, name in enumerate(class_names):
            row = [str(cm[i, j]) for j in range(len(class_names))]
            row.append(str(cm[i].sum()))
            f.write(f"{name}," + ",".join(row) + "\n")
        col_totals = [str(cm[:, j].sum()) for j in range(len(class_names))]
        col_totals.append(str(cm.sum()))
        f.write("Total," + ",".join(col_totals) + "\n")


# ============================================================================
# Metrics printing
# ============================================================================
def print_class_metrics(cm):
    print("\n" + "=" * 70)
    print("1. CLASS CONFUSION MATRIX METRICS (per positive cell)")
    print("=" * 70)
    print(f'{"Class":<20} {"Precision":>10} {"Recall":>10} {"F1":>10} {"Support":>10}')
    print("-" * 70)

    f1_scores, supports = [], []
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        f1_scores.append(f1)
        supports.append(support)

        print(
            f"{name:<20} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}"
        )

    print("-" * 70)
    accuracy = np.trace(cm) / (cm.sum() + 1e-7)
    macro_f1 = np.mean(f1_scores)
    weighted_f1 = sum(f1_scores[i] * supports[i] for i in range(len(CLASS_NAMES))) / (
        cm.sum() + 1e-7
    )

    print(f'{"Accuracy":<20} {accuracy:>10.4f}')
    print(f'{"Macro F1":<20} {macro_f1:>10.4f}')
    print(f'{"Weighted F1":<20} {weighted_f1:>10.4f}')


def print_obs_metrics(cm):
    print("\n" + "=" * 70)
    print("2. OBSERVABILITY CONFUSION MATRIX METRICS")
    print("=" * 70)

    if cm.sum() == 0:
        print("No spraying cells found!")
        return

    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]

    accuracy = (tp + tn) / (cm.sum() + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    specificity = tn / (tn + fp + 1e-7)

    print(f"  TP (Measurable correctly identified):     {tp}")
    print(f"  TN (Non-Measurable correctly identified): {tn}")
    print(f"  FP (Non-Measurable WRONG -> Measurable): {fp}")
    print(f"  FN (Measurable WRONG -> Non-Measurable): {fn}")
    print(f"  Total spraying samples: {cm.sum()}")
    print()
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}  (sensitivity)")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  F1 Score:    {f1:.4f}")


def print_detection_metrics(cm, fp_per_class, fn_per_class):
    """Detection metrics: TP from CM diagonal, FP/FN from separate tracking.

    Definitions:
      TP[c] = cm[c, c]                    (predicted c, GT also c, IoU match)
      FP[c] = fp_per_class[c]             (predicted c, NO matching GT)
            + sum(cm[i, c] for i != c)    (predicted c, but GT was different class)
      FN[c] = fn_per_class[c]             (GT c not detected)
            + sum(cm[c, j] for j != c)    (GT c, but predicted as different class)
    """
    print("\n" + "=" * 70)
    print("3. DETECTION METRICS (after NMS)")
    print("=" * 70)
    print(
        f'{"Class":<18} {"TP":>6} {"FP":>6} {"FN":>6} ' f'{"P":>8} {"R":>8} {"F1":>8}'
    )
    print("-" * 70)

    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        # FP: predicted i with no GT + predicted i but actual GT was different class
        fp_misclass = cm[:, i].sum() - tp
        fp = fp_per_class[i] + fp_misclass
        # FN: GT i not detected + GT i predicted as different class
        fn_misclass = cm[i, :].sum() - tp
        fn = fn_per_class[i] + fn_misclass

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)

        print(
            f"{CLASS_NAMES[i]:<18} {tp:>6} {fp:>6} {fn:>6} "
            f"{precision:>8.4f} {recall:>8.4f} {f1:>8.4f}"
        )

    print("-" * 70)

    print(f'\nFP from "no GT match" (predictions that match no GT):')
    for i in range(NUM_CLASSES):
        if fp_per_class[i] > 0:
            print(f"  {CLASS_NAMES[i]:<18} {fp_per_class[i]}")
    print(f"  Total: {fp_per_class.sum()}")

    print(f'\nFN from "GT not detected" (objects missed by the model):')
    for i in range(NUM_CLASSES):
        if fn_per_class[i] > 0:
            print(f"  {CLASS_NAMES[i]:<18} {fn_per_class[i]}")
    print(f"  Total: {fn_per_class.sum()}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="V9 Confusion Matrix Tool")
    parser.add_argument(
        "--run", type=str, default=None, help="Run directory (default: latest)"
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Checkpoint path (default: auto-detect)"
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.5,
        help="IoU threshold for detection matching (default 0.5)",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=None,
        help=f"Detection score threshold (default {OBJ_THRESH})",
    )
    parser.add_argument(
        "--obs_thresh",
        type=float,
        default=0.5,
        help="Observability binary threshold (default 0.5)",
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Skip individual plots, only combined",
    )
    args = parser.parse_args()

    # Setup
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        run_dir = ckpt_path.parent
    else:
        run_dir = Path(args.run) if args.run else find_latest_run()
        ckpt_path = find_checkpoint(run_dir)

    score_thresh = args.score_thresh if args.score_thresh is not None else OBJ_THRESH

    print("=" * 70)
    print("V9 Confusion Matrix Tool")
    print("=" * 70)
    print(f"Run dir:         {run_dir}")
    print(f"Checkpoint:      {ckpt_path.name}")
    print(f"Split:           {args.split}")
    print(f"IoU threshold:   {args.iou_thresh}")
    print(f"Score threshold: {score_thresh}")
    print(f"Obs threshold:   {args.obs_thresh}")
    print()

    # Load model
    model = load_model(ckpt_path)

    # Load dataset
    json_path = TEST_JSON if args.split == "test" else VAL_JSON
    print(f"\nLoading {args.split} dataset...")
    ds = AFGNetDataset(json_path, augment=False)
    print(f"  {len(ds)} samples loaded")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=afgnet_collate,
        pin_memory=True,
    )

    # Single inference pass — collect all data
    print("\nRunning inference (1 pass for all 3 matrices)...")
    data = collect_all_data(
        model,
        loader,
        iou_thresh=args.iou_thresh,
        score_thresh=score_thresh,
        obs_thresh=args.obs_thresh,
    )

    # Compute matrices
    cls_cm = compute_cm(data["cls_y_true"], data["cls_y_pred"], NUM_CLASSES)
    obs_cm = compute_cm(data["obs_y_true"], data["obs_y_pred"], 2)
    det_cm = data["det_cm"]
    fp_per_class = data["fp_per_class"]
    fn_per_class = data["fn_per_class"]

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f'  Total positive cells:    {len(data["cls_y_true"])}')
    print(f'  Total spraying cells:    {len(data["obs_y_true"])}')
    print(f'    Measurable (spray):    {(data["obs_y_true"] == 1).sum()}')
    print(f'    Non-meas (facing):     {(data["obs_y_true"] == 0).sum()}')
    print(f"  Total detection matches (in CM): {det_cm.sum()}")
    print(f"  False Positives (no GT match):   {fp_per_class.sum()}")
    print(f"  False Negatives (GT not detected): {fn_per_class.sum()}")

    # Print metrics
    print_class_metrics(cls_cm)
    print_obs_metrics(obs_cm)
    print_detection_metrics(det_cm, fp_per_class, fn_per_class)

    # Save outputs
    out_dir = run_dir / f"cm_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Saving outputs...")
    print("=" * 70)

    # 1. Combined plot (main)
    plot_combined(
        cls_cm,
        obs_cm,
        det_cm,
        fp_per_class,
        fn_per_class,
        out_dir / "combined_cm.png",
        split_name=args.split,
        iou_thresh=args.iou_thresh,
        score_thresh=score_thresh,
        obs_thresh=args.obs_thresh,
    )

    # 2. Individual plots (detailed, unless --no-individual)
    if not args.no_individual:
        plot_single_cm(
            cls_cm,
            CLASS_NAMES,
            out_dir / "cm_class.png",
            title=f"Class Confusion Matrix - {args.split.upper()}",
            cmap="Blues",
            figsize=(12, 10),
            annotate_fontsize=13,
        )
        plot_single_cm(
            obs_cm,
            OBS_NAMES,
            out_dir / "cm_observability.png",
            title=f"Observability CM - {args.split.upper()} (thr={args.obs_thresh})",
            cmap="Greens",
            figsize=(10, 9),
            annotate_fontsize=18,
            tag_corners=True,
        )
        plot_single_cm(
            det_cm,
            CLASS_NAMES,
            out_dir / "cm_detection.png",
            title=(
                f"Detection CM (4×4) - {args.split.upper()} "
                f"(IoU≥{args.iou_thresh}, score≥{score_thresh})\n"
                f"Total FP={fp_per_class.sum()}, FN={fn_per_class.sum()}"
            ),
            cmap="Oranges",
            figsize=(12, 10),
            annotate_fontsize=13,
        )

        # Observability score distribution
        if len(data["obs_y_pred_raw"]) > 0:
            plot_obs_score_distribution(
                data["obs_y_true"],
                data["obs_y_pred_raw"],
                out_dir / "obs_score_distribution.png",
                threshold=args.obs_thresh,
            )

    # 3. CSVs
    save_cm_csv(cls_cm, CLASS_NAMES, out_dir / "cm_class.csv")
    save_cm_csv(
        obs_cm, ["Non-Measurable", "Measurable"], out_dir / "cm_observability.csv"
    )
    save_cm_csv(det_cm, CLASS_NAMES, out_dir / "cm_detection.csv")

    # Bonus: save FP/FN summary
    with open(out_dir / "fp_fn_summary.csv", "w", encoding="utf-8") as f:
        f.write("Class,FP_no_match,FN_not_detected\n")
        for i in range(NUM_CLASSES):
            f.write(f"{CLASS_NAMES[i]},{fp_per_class[i]},{fn_per_class[i]}\n")
        f.write(f"Total,{fp_per_class.sum()},{fn_per_class.sum()}\n")
    print(f'  Saved: {out_dir / "fp_fn_summary.csv"}')

    print(f"\nAll outputs saved in: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
