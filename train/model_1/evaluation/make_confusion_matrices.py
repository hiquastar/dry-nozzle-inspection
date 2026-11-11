"""
make_confusion_matrices.py - Generate confusion matrices for Chapter 6.

Outputs (under repo_root/figures/):
  cm_class_<run>.png/.tex   4-class classification confusion matrix
  cm_obs_<run>.png/.tex     2-class observability confusion matrix (S1 cells)
  detection_summary_<run>.tex   per-class TP/FP/FN/precision/recall/F1

Run from train/train_v9_v1/ :
    python make_confusion_matrices.py
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "structure"))

from dataset import AFGNetDataset, afgnet_collate
from metrics import MetricsAccumulator, box_iou, cxcywh_to_xyxy

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DEVICE,
    FACING_IDX,
    NUM_CLASSES,
    NUM_WORKERS,
    REPO_ROOT,
    SPRAY_IDX,
    TEST_JSON,
)

RUNS = [
    ("full", "v9_fixes_reverse_seed42", "train_v9_v1.pth", "Full model", "train_v9_v1"),
    (
        "abl_A",
        "v9_rabl_A_no_hier_seed42",
        "train_v9_v1_rabl_A.pth",
        "Abl A — no hier mask",
        "train_v9_v1_rabl_A",
    ),
    (
        "abl_B",
        "v9_rabl_B_no_cellkpt_seed42",
        "train_v9_v1_rabl_B.pth",
        "Abl B — no cell-aware kpt",
        "train_v9_v1_rabl_B",
    ),
    (
        "abl_C",
        "v9_rabl_C_no_flipswap_seed42",
        "train_v9_v1_rabl_C.pth",
        "Abl C — no flip swap",
        "train_v9_v1_rabl_C",
    ),
    (
        "abl_D",
        "v9_abl_D_no_obs_head_seed42",
        "train_v9_v1_abl_D_no_obs_head.pth",
        "Abl D — no obs head",
        "train_v9_v1_abl_D_no_obs_head",
    ),
]

FIG_DIR = REPO_ROOT / "train" / "model_1" / "results" / "figures"
SHORT_NAMES = ["spray", "no_spray", "facing", "big_nozzle"]


def import_model_class(train_dir_name: str):
    model_py = REPO_ROOT / "train" / train_dir_name / "model.py"
    spec = importlib.util.spec_from_file_location(f"model_{train_dir_name}", model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.AFGNet


def load_model(ckpt_path, AFGNetCls):
    model = AFGNetCls(pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt["model"]
    for k in ("head.cell_x", "head.cell_y"):
        state.pop(k, None)
    res = model.load_state_dict(state, strict=False)
    miss = [
        k
        for k in res.missing_keys
        if k not in ("head.cell_x", "head.cell_y", "head.kpt_rho")
    ]
    if miss:
        raise RuntimeError(f"Missing learned keys: {miss}")
    model.to(DEVICE).eval()
    return model, ckpt


def run_accumulator(model, loader):
    acc = MetricsAccumulator()
    with torch.no_grad():
        for imgs, tgts in loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            acc.update_from_batch(out, tgts, DEVICE)
    return acc


def detection_tp_fp_fn(acc, iou_thresh=0.5):
    """Per-class TP/FP/FN from accumulator's pred_per_image + gt_per_image."""
    results = {c: {"TP": 0, "FP": 0, "FN": 0} for c in range(NUM_CLASSES)}

    for preds, gts, gt_classes in zip(
        acc.pred_per_image, acc.gt_per_image, acc.gt_classes_per_image
    ):
        # Per-class: GT and predictions
        for cls_id in range(NUM_CLASSES):
            cls_gts = [
                (i, gb)
                for i, (gb, gc) in enumerate(zip(gts, gt_classes))
                if gc == cls_id
            ]
            cls_preds = sorted(
                [p for p in preds if p["class_id"] == cls_id],
                key=lambda p: -p["score"],
            )

            matched_gt = set()
            for p in cls_preds:
                best_iou, best_idx = iou_thresh, None
                for gi, gb in cls_gts:
                    if gi in matched_gt:
                        continue
                    iou = box_iou(p["box"], cxcywh_to_xyxy(gb))
                    if iou > best_iou:
                        best_iou, best_idx = iou, gi
                if best_idx is not None:
                    matched_gt.add(best_idx)
                    results[cls_id]["TP"] += 1
                else:
                    results[cls_id]["FP"] += 1
            results[cls_id]["FN"] += len(cls_gts) - len(matched_gt)

    # Derive precision/recall/F1
    for cls_id, d in results.items():
        tp, fp, fn = d["TP"], d["FP"], d["FN"]
        d["precision"] = tp / (tp + fp) if (tp + fp) else float("nan")
        d["recall"] = tp / (tp + fn) if (tp + fn) else float("nan")
        if d["precision"] and d["recall"]:
            d["F1"] = 2 * d["precision"] * d["recall"] / (d["precision"] + d["recall"])
        else:
            d["F1"] = 0.0
    return results


def plot_confusion(cm, labels, title, out_path, normalize="row", colour="Blues"):
    """Plot a heatmap confusion matrix with absolute counts + row-normalised %."""
    if normalize == "row":
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_pct = cm / row_sums
    else:
        cm_pct = cm / max(cm.sum(), 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_pct, cmap=colour, vmin=0, vmax=1, aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(title, fontsize=12)

    n = len(labels)
    for i in range(n):
        for j in range(n):
            count = cm[i, j]
            pct = cm_pct[i, j] * 100
            # White text if cell is dark, black if light
            color = "white" if cm_pct[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def cm_to_latex(cm, labels, caption, label):
    """Return a LaTeX tabular for a confusion matrix (row-normalised %)."""
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_pct = cm / row_sums * 100

    n = len(labels)
    head_cols = "|c|" + "c" * n + "|"
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{tab:" + label + r"}",
        r"\begin{tabular}{" + head_cols + r"}",
        r"\hline",
        r"\textbf{Actual \textbackslash{} Predicted} & "
        + " & ".join(r"\textbf{" + l.replace("_", r"\_") + "}" for l in labels)
        + r" \\",
        r"\hline",
    ]
    for i, row_label in enumerate(labels):
        cells = []
        for j in range(n):
            v = cm[i, j]
            pct = cm_pct[i, j]
            if i == j:
                cells.append(r"\textbf{" + f"{v} ({pct:.1f}\\%)" + r"}")
            else:
                cells.append(f"{v} ({pct:.1f}\\%)")
        lines.append(row_label.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def detection_table_latex(det, caption, label):
    """LaTeX table of per-class TP/FP/FN/P/R/F1."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{tab:" + label + r"}",
        r"\begin{tabular}{|l|r|r|r|r|r|r|}",
        r"\hline",
        r"\textbf{Class} & \textbf{TP} & \textbf{FP} & \textbf{FN} & "
        r"\textbf{Precision} & \textbf{Recall} & \textbf{F1} \\",
        r"\hline",
    ]
    for cls_id, name in enumerate(SHORT_NAMES):
        d = det[cls_id]
        safe_name = name.replace("_", r"\_")
        lines.append(
            f'{safe_name} & {d["TP"]} & {d["FP"]} & {d["FN"]} '
            f'& {d["precision"]:.3f} & {d["recall"]:.3f} & {d["F1"]:.3f} \\\\'
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Filter RUNS by key (e.g. --only abl_D). " "Default: run all.",
    )
    args = parser.parse_args()

    FIG_DIR.mkdir(exist_ok=True)

    runs = RUNS if args.only is None else [r for r in RUNS if r[0] in args.only]
    if not runs:
        sys.exit(f"No runs match --only {args.only}; available: {[r[0] for r in RUNS]}")
    print(f"Running {len(runs)} model(s): {[r[0] for r in runs]}")

    # Shared test dataset
    print(f"Loading test set from {TEST_JSON}...")
    test_ds = AFGNetDataset(TEST_JSON, augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=afgnet_collate,
        pin_memory=(DEVICE.type == "cuda"),
    )
    print(f"  {len(test_ds)} images")

    summary = {}

    for key, run_dir, ckpt_name, label, train_dir in runs:
        ckpt_path = REPO_ROOT / "train" / "model_1" / "checkpoint" / run_dir / ckpt_name
        if not ckpt_path.exists():
            print(f"SKIP {label}: ckpt missing")
            continue

        print(f"\n=== {label} ===")
        AFGNetCls = import_model_class(train_dir)
        model, ckpt = load_model(ckpt_path, AFGNetCls)
        print(f'  ep={ckpt["epoch"]}, val_composite={ckpt["composite"]:.4f}')

        acc = run_accumulator(model, test_loader)

        # ── 4-class classification confusion matrix (cell-level) ─────────────
        cm_class = confusion_matrix(
            acc.cls_y_true,
            acc.cls_y_pred,
            labels=list(range(NUM_CLASSES)),
        )
        plot_confusion(
            cm_class,
            SHORT_NAMES,
            title=f"{label} — 4-class classification (test set, cell-level)",
            out_path=FIG_DIR / f"cm_class_{key}.png",
            normalize="row",
        )
        latex_class = cm_to_latex(
            cm_class,
            SHORT_NAMES,
            caption=f"Cell-level classification confusion matrix on the test set ({label}). "
            f"Rows = ground-truth class, columns = predicted class. Diagonal cells "
            f"(in bold) are per-class recall.",
            label=f"cm_class_{key}",
        )
        (FIG_DIR / f"cm_class_{key}.tex").write_text(latex_class, encoding="utf-8")

        # ── 2-class observability confusion matrix (S1 cells) ────────────────
        if acc.obs_y_true:
            cm_obs = confusion_matrix(
                acc.obs_y_true,
                acc.obs_y_pred,
                labels=[0, 1],
            )
            obs_labels = ["facing (obs=0)", "spray (obs=1)"]
            plot_confusion(
                cm_obs,
                obs_labels,
                title=f"{label} — observability on S1 cells (test set)",
                out_path=FIG_DIR / f"cm_obs_{key}.png",
                normalize="row",
                colour="Oranges",
            )
            latex_obs = cm_to_latex(
                cm_obs,
                obs_labels,
                caption=f"Observability confusion matrix on S1 cells "
                f"(class $\\in$ \\{{spray, facing\\}}) for {label}. "
                f"Rows = ground-truth observability target "
                f"(1 for spray cells, 0 for facing cells), "
                f"columns = predicted observability (threshold 0.5).",
                label=f"cm_obs_{key}",
            )
            (FIG_DIR / f"cm_obs_{key}.tex").write_text(latex_obs, encoding="utf-8")

        # ── Per-class detection summary (instance-level after NMS) ───────────
        det = detection_tp_fp_fn(acc, iou_thresh=0.5)
        latex_det = detection_table_latex(
            det,
            caption=f"Per-class detection summary on the test set for {label}, "
            f"computed at instance level after NMS with IoU $> 0.5$ for "
            f"true-positive matching.",
            label=f"det_summary_{key}",
        )
        (FIG_DIR / f"detection_summary_{key}.tex").write_text(
            latex_det, encoding="utf-8"
        )

        summary[key] = {
            "label": label,
            "cm_class": cm_class.tolist(),
            "cm_obs": cm_obs.tolist() if acc.obs_y_true else None,
            "detection": det,
            "n_cls_samples": len(acc.cls_y_true),
            "n_obs_samples": len(acc.obs_y_true) if acc.obs_y_true else 0,
        }

        print(f"  classification CM:")
        print(f'    {"":10s}' + "".join(f"{n[:8]:>10s}" for n in SHORT_NAMES))
        for i, name in enumerate(SHORT_NAMES):
            row = "".join(f"{cm_class[i, j]:>10d}" for j in range(NUM_CLASSES))
            print(f"    {name[:10]:10s}{row}")

        print(f"\n  detection per-class:")
        print(
            f'    {"class":<12s} {"TP":>5s} {"FP":>5s} {"FN":>5s} {"P":>6s} {"R":>6s} {"F1":>6s}'
        )
        for cls_id, name in enumerate(SHORT_NAMES):
            d = det[cls_id]
            print(
                f'    {name:<12s} {d["TP"]:>5d} {d["FP"]:>5d} {d["FN"]:>5d} '
                f'{d["precision"]:>6.3f} {d["recall"]:>6.3f} {d["F1"]:>6.3f}'
            )

    # Merge into existing JSON rather than overwrite, so a partial run
    # (e.g. --only abl_D) keeps the previously written entries for the
    # other models.
    json_path = FIG_DIR / "confusion_matrices.json"
    merged = {}
    if json_path.exists():
        with open(json_path) as f:
            merged = json.load(f)
    merged.update(summary)
    with open(json_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\nAll outputs written to: {FIG_DIR}")


if __name__ == "__main__":
    main()
