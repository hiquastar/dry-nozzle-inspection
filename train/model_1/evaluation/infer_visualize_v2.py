"""
infer_visualize_v2.py - Coverage check: 1 sample per video source

Instead of N random images, v2 takes 1 representative image from EACH video source in the split.
Purpose: check whether the model works well across all cameras/days.

Sources in test:  CAM2-V2, CAM2-V3, 20250510, 20250509_frames_275-304
Sources in val:   CAM2-V1, 20250509_frames_230-261
Sources in train: CAM1-V1, CAM1-V2, CAM1-V3, 20250508, 20250509_frames_0-213

Usage:
    python infer_visualize_v2.py                    # test split, 1 per source
    python infer_visualize_v2.py --split val
    python infer_visualize_v2.py --split train
    python infer_visualize_v2.py --split all        # all splits combined
    python infer_visualize_v2.py --n 2              # 2 images per source
    python infer_visualize_v2.py --threshold 0.20
    python infer_visualize_v2.py --seed 99          # different random seed
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "structure"))

from dataset import resolve_image_path
from metrics import compute_spray_angle, decode_predictions
from model import AFGNet

from config import (
    CLASS_NAMES,
    DEVICE,
    IMG_SIZE,
    NMS_IOU_THRESH,
    NUM_KEYPOINTS,
    OBJ_THRESH,
    REPO_ROOT,
    TEST_JSON,
    TRAIN_JSON,
    VAL_JSON,
)

CLASS_COLORS = {
    0: (0, 200, 0),
    1: (255, 165, 0),
    2: (255, 50, 50),
    3: (50, 100, 255),
}
CLASS_LABELS = ["SPRAY", "NO_SPRAY", "FACING", "BIG"]


# ============================================================================
# Model loading (same auto-detect logic as infer_visualize.py)
# ============================================================================
def _is_single_scale(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    return not any("smooth_p" in k for k in state.keys())


def find_latest_run(repo_root):
    runs_dir = Path(repo_root) / "train" / "model_1" / "checkpoint"
    if not runs_dir.exists():
        raise FileNotFoundError(f"No runs/ at {runs_dir}")
    v9_runs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("v9_")]
    )
    for rd in reversed(v9_runs):
        pth = next(rd.glob("*.pth"), None)
        if pth and _is_single_scale(pth):
            return rd
    raise FileNotFoundError("No single-scale v9 run found")


def load_model(run_dir):
    import importlib.util
    import sys as _sys

    run_dir = Path(run_dir)
    candidates = list(run_dir.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No .pth in {run_dir}")
    ckpt_path = candidates[0]

    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    is_multiscale = any("smooth_p" in k for k in state.keys())
    is_decoupled = any("obs_head" in k for k in state.keys())

    if is_multiscale:
        train_dir = (
            (REPO_ROOT / "train" / "train_v9_v4")
            if is_decoupled
            else (REPO_ROOT / "train" / "train_v9_v2")
        )

        def _load_mod(name, filepath):
            spec = importlib.util.spec_from_file_location(name, filepath)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        cfg_mod = _load_mod("config_ext", train_dir / "config.py")
        old_config = _sys.modules.get("config")
        _sys.modules["config"] = cfg_mod
        model_mod = _load_mod("model_ext", train_dir / "model.py")
        _sys.modules["config"] = old_config
        ModelClass = model_mod.AFGNet
    else:
        ModelClass = AFGNet

    model = ModelClass(pretrained=False).to(DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()

    if isinstance(ckpt, dict):
        if "epoch" in ckpt:
            print(f"  Best epoch: {ckpt['epoch']}")
        if "composite" in ckpt:
            print(f"  Composite: {ckpt['composite']:.4f}")
        if "metrics" in ckpt:
            m = ckpt["metrics"]
            if "mAP_50" in m:
                print(f"  mAP@.5: {m['mAP_50']:.3f}")
            if "cls_f1" in m:
                print(f"  cls_f1: {m['cls_f1']:.3f}")
            if "angle_mae_deg" in m:
                print(f"  angle_mae: {m['angle_mae_deg']:.2f}°")
    return model


# ============================================================================
# Data loading — grouped by source_video
# ============================================================================
def load_split_by_source(json_paths):
    """Load samples from one or more splits, group by source_video."""
    by_source = defaultdict(list)

    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_to_anns = defaultdict(list)
        for ann in data.get("annotations", []):
            img_to_anns[ann["image_id"]].append(ann)

        for img_info in data.get("images", []):
            anns = img_to_anns.get(img_info["id"], [])
            if not anns:
                continue
            try:
                src_hint = img_info.get("source_video", None)
                img_path = resolve_image_path(img_info["file_name"], src_hint)
            except FileNotFoundError:
                continue

            # Strip frame range suffix from source name for grouping
            source = (src_hint or "unknown").split("_frames_")[0]

            by_source[source].append(
                {
                    "image_path": str(img_path),
                    "file_name": img_info["file_name"],
                    "source": source,
                    "source_full": src_hint or "unknown",
                    "orig_w": img_info.get("width", 1),
                    "orig_h": img_info.get("height", 1),
                    "anns": anns,
                }
            )

    return by_source


# ============================================================================
# Inference + drawing (same as infer_visualize.py)
# ============================================================================
_transform = T.Compose(
    [
        T.ToTensor(),
        T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@torch.no_grad()
def run_inference(model, image_path, threshold):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = _transform(img_rgb).unsqueeze(0).to(DEVICE)
    preds = model(tensor)
    dets = decode_predictions(
        preds, batch_idx=0, obj_thresh=threshold, nms_iou=NMS_IOU_THRESH
    )
    return img_rgb, dets


def draw_sample(ax, img_rgb, dets, sample, threshold):
    """Draw predictions + GT on one axes."""
    h, w = img_rgb.shape[:2]
    ax.imshow(img_rgb)

    # GT boxes (yellow)
    for ann in sample["anns"]:
        bbox = ann.get("bbox", [])
        if len(bbox) < 4:
            continue
        x, y, bw, bh = [float(v) for v in bbox[:4]]
        rect = plt.Rectangle(
            (x, y),
            bw,
            bh,
            linewidth=1.2,
            edgecolor="yellow",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)

    # Predictions
    for det in dets:
        x1n, y1n, x2n, y2n = det["box"]
        x1, y1 = x1n * w, y1n * h
        x2, y2 = x2n * w, y2n * h
        cls_id = det["class_id"]
        score = det["score"]
        obs = det["observability"]
        kpts = det["keypoints"]

        color = [c / 255 for c in CLASS_COLORS.get(cls_id, (200, 200, 200))]
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        label = CLASS_LABELS[cls_id]
        txt = f"{label} {score:.2f}"
        if cls_id in (0, 2):
            txt += f" obs={obs:.2f}"

        ax.text(
            x1,
            y1 - 3,
            txt,
            fontsize=5.5,
            color="white",
            bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor="none"),
        )

        # Keypoints for spray
        if cls_id == 0:
            kx = [kpts[i] * w for i in (0, 2, 4)]
            ky = [kpts[i] * h for i in (1, 3, 5)]
            ax.plot([kx[0], kx[1]], [ky[0], ky[1]], "-", color="cyan", lw=1)
            ax.plot([kx[0], kx[2]], [ky[0], ky[2]], "-", color="cyan", lw=1)
            ax.scatter(kx[0], ky[0], s=15, c="cyan", zorder=5)
            ax.scatter(kx[1:], ky[1:], s=10, c="magenta", zorder=5)
            angle = compute_spray_angle(kpts)
            if angle is not None:
                ax.text(
                    kx[0],
                    ky[0] - 8,
                    f"{angle:.1f}°",
                    fontsize=5,
                    color="cyan",
                    ha="center",
                )

    # Title
    source = sample["source"]
    n_det = len(dets)
    cls_counts = defaultdict(int)
    for d in dets:
        cls_counts[CLASS_LABELS[d["class_id"]]] += 1
    cls_str = " ".join(f"{k}:{v}" for k, v in sorted(cls_counts.items()))
    ax.set_title(
        f'{source}\n{cls_str or "no detections"} ({n_det} total)', fontsize=6.5, pad=2
    )
    ax.axis("off")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which split(s) to visualize",
    )
    parser.add_argument(
        "--n", type=int, default=1, help="Images per source (default: 1)"
    )
    parser.add_argument("--threshold", type=float, default=OBJ_THRESH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run", type=str, default=None)
    parser.add_argument(
        "--save", type=str, default=None, help="Save path (default: auto in run dir)"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Select split JSONs
    split_map = {
        "train": [TRAIN_JSON],
        "val": [VAL_JSON],
        "test": [TEST_JSON],
        "all": [TRAIN_JSON, VAL_JSON, TEST_JSON],
    }
    json_paths = split_map[args.split]

    # Load model
    run_dir = Path(args.run) if args.run else find_latest_run(REPO_ROOT)
    print(f"Run: {run_dir}")
    model = load_model(run_dir)

    # Load data grouped by source
    print(f"\nLoading split: {args.split}")
    by_source = load_split_by_source(json_paths)
    sources = sorted(by_source.keys())
    print(f"Found {len(sources)} sources: {sources}")

    # Pick N samples per source
    selected = []
    for src in sources:
        pool = by_source[src]
        n = min(args.n, len(pool))
        selected.extend(random.sample(pool, n))

    print(
        f"Total samples to visualize: {len(selected)} "
        f"({args.n} per source x {len(sources)} sources)"
    )

    # Layout: cols=n_sources, rows=n
    n_sources = len(sources)
    n_per = args.n
    total = len(selected)

    # Grid: each source = 1 column, each row = 1 sample per source
    n_cols = n_sources
    n_rows = n_per
    fig_w = max(4 * n_cols, 12)
    fig_h = max(3.5 * n_rows + 1, 5)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    # Map source -> column index
    src_to_col = {src: i for i, src in enumerate(sources)}
    row_counters = defaultdict(int)

    print(f"\nRunning inference...")
    total_dets = 0

    for sample in selected:
        src = sample["source"]
        col = src_to_col[src]
        row = row_counters[src]
        row_counters[src] += 1

        img_rgb, dets = run_inference(model, sample["image_path"], args.threshold)
        ax = axes[row][col]

        if img_rgb is None:
            ax.text(
                0.5,
                0.5,
                "Failed to load",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            continue

        draw_sample(ax, img_rgb, dets, sample, args.threshold)
        total_dets += len(dets)

    # Hide unused axes
    for src, col in src_to_col.items():
        for row in range(row_counters[src], n_rows):
            axes[row][col].axis("off")

    split_label = args.split.upper()
    fig.suptitle(
        f"AFGNet v9 — {split_label} coverage ({len(sources)} sources, "
        f"{args.n} sample/source, threshold={args.threshold:.2f})\n"
        f"Yellow=GT dashed, Colors=Predictions — Total detections: {total_dets}",
        fontsize=9,
        y=1.01,
    )
    plt.tight_layout()

    # Save
    if args.save:
        save_path = Path(args.save)
    else:
        save_path = (
            run_dir / f"coverage_{args.split}_n{args.n}_thr{args.threshold:.2f}.png"
        )

    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {save_path}")
    print(f"Total detections: {total_dets} across {len(selected)} images")


if __name__ == "__main__":
    main()
