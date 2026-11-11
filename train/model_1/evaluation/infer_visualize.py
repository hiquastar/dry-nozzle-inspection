"""
infer_visualize.py - Run v9 model on images and draw predictions

Visualizes:
  - Bounding boxes (color-coded by class)
  - Spray keypoints (3 dots: tip, left, right)
  - Spray cone angle (computed from keypoints)
  - Observability score (1.000 = measurable, lower = uncertain)
  - Class label + confidence

Usage:
    python infer_visualize.py                          # latest run, 8 random test images
    python infer_visualize.py --n 16                   # 16 images
    python infer_visualize.py --split val              # use val instead of test
    python infer_visualize.py --image path/to/img.jpg  # single specific image
    python infer_visualize.py --run train/model_1/checkpoint/v9_xxx        # specific run
    python infer_visualize.py --threshold 0.3          # detection score threshold
"""

import argparse
import json
import random
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

# Class colors (BGR for cv2 → RGB tuple for matplotlib)
CLASS_COLORS = {
    0: (0, 200, 0),  # nozzle_spray (green)
    1: (255, 165, 0),  # nozzle_no_spray (orange)
    2: (255, 50, 50),  # nozzle_facing (red)
    3: (50, 100, 255),  # big_nozzle (blue)
}
CLASS_LABELS = ["SPRAY", "NO_SPRAY", "FACING", "BIG"]


def _is_single_scale(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    return not any("smooth_p" in k for k in state.keys())


def find_latest_run(repo_root):
    """Find most recent single-scale v9 run directory."""
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
    """Load best model weights, auto-detecting architecture."""
    import importlib.util
    import sys as _sys

    run_dir = Path(run_dir)
    candidates = list(run_dir.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"Model not found: {run_dir}/*.pth")
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

    if isinstance(ckpt, dict) and "metrics" in ckpt:
        m = ckpt["metrics"]
        print(f"  Best epoch: {ckpt.get('epoch', '?')}")
        print(f"  Composite: {ckpt.get('composite', 0):.4f}")
        if "mAP_50" in m:
            print(f"  mAP@.5: {m['mAP_50']:.3f}")
        if "cls_f1" in m:
            print(f"  cls_f1: {m['cls_f1']:.3f}")
        obs_key = "observ_f1" if "observ_f1" in m else "obs_f1"
        if obs_key in m:
            print(f"  obs_f1: {m[obs_key]:.3f}")
        if "angle_mae_deg" in m:
            print(f"  angle_mae: {m['angle_mae_deg']:.2f}°")

    return model


def load_split_samples(json_path):
    """Load image list from a split JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    img_to_anns = {}
    for ann in data.get("annotations", []):
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    for img_info in data.get("images", []):
        anns = img_to_anns.get(img_info["id"], [])
        if not anns:
            continue
        try:
            src_hint = img_info.get("source_video", None)
            img_path = resolve_image_path(img_info["file_name"], src_hint)
            samples.append(
                {
                    "image_path": str(img_path),
                    "file_name": img_info["file_name"],
                    "orig_w": img_info.get("width", 0),
                    "orig_h": img_info.get("height", 0),
                    "anns": anns,
                }
            )
        except FileNotFoundError:
            continue

    return samples


def preprocess_image(img_bgr, image_size=IMG_SIZE):
    """BGR → tensor."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((image_size, image_size), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(img_rgb).unsqueeze(0)


@torch.no_grad()
def run_inference(model, image_path, threshold=None):
    """Run model on one image, return decoded predictions."""
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None, None

    orig_h, orig_w = img_bgr.shape[:2]

    img_tensor = preprocess_image(img_bgr).to(DEVICE)
    preds = model(img_tensor)

    threshold = threshold if threshold is not None else OBJ_THRESH
    detections = decode_predictions(
        preds, batch_idx=0, obj_thresh=threshold, nms_iou=NMS_IOU_THRESH
    )

    return img_bgr, detections


def draw_predictions(img_bgr, detections, gt_anns=None, orig_w=None, orig_h=None):
    """Draw detections (and optionally GT) on image. Returns RGB image for plt."""
    img = img_bgr.copy()
    h, w = img.shape[:2]

    if orig_w is None:
        orig_w = w
    if orig_h is None:
        orig_h = h

    # Draw GT (thin yellow boxes)
    if gt_anns:
        for ann in gt_anns:
            bbox = ann.get("bbox", [])
            if len(bbox) < 4:
                continue
            try:
                x, y, bw, bh = [float(v) for v in bbox[:4]]
            except (ValueError, TypeError):
                continue
            # Scale from original coords to display
            sx = w / orig_w
            sy = h / orig_h
            x1 = int(x * sx)
            y1 = int(y * sy)
            x2 = int((x + bw) * sx)
            y2 = int((y + bh) * sy)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 1)  # yellow

    # Draw predictions
    for det in detections:
        x1n, y1n, x2n, y2n = det["box"]
        cls_id = det["class_id"]
        score = det["score"]
        obs = det["observability"]
        kpts = det["keypoints"]

        # To pixel coords (in displayed image)
        x1 = int(x1n * w)
        y1 = int(y1n * h)
        x2 = int(x2n * w)
        y2 = int(y2n * h)

        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        # cv2 uses BGR
        color_bgr = (color[2], color[1], color[0])

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)

        # Label
        label = CLASS_LABELS[cls_id]
        text = f"{label} {score:.2f}"

        if cls_id == 0:  # nozzle_spray - show kpts + angle
            tip_x = int(kpts[0] * w)
            tip_y = int(kpts[1] * h)
            left_x = int(kpts[2] * w)
            left_y = int(kpts[3] * h)
            right_x = int(kpts[4] * w)
            right_y = int(kpts[5] * h)

            # Spray cone lines (from tip)
            cv2.line(img, (tip_x, tip_y), (left_x, left_y), color_bgr, 1)
            cv2.line(img, (tip_x, tip_y), (right_x, right_y), color_bgr, 1)

            # Keypoint dots
            cv2.circle(img, (tip_x, tip_y), 4, (0, 255, 255), -1)  # tip yellow
            cv2.circle(img, (left_x, left_y), 3, (255, 0, 255), -1)  # magenta
            cv2.circle(img, (right_x, right_y), 3, (255, 0, 255), -1)

            # Compute angle
            angle = compute_spray_angle(kpts)
            if angle is not None:
                text = f"{label} {score:.2f} | {angle:.1f}°"

        # Show observability for spray + facing classes
        if cls_id in (0, 2):
            text += f" | obs={obs:.2f}"

        # Text background
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color_bgr, -1)
        cv2.putText(
            img,
            text,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--n", type=int, default=8, help="Number of images")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Specific image path (overrides --split)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Detection score threshold (default {OBJ_THRESH})",
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Save plot to file (default: in run dir)"
    )
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Find run + load model
    run_dir = Path(args.run) if args.run else find_latest_run(REPO_ROOT)
    print(f"Run: {run_dir}")
    model = load_model(run_dir)

    # Get images to evaluate
    if args.image:
        # Single image mode
        samples = [
            {"image_path": args.image, "file_name": Path(args.image).name, "anns": []}
        ]
    else:
        split_path = {
            "train": TRAIN_JSON,
            "val": VAL_JSON,
            "test": TEST_JSON,
        }[args.split]

        if not split_path.exists():
            print(f"ERROR: {split_path} not found")
            return

        print(f"Split: {args.split} ({split_path.name})")
        all_samples = load_split_samples(split_path)
        print(f"Loaded {len(all_samples)} samples from split")

        if not all_samples:
            print("ERROR: No samples loaded")
            return

        random.seed(args.seed)
        n = min(args.n, len(all_samples))
        samples = random.sample(all_samples, n)

    # Run inference + draw
    print(f"\nRunning inference on {len(samples)} images...")

    cols = 4
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    n_dets_total = 0

    for idx, sample in enumerate(samples):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]

        img_bgr, detections = run_inference(
            model,
            sample["image_path"],
            threshold=args.threshold,
        )

        if img_bgr is None:
            ax.text(
                0.5,
                0.5,
                "Failed to load",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(sample["file_name"][:30], fontsize=9)
            ax.axis("off")
            continue

        n_dets_total += len(detections)

        # Resize image for display
        h, w = img_bgr.shape[:2]
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

        rgb = draw_predictions(
            img_bgr,
            detections,
            gt_anns=sample.get("anns", []),
            orig_w=sample.get("orig_w", img_bgr.shape[1]),
            orig_h=sample.get("orig_h", img_bgr.shape[0]),
        )
        ax.imshow(rgb)

        # Class breakdown
        cls_counts = {}
        for d in detections:
            cls_counts[CLASS_LABELS[d["class_id"]]] = (
                cls_counts.get(CLASS_LABELS[d["class_id"]], 0) + 1
            )

        cls_str = (
            ", ".join(f"{k}:{v}" for k, v in sorted(cls_counts.items()))
            if cls_counts
            else "no detections"
        )
        title = f'{sample["file_name"][:35]}\n{cls_str}'
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Hide empty subplots
    for idx in range(len(samples), rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis("off")

    threshold_str = args.threshold if args.threshold is not None else OBJ_THRESH
    suptitle = (
        f"AFGNet v9 — {args.split} predictions ({len(samples)} images, "
        f"{n_dets_total} total detections, threshold={threshold_str})\n"
        f"Yellow=GT, Colors=Predictions, Cyan dot=tip, Magenta=cone keypoints"
    )
    fig.suptitle(suptitle, fontsize=11, y=1.0)
    plt.tight_layout()

    # Save
    if args.save:
        save_path = Path(args.save)
    else:
        save_path = run_dir / f"inference_{args.split}_n{len(samples)}.png"

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    print(f"\nSaved: {save_path}")
    print(f"Total detections across {len(samples)} images: {n_dets_total}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
