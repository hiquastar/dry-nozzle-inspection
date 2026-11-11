"""
bench_degradation.py — Robustness test under input degradations.

Methodology (self-consistency under degradation, NO GT needed):
  1. Run inference on the CLEAN test image -> baseline detections.
  2. Apply a degradation transform to the same image.
  3. Run inference on the degraded image -> degraded detections.
  4. Match degraded -> baseline by greedy IoU (same class_id required).
  5. Report:
       - Detection retention  (% baseline detections still found)
       - Score retention      (mean score_deg / score_clean for matched)
       - Angle drift          (mean |angle_clean - angle_deg| for matched spray)

Degradation axes (each tested at multiple levels):
  - input_resolution : downsample input to {640, 512, 384, 256, 128} then upscale to 640
  - motion_blur      : Gaussian blur sigma {0, 2, 5, 10, 15} px
  - low_light        : gamma {1.0, 0.7, 0.5, 0.3} (lower = darker)
  - jpeg             : JPEG quality {95, 75, 50, 25, 10}
  - occlusion        : random box covering {0%, 10%, 20%, 30%, 50%} of image area

Operational envelope = highest degradation where detection retention ≥ 0.80.

Usage:
    python bench_degradation.py                  # default 200 test images
    python bench_degradation.py --n 100          # quicker
    python bench_degradation.py --axes resolution,blur   # subset of axes
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))
    sys.path.insert(0, str(_PARENT.parent / "structure"))

from dataset import resolve_image_path
from metrics import box_iou, compute_spray_angle, decode_predictions
from model import AFGNet

from config import CLASS_NAMES, IMG_SIZE, NMS_IOU_THRESH, REPO_ROOT, TEST_JSON

CANONICAL_RUN = "v9_fixes_reverse_seed42"
CKPT_PATH = REPO_ROOT / "train" / "model_1" / "checkpoint" / CANONICAL_RUN / "train_v9_v1.pth"


# -----------------------------------------------------------------------------
# Degradation transforms (operate on RGB uint8 numpy arrays, return RGB uint8)
# -----------------------------------------------------------------------------
def degrade_resolution(img, target_size):
    """Downsample to target_size then upscale back to original."""
    h, w = img.shape[:2]
    if target_size >= max(h, w):
        return img
    small = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def degrade_blur(img, sigma):
    if sigma <= 0:
        return img
    k = max(3, int(sigma * 6) | 1)  # odd kernel size
    return cv2.GaussianBlur(img, (k, k), sigma)


def degrade_gamma(img, gamma):
    """gamma < 1.0 = darker."""
    if abs(gamma - 1.0) < 1e-3:
        return img
    inv = 1.0 / max(gamma, 1e-3)
    table = ((np.arange(256) / 255.0) ** inv * 255.0).astype(np.uint8)
    return cv2.LUT(img, table)


def degrade_jpeg(img, quality):
    if quality >= 100:
        return img
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return img
    bgr2 = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)


def degrade_occlusion(img, pct, rng):
    """Paint a random gray box covering pct of image area."""
    if pct <= 0:
        return img
    h, w = img.shape[:2]
    area = h * w * pct
    bw = int(np.sqrt(area))
    bh = bw
    if bw >= w or bh >= h:
        return img
    out = img.copy()
    x = rng.randint(0, w - bw)
    y = rng.randint(0, h - bh)
    out[y : y + bh, x : x + bw] = 128
    return out


# -----------------------------------------------------------------------------
# Matching baseline ↔ degraded detections
# -----------------------------------------------------------------------------
def match_dets(base_dets, deg_dets, iou_thresh=0.5):
    """Greedy match degraded -> baseline by IoU within same class.
    Returns list of (base_idx, deg_idx, iou)."""
    if not base_dets or not deg_dets:
        return []
    used_deg = set()
    matches = []
    # Sort base by score desc, match each to best degraded
    base_sorted = sorted(range(len(base_dets)), key=lambda i: -base_dets[i]["score"])
    for bi in base_sorted:
        bd = base_dets[bi]
        best_di, best_iou = -1, 0.0
        for di, dd in enumerate(deg_dets):
            if di in used_deg or dd["class_id"] != bd["class_id"]:
                continue
            iou = box_iou(bd["box"], dd["box"])
            if iou > best_iou:
                best_iou = iou
                best_di = di
        if best_di >= 0 and best_iou >= iou_thresh:
            used_deg.add(best_di)
            matches.append((bi, best_di, best_iou))
    return matches


def summarize(base_per_img, deg_per_img):
    """Compute aggregate retention / score / angle drift over all images."""
    n_base = 0
    n_match = 0
    score_ratios = []
    angle_drifts = []
    # Per-class retention
    per_class_base = [0, 0, 0, 0]
    per_class_match = [0, 0, 0, 0]
    for base, deg in zip(base_per_img, deg_per_img):
        matches = match_dets(base, deg)
        n_base += len(base)
        n_match += len(matches)
        for bd in base:
            per_class_base[bd["class_id"]] += 1
        matched_base_idx = {m[0] for m in matches}
        for bi, di, _iou in matches:
            per_class_match[base[bi]["class_id"]] += 1
            score_ratios.append(deg[di]["score"] / max(base[bi]["score"], 1e-6))
            if base[bi]["class_id"] == 0:  # spray
                a_base = compute_spray_angle(base[bi]["keypoints"])
                a_deg = compute_spray_angle(deg[di]["keypoints"])
                if a_base is not None and a_deg is not None:
                    angle_drifts.append(abs(a_base - a_deg))
    retention = n_match / max(n_base, 1)
    per_class_ret = [per_class_match[i] / max(per_class_base[i], 1) for i in range(4)]
    return {
        "n_base_detections": n_base,
        "n_matched": n_match,
        "retention": retention,
        "per_class_retention": per_class_ret,
        "mean_score_ratio": (
            float(np.mean(score_ratios)) if score_ratios else float("nan")
        ),
        "mean_angle_drift_deg": (
            float(np.mean(angle_drifts)) if angle_drifts else float("nan")
        ),
        "n_spray_with_angle": len(angle_drifts),
    }


# -----------------------------------------------------------------------------
# Inference helper (operates on RGB uint8 numpy array)
# -----------------------------------------------------------------------------
class Inferer:
    def __init__(self, model, device, threshold):
        self.model = model
        self.device = device
        self.threshold = threshold
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def __call__(self, rgb_uint8):
        tensor = self.transform(rgb_uint8).unsqueeze(0).to(self.device)
        preds = self.model(tensor)
        return decode_predictions(
            preds,
            batch_idx=0,
            obj_thresh=self.threshold,
            nms_iou=NMS_IOU_THRESH,
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def load_test_images(n, seed):
    with open(TEST_JSON) as f:
        coco = json.load(f)
    rng = random.Random(seed)
    images = list(coco["images"])
    rng.shuffle(images)
    paths = []
    for img in images:
        try:
            p = resolve_image_path(img["file_name"], img.get("source_video_hint"))
            paths.append(p)
        except FileNotFoundError:
            continue
        if len(paths) >= n:
            break
    return paths


def load_model(device):
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    state = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
    state.pop("head.cell_x", None)
    state.pop("head.cell_y", None)
    model = AFGNet(pretrained=False)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Number of test images")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument(
        "--axes", type=str, default="resolution,blur,gamma,jpeg,occlusion"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--out", type=str, default=str(_HERE / "bench_degradation_results.json")
    )
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    axes = [a.strip() for a in args.axes.split(",")]
    rng = random.Random(args.seed)

    print("=" * 70)
    print(f"AFGNet v9 — Degradation robustness benchmark")
    print("=" * 70)
    print(f"Device : {device}")
    print(f"N imgs : {args.n}")
    print(f"Axes   : {axes}")
    print(f"Thresh : {args.threshold}")

    # Load images
    print("\nResolving image paths...")
    paths = load_test_images(args.n, args.seed)
    print(f"Loaded {len(paths)} test image paths")

    # Load + cache all images as RGB uint8
    print("Caching images...")
    cache = []
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cache.append(rgb)
    print(f"Cached {len(cache)} images in memory")

    # Load model
    model = load_model(device)
    infer = Inferer(model, device, args.threshold)

    # Baseline inference (clean images)
    print("\nRunning baseline inference (clean)...")
    t0 = time.time()
    base_per_img = [infer(rgb) for rgb in cache]
    print(
        f"  Done in {time.time()-t0:.1f}s. "
        f"Avg {np.mean([len(b) for b in base_per_img]):.2f} dets/img"
    )

    # Degradation levels per axis
    LEVELS = {
        "resolution": [
            ("640", None),
            ("512", 512),
            ("384", 384),
            ("256", 256),
            ("128", 128),
        ],
        "blur": [("s=0", 0), ("s=2", 2), ("s=5", 5), ("s=10", 10), ("s=15", 15)],
        "gamma": [("g=1.0", 1.0), ("g=0.7", 0.7), ("g=0.5", 0.5), ("g=0.3", 0.3)],
        "jpeg": [
            ("q=100", 100),
            ("q=75", 75),
            ("q=50", 50),
            ("q=25", 25),
            ("q=10", 10),
        ],
        "occlusion": [
            ("0%", 0.0),
            ("10%", 0.10),
            ("20%", 0.20),
            ("30%", 0.30),
            ("50%", 0.50),
        ],
    }

    results = {
        "baseline_avg_dets_per_img": float(np.mean([len(b) for b in base_per_img]))
    }

    for axis in axes:
        if axis not in LEVELS:
            print(f"WARNING: unknown axis {axis}, skipping")
            continue
        print(f"\n--- Axis: {axis} ---")
        axis_rows = []
        for label, param in LEVELS[axis]:
            t0 = time.time()
            deg_per_img = []
            for rgb in cache:
                if axis == "resolution":
                    if param is None:
                        deg = rgb
                    else:
                        deg = degrade_resolution(rgb, param)
                elif axis == "blur":
                    deg = degrade_blur(rgb, param)
                elif axis == "gamma":
                    deg = degrade_gamma(rgb, param)
                elif axis == "jpeg":
                    deg = degrade_jpeg(rgb, param)
                elif axis == "occlusion":
                    deg = degrade_occlusion(rgb, param, rng)
                deg_per_img.append(infer(deg))
            stats = summarize(base_per_img, deg_per_img)
            stats["label"] = label
            stats["param"] = param if isinstance(param, (int, float)) else None
            stats["n_deg_detections"] = sum(len(d) for d in deg_per_img)
            axis_rows.append(stats)
            print(
                f'  {label:>8}  retention={stats["retention"]:.3f}  '
                f'score_ratio={stats["mean_score_ratio"]:.3f}  '
                f'angle_drift={stats["mean_angle_drift_deg"]:.2f}°  '
                f"({time.time()-t0:.1f}s)"
            )
        results[axis] = axis_rows

    # ── Print summary table ──────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("DEGRADATION SUMMARY")
    print("=" * 70)
    for axis in axes:
        if axis not in results:
            continue
        print(f"\n{axis.upper()}")
        print(
            f'  {"level":<8}  {"retention":>10}  {"score_ratio":>12}  '
            f'{"angle_drift":>12}  {"per-class retention (sp/ns/fa/big)":>30}'
        )
        for row in results[axis]:
            pcr = row["per_class_retention"]
            print(
                f'  {row["label"]:<8}  '
                f'{row["retention"]:>10.3f}  '
                f'{row["mean_score_ratio"]:>12.3f}  '
                f'{row["mean_angle_drift_deg"]:>11.2f}°  '
                f"   {pcr[0]:.2f}/{pcr[1]:.2f}/{pcr[2]:.2f}/{pcr[3]:.2f}"
            )

    # ── Operational envelope ─────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("OPERATIONAL ENVELOPE  (retention >= 0.80)")
    print("=" * 70)
    for axis in axes:
        if axis not in results:
            continue
        ok_levels = [r["label"] for r in results[axis] if r["retention"] >= 0.80]
        print(f'  {axis:<12}: {", ".join(ok_levels) if ok_levels else "NONE"}')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
